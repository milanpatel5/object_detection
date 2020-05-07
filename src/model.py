import numpy
from tensorflow.python import Constant, Min, Max
from tensorflow.python.keras.backend import log
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Add, Reshape, Softmax, Concatenate, Average
from tensorflow.python.keras.losses import huber_loss, categorical_crossentropy


class BottleneckBlock:
    def __init__(self, t, c, s, residual=False):
        self.t = t
        self.c = c
        self.s = s
        self.r = residual

    def __call__(self, input):
        x = Conv2D(filters=input.shape[3] * self.t, kernel_size=(1, 1), padding='same')(input)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        x = SeparableConv2D(filters=input.shape[3] * self.t, kernel_size=(3, 3), strides=self.s, padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        x = Conv2D(filters=self.c, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        if self.r:
            x = Add()([input, x])
        return x


class InvertedResidualBlock:
    def __init__(self, t, c, n, s):
        self.t = t
        self.c = c
        self.n = n
        self.s = s

    def __call__(self, input):
        x = BottleneckBlock(self.t, self.c, self.s)(input)
        for _ in range(self.n - 1):
            x = BottleneckBlock(self.t, self.c, 1, True)(x)
        return x


class MobileNetV2:
    def __call__(self, input):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        x = InvertedResidualBlock(t=1, c=16, n=1, s=1)(x)
        x = InvertedResidualBlock(t=6, c=24, n=2, s=2)(x)
        x = InvertedResidualBlock(t=6, c=32, n=3, s=2)(x)
        x = InvertedResidualBlock(t=6, c=64, n=4, s=2)(x)
        x = InvertedResidualBlock(t=6, c=96, n=3, s=1)(x)
        x1 = InvertedResidualBlock(t=6, c=160, n=3, s=2)(x)
        x2 = InvertedResidualBlock(t=6, c=320, n=1, s=1)(x1)
        return x1, x2


class MultiScaleFeatureMaps:
    def __init__(self):
        pass

    def __call__(self, input):
        conv1 = Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same')(input)
        conv1 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', padding='same')(conv1)

        conv2 = Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='same')(conv1)
        conv2 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(conv2)
        conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(conv3)

        conv4 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(conv3)
        conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)

        conv5 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(conv4)
        conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv5)

        return conv1, conv2, conv3, conv4, conv5


class Predictors:
    def __init__(self, n_boxes, n_classes):
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        self.default_boxes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2), (1, 2), (2, 1), (2, 2)][:n_boxes]

    def __call__(self, *features):
        predictions = []
        for feature in features:
            class_conf = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(feature)
            class_conf = Reshape(target_shape=(-1, self.n_classes))(class_conf)
            class_conf = Softmax()(class_conf)

            loc_var = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(feature)
            loc_var = Reshape(target_shape=(-1, 4))(loc_var)
            loc_ref = self.make_boxes(feature.shape[0], feature.shape[1], feature.shape[2])

            feature_predictions = Concatenate(axis=-1)([class_conf, loc_var, loc_ref])
            predictions.append(feature_predictions)

        predictions = Concatenate(axis=1)(predictions)
        return predictions

    def make_boxes(self, batch_size, feature_height, feature_width):
        boxes = []
        for y in range(feature_height):
            for x in range(feature_width):
                for default_box in self.default_boxes:
                    boxes.append([feature_height, feature_width, y, x, 1 + default_box[0], 1 + default_box[1]])
        boxes = numpy.array(boxes, copy=False, dtype=float)
        boxes = numpy.broadcast_to(boxes, shape=(batch_size if batch_size else 0, feature_height * feature_width * self.n_boxes, boxes.shape[1]))
        return boxes


class SingleShotDetector:
    def __init__(self, image_shape, class_conf_threshold=0.1, jaccard_similarity_threshold=0.5, mode='train', n_boxes=4, n_classes=100, loc_loss_weight=1):
        self.class_conf_threshold = Constant(class_conf_threshold)
        self.jaccard_similarity_threshold = Constant(jaccard_similarity_threshold)
        self.mode = mode
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        self.loc_loss_weight = loc_loss_weight
        self.image_shape = image_shape

    def __call__(self, base_1, base_2):
        conv_layers = MultiScaleFeatureMaps()(base_2)
        predictions = Predictors(n_boxes=self.n_boxes, n_classes=self.n_classes)(base_1, base_2, *conv_layers)
        if self.mode == 'train':
            return predictions

    def loss_fn(self, y_true, y_pred):
        loss_vec = []
        # separate loss calculation for all batches
        for ground_truth_boxes, predicted_and_default_boxes in zip(y_true, y_pred):
            batch_loss_vec = []
            # processing each predicted box
            for predicted_and_default_box in predicted_and_default_boxes:
                loss = self.calculate_loss(predicted_and_default_box[:self.n_classes],
                                           predicted_and_default_box[self.n_classes:self.n_classes + 4],
                                           predicted_and_default_box[self.n_classes + 4:],
                                           ground_truth_boxes[..., 1:])
                if loss != 0:
                    batch_loss_vec.append(loss)
            batch_loss_vec = Add()(batch_loss_vec)[0]
            loss_vec.append(batch_loss_vec)
        loss = Average()(loss_vec)
        return loss[0]

    def calculate_loss(self, predicted_classes, predicted_box, default_box, ground_truth_boxes):
        l_cy = predicted_box[0] * self.image_shape[0]
        l_cx = predicted_box[1] * self.image_shape[1]
        l_h = predicted_box[2] * self.image_shape[0]
        l_w = predicted_box[3] * self.image_shape[1]

        for g_box in ground_truth_boxes:
            g_cy = g_box[0] * self.image_shape[0]
            g_cx = g_box[1] * self.image_shape[1]
            g_h = g_box[2] * self.image_shape[0]
            g_w = g_box[3] * self.image_shape[1]

            if self.intersection_over_union((l_cy - 0.5 * l_h, l_cy + 0.5 * l_h, l_cx - 0.5 * l_w, l_cx + 0.5 * l_w),
                                            (g_cy - 0.5 * g_h, g_cy + 0.5 * g_h, g_cx - 0.5 * g_w, g_cx + 0.5 * g_w)) > self.jaccard_similarity_threshold:
                default_box = Reshape(target_shape=(1, 4))(default_box)
                y_true = Reshape(target_shape=(1, 4))(g_box)
                y_true[..., :2] = (y_true[..., :2] - default_box[..., :2]) / default_box[..., 2:]
                y_true[..., 2:] = log(y_true[..., 2:] / default_box[..., 2:])
                loc_loss = Add()(huber_loss(y_true=y_true, y_pred=Reshape(target_shape=predicted_box.shape)(predicted_box)))

                predicted_classes = Reshape(target_shape=(1, self.n_classes))(predicted_classes)
                y_true = numpy.zeros(shape=predicted_classes.shape)
                y_true[..., g_box[4]] = 1
                classification_loss = categorical_crossentropy(y_true=y_true, y_pred=predicted_classes)

                return self.loc_loss_weight * loc_loss + classification_loss
        return 0

    @staticmethod
    def intersection_over_union(predicted_box, ground_truth_box):
        inter_y_min = Max([predicted_box[0], ground_truth_box[0]])
        inter_y_max = Min([predicted_box[1], ground_truth_box[1]])
        inter_x_min = Max([predicted_box[2], ground_truth_box[2]])
        inter_x_max = Min([predicted_box[3], ground_truth_box[3]])

        union_y_min = Min([predicted_box[0], ground_truth_box[0]])
        union_y_max = Max([predicted_box[1], ground_truth_box[1]])
        union_x_min = Min([predicted_box[2], ground_truth_box[2]])
        union_x_max = Max([predicted_box[3], ground_truth_box[3]])

        return ((inter_y_max - inter_y_min) * (inter_x_max - inter_x_min)) / ((union_y_max - union_y_min) * (union_x_max - union_x_min))
