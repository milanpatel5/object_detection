import numpy
from tensorflow.python import Constant, Min, Max
from tensorflow.python.keras.backend import log
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Add, Reshape, Softmax, Concatenate, Average
from tensorflow.python.keras.losses import huber_loss, categorical_crossentropy


class BottleneckBlock:
    def __init__(self, t, c, s, residual=False):
        """
        bottleneck module of MobileNetV2
        :param t: channel expansion factor
        :param c: output channels
        :param s: strides
        :param residual: is residual operation required
        """
        self.t = t
        self.c = c
        self.s = s
        self.r = residual

    def __call__(self, input):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param input: input tensor of size (batches, H, W, C)
        :return: output tensor of size (batches, H/s, W/s, c)
        """
        # expansion
        x = Conv2D(filters=input.shape[3] * self.t, kernel_size=(1, 1), padding='same')(input)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        # real Convolution operations
        x = SeparableConv2D(filters=input.shape[3] * self.t, kernel_size=(3, 3), strides=self.s, padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        # bottleneck
        x = Conv2D(filters=self.c, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        # adding input residual
        if self.r:
            x = Add()([input, x])
        return x


class InvertedResidualBlock:
    def __init__(self, t, c, n, s):
        """
        InvertedResidualBlock: the unit module of MobileNetV2
        :param t: channel expansion factor
        :param c: output channels
        :param s: strides
        :param n: repetition count
        """
        self.t = t
        self.c = c
        self.n = n
        self.s = s

    def __call__(self, input):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param input: input tensor of size (batches, H, W, C)
        :return: output tensor of size (batches, H/s, W/s, c)
        """

        # First repetition without residual operation and actual strides
        x = BottleneckBlock(self.t, self.c, self.s)(input)

        # remaining repetitions with residual operation and with stride=1
        for _ in range(self.n - 1):
            x = BottleneckBlock(self.t, self.c, 1, True)(x)
        return x


class MobileNetV2:
    def __call__(self, input):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param input: input tensor of size (batches, H, W, C)
        :return: output two feature tensors from last two group of layers
        """
        # Initial Conv layer
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        # stack of InvertedResidual blocks
        x = InvertedResidualBlock(t=1, c=16, n=1, s=1)(x)
        x = InvertedResidualBlock(t=6, c=24, n=2, s=2)(x)
        x = InvertedResidualBlock(t=6, c=32, n=3, s=2)(x)
        x = InvertedResidualBlock(t=6, c=64, n=4, s=2)(x)
        x = InvertedResidualBlock(t=6, c=96, n=3, s=1)(x)
        x1 = InvertedResidualBlock(t=6, c=160, n=3, s=2)(x)
        x2 = InvertedResidualBlock(t=6, c=320, n=1, s=1)(x1)

        # output of last two InvertedResidual blocks
        return x1, x2


class MultiScaleFeatureMaps:
    def __init__(self):
        pass

    def __call__(self, input):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param input: input tensor of size (batches, H, W, C)
        :return: output five feature tensors from all feature extraction layers of different feature sizes
        """
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
        """
        Predictor module of SSD architecture
        :param n_boxes: number of default boxes per location on feature map
        :param n_classes: number of classes
        """
        self.n_boxes = n_boxes
        self.n_classes = n_classes

        # Fixed set of default boxes with different scales and aspect ratios (Other random generation method can be used)
        self.default_boxes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2), (1, 2), (2, 1), (2, 2)][:n_boxes]

    def __call__(self, *features):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param features: all the feature maps collected from MultiScaleFeatureMaps generation module and also from base module
        :return: prediction tensor with size: (batches, total_boxes, n_classes+8)
        """
        predictions = []
        for feature in features:
            # Classification confidence with softmax operation
            class_conf = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(feature)
            class_conf = Reshape(target_shape=(-1, self.n_classes))(class_conf)
            class_conf = Softmax()(class_conf)

            # Localization layers
            loc_var = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(feature)
            loc_var = Reshape(target_shape=(-1, 4))(loc_var)

            # Default boxes for feature-map reference
            loc_ref = self.make_boxes(feature.shape[0], feature.shape[1], feature.shape[2])

            # concatenation of all the prediction features along last axis
            feature_predictions = Concatenate(axis=-1)([class_conf, loc_var, loc_ref])
            predictions.append(feature_predictions)

        # concatenation of all boxes
        predictions = Concatenate(axis=1)(predictions)
        return predictions

    def make_boxes(self, batch_size, feature_height, feature_width):
        """
        Create AnchorBox for feature-map reference
        :param batch_size:
        :param feature_height:
        :param feature_width:
        :return: Anchor boxes in shape (batch_size, feature_height * feature_width * self.n_boxes, 4)
        """
        boxes = []
        for y in range(feature_height):
            for x in range(feature_width):
                for default_box in self.default_boxes:
                    cy = y / feature_height
                    cx = x / feature_width
                    h = (1 + default_box[0]) / feature_height
                    w = (1 + default_box[1]) / feature_width
                    boxes.append([cy, cx, h, w])
        boxes = numpy.array(boxes, copy=False, dtype=float)
        # broadcast them to all batches
        boxes = numpy.broadcast_to(boxes, shape=(batch_size if batch_size else 0, feature_height * feature_width * self.n_boxes, boxes.shape[1]))
        return boxes


class SingleShotDetector:
    def __init__(self, image_shape, class_conf_threshold=0.1, jaccard_similarity_threshold=0.5, mode='train', n_boxes=4, n_classes=100, loc_loss_weight=1):
        """
        SingleShotDetector model class
        :param image_shape: a tuple of image shape
        :param class_conf_threshold: Threshold value for classification confidence
        :param jaccard_similarity_threshold: Threshold value for jaccard similarity(Intersection Over Union)
        :param mode: Mention whether it is 'train' or 'valid' mode
        :param n_boxes: Number of default boxes per position of feature map
        :param n_classes: Number of classes
        :param loc_loss_weight: Weight(alpha) of localization loss
        """
        self.class_conf_threshold = Constant(class_conf_threshold)
        self.jaccard_similarity_threshold = Constant(jaccard_similarity_threshold)
        self.mode = mode
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        self.loc_loss_weight = loc_loss_weight
        self.image_shape = image_shape

    def __call__(self, base_1, base_2):
        # Extract features of varying scales from input of base model
        conv_layers = MultiScaleFeatureMaps()(base_2)

        # Prepare predictions from extracted features
        predictions = Predictors(n_boxes=self.n_boxes, n_classes=self.n_classes)(base_1, base_2, *conv_layers)

        if self.mode == 'train':
            # if it's training mode then return predictions as they are
            return predictions
        elif self.mode == 'valid':
            # In case of validation decode the predictions and apply Non-Maximum Suppression
            pass

    def loss_fn(self, y_true, y_pred):
        """
        Method to calculate loss during training
        :param y_true: tensor containing the ground truth boxes
        :param y_pred: tensor containing the default boxes and predicted boxes
        :return: mean loss value
        """
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
                # loss will be non-zero in case of active default box
                if loss != 0:
                    batch_loss_vec.append(loss)
            # Add losses of all the boxes in a batch
            batch_loss_vec = Add()(batch_loss_vec)[0]
            loss_vec.append(batch_loss_vec)
        # Mean of loss values over all batches
        loss = Average()(loss_vec)
        return loss[0]

    def calculate_loss(self, predicted_classes, predicted_box, default_box, ground_truth_boxes):
        """
        Method to calculate loss value for given default box and corresponding prediction
        :param predicted_classes: Output of softmax from classification conf module
        :param predicted_box: Predicted localization parameters
        :param default_box: Localization parameters of default box for reference
        :param ground_truth_boxes: All ground truth boxes
        :return:
        """
        # Scaled (cy, cx, h, w) parameters of predicted box based on input image dimensions
        l_cy = predicted_box[0] * self.image_shape[0]
        l_cx = predicted_box[1] * self.image_shape[1]
        l_h = predicted_box[2] * self.image_shape[0]
        l_w = predicted_box[3] * self.image_shape[1]

        for g_box in ground_truth_boxes:
            # Scaled (cy, cx, h, w) parameters of ground_truth box based on input image dimensions
            g_cy = g_box[0] * self.image_shape[0]
            g_cx = g_box[1] * self.image_shape[1]
            g_h = g_box[2] * self.image_shape[0]
            g_w = g_box[3] * self.image_shape[1]

            # Check whether ground_truth box and default_box match
            if self.intersection_over_union((l_cy - 0.5 * l_h, l_cy + 0.5 * l_h, l_cx - 0.5 * l_w, l_cx + 0.5 * l_w),
                                            (g_cy - 0.5 * g_h, g_cy + 0.5 * g_h, g_cx - 0.5 * g_w, g_cx + 0.5 * g_w)) > self.jaccard_similarity_threshold:
                # Reshape to add batch dimension
                default_box = Reshape(target_shape=(1, 4))(default_box)

                # Prepare G^ vector as mentioned in the paper
                y_true = Reshape(target_shape=(1, 4))(g_box)
                y_true[..., :2] = (y_true[..., :2] - default_box[..., :2]) / default_box[..., 2:]
                y_true[..., 2:] = log(y_true[..., 2:] / default_box[..., 2:])
                loc_loss = Add()(huber_loss(y_true=y_true, y_pred=Reshape(target_shape=predicted_box.shape)(predicted_box)))

                # Reshape to add batch dimension
                predicted_classes = Reshape(target_shape=(1, self.n_classes))(predicted_classes)
                # Prepare y_true vector based on actual classification category of ground_truth box
                y_true = numpy.zeros(shape=predicted_classes.shape)
                y_true[..., g_box[4]] = 1

                classification_loss = categorical_crossentropy(y_true=y_true, y_pred=predicted_classes)

                # Calculation of weighted total loss
                return self.loc_loss_weight * loc_loss + classification_loss
        return 0

    @staticmethod
    def intersection_over_union(predicted_box, ground_truth_box):
        """
        Method for Jaccard Similarity calculation
        :param predicted_box:
        :param ground_truth_box:
        :return:
        """
        inter_y_min = Max([predicted_box[0], ground_truth_box[0]])
        inter_y_max = Min([predicted_box[1], ground_truth_box[1]])
        inter_x_min = Max([predicted_box[2], ground_truth_box[2]])
        inter_x_max = Min([predicted_box[3], ground_truth_box[3]])

        union_y_min = Min([predicted_box[0], ground_truth_box[0]])
        union_y_max = Max([predicted_box[1], ground_truth_box[1]])
        union_x_min = Min([predicted_box[2], ground_truth_box[2]])
        union_x_max = Max([predicted_box[3], ground_truth_box[3]])

        return ((inter_y_max - inter_y_min) * (inter_x_max - inter_x_min)) / ((union_y_max - union_y_min) * (union_x_max - union_x_min))
