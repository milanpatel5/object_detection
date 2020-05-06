import numpy
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Add, Reshape, Softmax, Concatenate


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
                    cy = y / feature_height
                    cx = x / feature_width
                    h = (1 + default_box[0]) / feature_height
                    w = (1 + default_box[1]) / feature_width
                    boxes.append([cy, cx, h, w])
        boxes = numpy.array(boxes, copy=False, dtype=float)
        boxes = numpy.broadcast_to(boxes, shape=(batch_size if batch_size else 0, feature_height * feature_width * self.n_boxes, 4))
        return boxes


class SingleShotDetector:
    def __init__(self, class_conf_threshold=0.5, jaccard_similarity_threshold=0.5):
        self.class_conf_threshold = class_conf_threshold
        self.jaccard_similarity_threshold = jaccard_similarity_threshold

    def __call__(self, base_1, base_2):
        conv_layers = MultiScaleFeatureMaps()(base_2)
        x = Predictors(n_boxes=4, n_classes=100)(base_1, base_2, *conv_layers)
        return x
