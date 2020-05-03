import time
import warnings

from dateutil.relativedelta import relativedelta
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Add
from tensorflow.python.keras.models import Model


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

    def __call__(self, base_1, base_2, conv_1, conv_2, conv_3, conv_4, conv_5):
        return base_1
        """
        Under construction
        base_1_cla = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(base_1)
        base_1_cla = Reshape(target_shape=(base_1_cla.shape[1], base_1_cla.shape[2], self.n_boxes, self.n_classes))(base_1_cla)
        base_1_cla = Softmax(axis=3)(base_1_cla)
        base_1_loc = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(base_1)
        base_1_loc = Reshape(target_shape=(base_1_loc.shape[1], base_1_loc.shape[2], self.n_boxes, 4))

        base_2_cla = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(base_2)
        base_2_loc = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(base_2)

        conv_1_cla = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(conv_1)
        conv_1_loc = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(conv_1)

        conv_2_cla = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(conv_2)
        conv_2_loc = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(conv_2)

        conv_3_cla = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(conv_3)
        conv_3_loc = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(conv_3)

        conv_4_cla = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(conv_4)
        conv_4_loc = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(conv_4)

        conv_5_cla = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(conv_5)
        conv_5_loc = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(conv_5)
        """


class SingleShotDetector:
    def __init__(self):
        pass

    def __call__(self, base_1, base_2):
        conv_layers = MultiScaleFeatureMaps()(base_2)
        x = Predictors(n_boxes=4, n_classes=100)(base_1, base_2, *conv_layers)
        return x


def main():
    input = Input(shape=(512, 512, 3))

    base_1, base_2 = MobileNetV2()(input)
    output = SingleShotDetector()(base_1, base_2)

    model = Model(input, output)
    model.compile()
    model.summary()


if __name__ == '__main__':
    # os.nice(2)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=(time.time() - start_time))
    print('\n\nTime taken: ' +
          (' '.join(
              '{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
