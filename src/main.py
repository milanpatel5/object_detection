import time
import warnings

from dateutil.relativedelta import relativedelta
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, AveragePooling2D, Add


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


class MobileNetV2():
    def __call__(self, input):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        x = InvertedResidualBlock(t=1, c=16, n=1, s=1)(x)
        x = InvertedResidualBlock(t=6, c=24, n=2, s=2)(x)
        x = InvertedResidualBlock(t=6, c=32, n=3, s=2)(x)
        x = InvertedResidualBlock(t=6, c=64, n=4, s=2)(x)
        x = InvertedResidualBlock(t=6, c=96, n=3, s=1)(x)
        x = InvertedResidualBlock(t=6, c=160, n=3, s=2)(x)
        x = InvertedResidualBlock(t=6, c=320, n=1, s=1)(x)

        x = Conv2D(filters=1280, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)
        x = AveragePooling2D(pool_size=(7, 7))(x)
        return x


def prep_model():
    input = Input(shape=(224, 224, 3))
    x = MobileNetV2()(input)
    return Model(input, x)


def main():
    model = prep_model()
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
