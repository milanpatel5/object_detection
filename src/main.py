import time
import warnings

from dateutil.relativedelta import relativedelta
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model

from model import SingleShotDetector, MobileNetV2


def main():
    input = Input(shape=(512, 512, 3))
    ssd = SingleShotDetector(image_shape=input.shape)

    base_1, base_2 = MobileNetV2()(input)
    output = ssd(base_1, base_2)

    model = Model(input, output)
    model.compile(optimizer='adam', loss=ssd.loss_fn)
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
