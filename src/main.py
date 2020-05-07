import time
import warnings

from dateutil.relativedelta import relativedelta
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model

from model import SingleShotDetector, MobileNetV2


def main():
    """
    Main method for experimentation
    :return: Nothing(0)
    """
    # Input tensor
    input = Input(shape=(512, 512, 3))

    # Initialize SSD class object
    ssd = SingleShotDetector(image_shape=input.shape)

    # Fetch features from base model and then pass them to SSD model for predictions
    base_1, base_2 = MobileNetV2()(input)
    output = ssd(base_1, base_2)

    # Creation of model object based on input tensor and output tensor from SSD
    model = Model(input, output)
    # Compile model with specified loss method
    model.compile(optimizer='adam', loss=ssd.loss_fn)
    # Printing model summary to stdout
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
