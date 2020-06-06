import os
import time
import warnings

from dateutil.relativedelta import relativedelta
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data_loader import DataLoader
from model import prepare_model


def main():
    model_checkpoint_file = 'saved_model.h5'

    model, ssd = prepare_model()
    # model.load_weights(model_checkpoint_file)

    model.fit(x=DataLoader(ssd, batch_size=8, file='datasets/train/annotations.json'), epochs=100, initial_epoch=0,
              callbacks=[ModelCheckpoint(filepath=model_checkpoint_file, monitor='accuracy', save_best_only=True, save_weights_only=True, verbose=1)])


if __name__ == '__main__':
    os.nice(2)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=(time.time() - start_time))
    print('\n\nTime taken: ' + (' '.join('{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
