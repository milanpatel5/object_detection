import os
import time
import warnings

from dateutil.relativedelta import relativedelta

from data_loader import DataLoader
from model import prepare_model


def main():
    model, ssd = prepare_model()
    model.load_weights('saved_model.h5')

    for idx, (image, temp) in enumerate(DataLoader(ssd, batch_size=1, file='datasets/test/annotations.json', shuffle=False)):
        predictions = model.predict(x=image)
        classes, scores, boxes = ssd.decode_output(predictions[0])
        # classes, scores, boxes = ssd.decode_output(temp[0, :, 1:])
        ssd.plot_boxes(image[0], classes=classes, scores=scores, boxes=boxes, file_name='output/' + str(idx) + '.png', visualize=False)


if __name__ == '__main__':
    os.nice(2)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=(time.time() - start_time))
    print('\n\nTime taken: ' + (' '.join('{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
