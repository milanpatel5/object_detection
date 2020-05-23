import json
import os
import time
import warnings
from math import ceil
from random import shuffle

import cv2
import numpy
from dateutil.relativedelta import relativedelta
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.data_utils import Sequence

from model import SingleShotDetector, MobileNetV2


def main():
    """
    Main method for experimentation
    :return: Nothing(0)
    """
    # Input tensor
    image_shape = (512, 512, 3)
    input = Input(shape=image_shape)

    # Initialize SSD class object
    ssd = SingleShotDetector(image_shape=image_shape, n_classes=10)

    # Fetch features from base model and then pass them to SSD model for predictions
    base_1, base_2 = MobileNetV2()(input)
    output = ssd(base_1, base_2)

    # Creation of model object based on input tensor and output tensor from SSD
    model = Model(input, output)
    # Compile model with specified loss method
    model.compile(optimizer=Adam(), loss=ssd.loss_fn, metrics=[ssd.accuracy_fn])
    # Printing model summary to stdout
    # model.summary()
    # plot_model(model=model, show_shapes=True, expand_nested=True, dpi=96, to_file='model.png')

    # sample training
    model.fit(x=DataLoader(ssd, batch_size=8), epochs=10, callbacks=[ModelCheckpoint(filepath='saved_model.h5', monitor='accuracy_fn', save_best_only=True, verbose=1)])


class DataLoader(Sequence):
    def __init__(self, model, batch_size=4):
        self.model = model
        self.batch_size = batch_size
        self.image_shape = self.model.image_shape
        with open('datasets/train/annotations.json') as data_file:
            self.train_dataset = json.load(data_file)
            shuffle(self.train_dataset)

    def __getitem__(self, index):
        x = []
        y = []
        for train_data in self.train_dataset[index * self.batch_size: (index + 1) * self.batch_size]:
            image_path = train_data['path']
            ground_truth_boxes = []
            for obj in train_data['annotations']:
                loc = obj['coordinates']
                label = obj['label']
                cy = loc['y'] / self.image_shape[0]
                cx = loc['x'] / self.image_shape[1]
                h = loc['height'] / self.image_shape[0]
                w = loc['width'] / self.image_shape[1]
                ground_truth_boxes.append([cy, cx, h, w, int(label)])
            x.append(cv2.imread(image_path))
            y.append(self.model.encode_input(numpy.array(ground_truth_boxes)))
        return numpy.array(x), numpy.array(y)

    def __len__(self):
        return ceil(len(self.train_dataset) / self.batch_size)


if __name__ == '__main__':
    os.nice(2)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=(time.time() - start_time))
    print('\n\nTime taken: ' +
          (' '.join(
              '{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
