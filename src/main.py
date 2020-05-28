import json
import os
import random
import time
import warnings
from math import ceil

import numpy
from cv2 import cv2
from dateutil.relativedelta import relativedelta
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.utils.data_utils import Sequence

from model import SingleShotDetector, MobileNetV2


class DataLoader(Sequence):
    def __init__(self, model, file, batch_size=4, shuffle=True):
        self.model = model
        self.batch_size = batch_size
        self.image_shape = self.model.image_shape
        self.shuffle = shuffle
        self.train_dataset = []
        with open(file) as data_file:
            json_data = json.load(data_file)
            for train_data in json_data:
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
                self.train_dataset.append((cv2.imread(image_path), self.model.encode_input(numpy.array(ground_truth_boxes, copy=False))))

            if shuffle:
                random.shuffle(self.train_dataset)

    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            random.shuffle(self.train_dataset)
        x = []
        y = []
        for data in self.train_dataset[self.batch_size * index:self.batch_size * (index + 1)]:
            x.append(data[0])
            y.append(data[1])
        return numpy.array(x, copy=False), numpy.array(y, copy=False)

    def __len__(self):
        return ceil(len(self.train_dataset) / self.batch_size)


def main():
    """
    Main method for experimentation
    :return: Nothing(0)
    """
    # Input tensor
    image_shape = (512, 512, 3)
    input = Input(shape=image_shape)

    # Initialize SSD class object
    ssd = SingleShotDetector(image_shape=image_shape, n_classes=10, loc_loss_weight=0.75, class_conf_threshold=0.9)

    # Fetch features from base model and then pass them to SSD model for predictions
    base_1, base_2 = MobileNetV2()(input)
    output = ssd(base_1, base_2)

    # Creation of model object based on input tensor and output tensor from SSD
    model = Model(input, output)
    # Compile model with specified loss method
    model.compile(optimizer=RMSprop(centered=True), loss=ssd.loss_fn, metrics=[ssd.accuracy, ssd.precision, ssd.recall])
    # Printing model summary to stdout
    # model.summary()
    # plot_model(model=model, show_shapes=True, expand_nested=True, dpi=96, to_file='model.png')

    # sample training
    # model.load_weights('saved_model.h5')
    # model.fit(x=DataLoader(ssd, batch_size=8, file='datasets/train/annotations.json'), epochs=1000, initial_epoch=350,
    #           callbacks=[ModelCheckpoint(filepath='saved_model.h5', monitor='accuracy', save_best_only=False, save_weights_only=True, verbose=0)])

    # testing
    model.load_weights('saved_model.h5')
    for idx, (image, _) in enumerate(DataLoader(ssd, batch_size=1, file='datasets/test/annotations.json', shuffle=False)):
        predictions = model.predict(x=image)
        classes, scores, boxes = ssd.decode_output(predictions[0])
        ssd.plot_boxes(image[0], classes=classes, scores=scores, boxes=boxes, file_name='output/' + str(idx) + '.png', visualize=False)


if __name__ == '__main__':
    os.nice(2)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=(time.time() - start_time))
    print('\n\nTime taken: ' +
          (' '.join(
              '{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
