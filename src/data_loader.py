import json
import random
from math import ceil

import numpy
from cv2 import cv2
from tensorflow.python.keras.utils.data_utils import Sequence


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
                self.train_dataset.append((image_path, self.model.encode_input(numpy.array(ground_truth_boxes, copy=False))))

            if shuffle:
                random.shuffle(self.train_dataset)

    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            random.shuffle(self.train_dataset)
        x = []
        y = []
        for data in self.train_dataset[self.batch_size * index:self.batch_size * (index + 1)]:
            x.append(cv2.imread(data[0]))
            y.append(data[1])
        return numpy.array(x, copy=False), numpy.array(y, copy=False)

    def __len__(self):
        return ceil(len(self.train_dataset) / self.batch_size)
