from random import random

import numpy
from matplotlib import pyplot
from tensorflow import int64
from tensorflow.python.framework import ops
from tensorflow.python.keras.backend import sum, mean, log, floatx, cast, equal, argmax, any
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Add, Reshape, Softmax, Concatenate, Multiply
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.ops import math_ops


class BottleneckBlock:
    def __init__(self, t, c, s, residual=False):
        """
        bottleneck module of MobileNetV2
        :param t: channel expansion factor
        :param c: output channels
        :param s: strides
        :param residual: is residual operation required
        """
        self.t = t
        self.c = c
        self.s = s
        self.r = residual

    def __call__(self, input):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param input: input tensor of size (batches, H, W, C)
        :return: output tensor of size (batches, H/s, W/s, c)
        """
        # expansion
        x = Conv2D(filters=input.shape[3] * self.t, kernel_size=(1, 1), padding='same')(input)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        # real Convolution operations
        x = SeparableConv2D(filters=input.shape[3] * self.t, kernel_size=(3, 3), strides=self.s, padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        # bottleneck
        x = Conv2D(filters=self.c, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        # adding input residual
        if self.r:
            x = Add()([input, x])
        return x


class InvertedResidualBlock:
    def __init__(self, t, c, n, s):
        """
        InvertedResidualBlock: the unit module of MobileNetV2
        :param t: channel expansion factor
        :param c: output channels
        :param s: strides
        :param n: repetition count
        """
        self.t = t
        self.c = c
        self.n = n
        self.s = s

    def __call__(self, input):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param input: input tensor of size (batches, H, W, C)
        :return: output tensor of size (batches, H/s, W/s, c)
        """
        # First repetition without residual operation and actual strides
        x = BottleneckBlock(self.t, self.c, self.s)(input)

        # remaining repetitions with residual operation and with stride=1
        for _ in range(self.n - 1):
            x = BottleneckBlock(self.t, self.c, 1, residual=True)(x)
        return x


class MobileNetV2:
    def __call__(self, input):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param input: input tensor of size (batches, H, W, C)
        :return: output two feature tensors from last two group of layers
        """
        # Initial Conv layer
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input)
        x = BatchNormalization(axis=3)(x)
        x = ReLU(max_value=6)(x)

        # stack of InvertedResidual blocks
        x = InvertedResidualBlock(t=1, c=16, n=1, s=1)(x)
        x = InvertedResidualBlock(t=6, c=24, n=2, s=2)(x)
        x = InvertedResidualBlock(t=6, c=32, n=3, s=2)(x)
        x = InvertedResidualBlock(t=6, c=64, n=4, s=2)(x)
        x = InvertedResidualBlock(t=6, c=96, n=3, s=1)(x)
        x1 = InvertedResidualBlock(t=6, c=160, n=3, s=2)(x)
        x2 = InvertedResidualBlock(t=6, c=320, n=1, s=1)(x1)

        # output of last two InvertedResidual blocks
        return x1, x2


class MultiScaleFeatureMaps:
    def __init__(self):
        pass

    def __call__(self, input):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param input: input tensor of size (batches, H, W, C)
        :return: output five feature tensors from all feature extraction layers of different feature sizes
        """
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
        """
        Predictor module of SSD architecture
        :param n_boxes: number of default boxes per location on feature map
        :param n_classes: number of classes
        """
        self.n_boxes = n_boxes
        self.n_classes = n_classes

    def __call__(self, features):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param features: all the feature maps collected from MultiScaleFeatureMaps generation module and also from base module
        :return: prediction tensor with size: (batches, total_boxes, n_classes+8)
        """
        predictions = []
        for feature in features:
            # Classification confidence with softmax operation
            class_conf = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(feature)
            class_conf = Reshape(target_shape=(feature.shape[1] * feature.shape[2] * self.n_boxes, self.n_classes))(class_conf)
            class_conf = Softmax()(class_conf)

            # Localization layers
            loc_var = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(feature)
            loc_var = Reshape(target_shape=(feature.shape[1] * feature.shape[2] * self.n_boxes, 4))(loc_var)

            # concatenation of all the prediction features along last axis
            feature_predictions = Concatenate(axis=-1)([class_conf, loc_var])
            predictions.append(feature_predictions)

        # concatenation of all boxes
        predictions = Concatenate(axis=1)(predictions)
        return predictions


class SingleShotDetector:
    def __init__(self, image_shape, class_conf_threshold=0.1, jaccard_similarity_threshold=0.5, n_boxes=4, n_classes=100, loc_loss_weight=1):
        """
        SingleShotDetector model class
        :param image_shape: a tuple of image shape
        :param class_conf_threshold: Threshold value for classification confidence
        :param jaccard_similarity_threshold: Threshold value for jaccard similarity(Intersection Over Union)
        :param n_boxes: Number of default boxes per position of feature map
        :param n_classes: Number of classes
        :param loc_loss_weight: Weight(alpha) of localization loss
        """
        self.class_conf_threshold = class_conf_threshold
        self.jaccard_similarity_threshold = jaccard_similarity_threshold
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        self.loc_loss_weight = loc_loss_weight
        self.image_shape = image_shape

        # Fixed set of default boxes with different scales and aspect ratios (Other random generation method can be used)
        self.default_box_scales = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 0), (0, 2), (2, 2)][:n_boxes]
        self.default_boxes = None
        self.accuracy_data = None
        self.precision_data = None
        self.recall_data = None

    def __call__(self, base_1, base_2):
        # Extract features of varying scales from input of base model
        conv_layers = MultiScaleFeatureMaps()(base_2)

        # Prepare predictions from extracted features
        features = base_1, base_2, *conv_layers
        predictions = Predictors(n_boxes=self.n_boxes, n_classes=self.n_classes)(features)

        # generate anchors of default boxes (only once)
        if self.default_boxes is None:
            self.default_boxes = []
            for feature in features:
                self.default_boxes.append(self.make_boxes(feature.shape[1], feature.shape[2]))
            self.default_boxes = numpy.concatenate(self.default_boxes, axis=1)

        return predictions

    def make_boxes(self, feature_height, feature_width):
        """
        Create AnchorBox for feature-map reference
        :param feature_height:
        :param feature_width:
        :return: Anchor boxes in shape (batch_size, feature_height * feature_width * self.n_boxes, 4)
        """
        boxes = numpy.zeros(shape=(feature_height, feature_width, self.n_boxes, 4))
        for y in range(feature_height):
            for x in range(feature_width):
                boxes[y, x, :, 0] = y / feature_height
                boxes[y, x, :, 1] = x / feature_width
                for i, default_box_scale in enumerate(self.default_box_scales):
                    boxes[y, x, i, 2] = (1 + default_box_scale[0]) / feature_height
                    boxes[y, x, i, 3] = (1 + default_box_scale[1]) / feature_width
        # include batch dimension
        boxes = boxes.reshape((1, feature_height * feature_width * self.n_boxes, 4))
        return boxes

    def loss_fn(self, y_true, y_pred):
        """
        Method to calculate loss during training
        :param y_true: tensor containing the Anchor boxes of corresponding ground truth boxes in shape(batch, total_boxes, n_classes + 5)
        :param y_pred: tensor containing the predicted boxes and default boxes in shape(batch, total_boxes, n_classes + 4)
        :return: mean loss value
        """
        # identifying matched anchor boxes
        is_match = sum(1 - y_true[:, :, :1], axis=2)
        n_matched_boxes = sum(is_match, axis=1)

        # calculating accuracy
        true_idx = argmax(y_true[:, :, :self.n_classes + 1])
        pred_idx = cast(any(y_pred[:, :, :self.n_classes] > self.class_conf_threshold, axis=-1), int64)
        pred_idx = pred_idx * (argmax(y_pred[:, :, :self.n_classes]) + 1)
        self.accuracy_data = cast(equal(true_idx, pred_idx), floatx())

        # calculating precision-recall
        true_positives = sum(cast(true_idx > 0, floatx()) * cast(pred_idx > 0, floatx()))
        self.precision_data = true_positives / sum(cast(pred_idx > 0, floatx()))
        self.recall_data = true_positives / sum(cast(true_idx > 0, floatx()))
        del true_idx, pred_idx, true_positives

        # calculating losses
        classification_loss = categorical_crossentropy(y_true=y_true[:, :, 1:self.n_classes + 1], y_pred=y_pred[:, :, :self.n_classes])
        localization_loss = Add()([smooth_l1_loss(y_true=y_true[:, :, self.n_classes + 1 + i], y_pred=y_pred[:, :, self.n_classes + i]) for i in range(4)])

        # we only want to calculate losses for matched anchor boxes
        localization_loss = Multiply()([is_match, localization_loss])

        # weighted total loss
        loss = classification_loss + self.loc_loss_weight * localization_loss

        # taking mean over matched anchor boxes
        loss = sum(loss, axis=1)
        loss = Multiply()([loss, n_matched_boxes])
        n_matched_boxes = Multiply()([n_matched_boxes, n_matched_boxes])
        n_matched_boxes += 1e-7
        loss = loss / n_matched_boxes

        # taking mean over batch
        loss = mean(loss, axis=0)
        return loss

    def accuracy(self, y_true, y_pred):
        return self.accuracy_data

    def precision(self, y_true, y_pred):
        return self.precision_data

    def recall(self, y_true, y_pred):
        return self.recall_data

    def decode_output(self, predictions):
        cy = (predictions[:, self.n_classes:self.n_classes + 1] * self.default_boxes[0][:, 2:3] + self.default_boxes[0][:, 0:1]) * self.image_shape[0]
        cx = (predictions[:, self.n_classes + 1:self.n_classes + 2] * self.default_boxes[0][:, 3:4] + self.default_boxes[0][:, 1:2]) * self.image_shape[1]
        h = numpy.exp(predictions[:, self.n_classes + 2:self.n_classes + 3]) * self.default_boxes[0][:, 2:3] * self.image_shape[0]
        w = numpy.exp(predictions[:, self.n_classes + 3:self.n_classes + 4]) * self.default_boxes[0][:, 3:4] * self.image_shape[1]
        # cy = self.default_boxes[0, :, 0:1] * self.image_shape[0]
        # cx = self.default_boxes[0, :, 1:2] * self.image_shape[1]
        # h = self.default_boxes[0, :, 2:3] * self.image_shape[0]
        # w = self.default_boxes[0, :, 3:4] * self.image_shape[1]
        predictions[:, self.n_classes:self.n_classes + 1] = cy - h / 2
        predictions[:, self.n_classes + 1:self.n_classes + 2] = cy + h / 2
        predictions[:, self.n_classes + 2:self.n_classes + 3] = cx - w / 2
        predictions[:, self.n_classes + 3:self.n_classes + 4] = cx + w / 2

        predictions = predictions[numpy.any(predictions[:, self.n_classes + 1:self.n_classes + 2] - predictions[:, self.n_classes:self.n_classes + 1] > 2, axis=1)]
        predictions = predictions[numpy.any(predictions[:, self.n_classes + 3:self.n_classes + 4] - predictions[:, self.n_classes + 2:self.n_classes + 3] > 2, axis=1)]
        predictions = predictions[numpy.any(predictions[:, :self.n_classes] > self.class_conf_threshold, axis=1)]
        predictions = predictions[numpy.argsort(numpy.amax(predictions[:, :self.n_classes], axis=1), axis=0)][::-1][:100]  # top 100 only
        classes, scores, boxes = [], [], []
        prev_predictions = {}
        for prediction in predictions:
            predicted_class = numpy.argmax(prediction[:self.n_classes])
            any_match = False
            if predicted_class in prev_predictions.keys():
                for prev_prediction in prev_predictions[predicted_class]:
                    if self.intersection_over_union(prev_prediction, prediction[self.n_classes:]) > self.jaccard_similarity_threshold:
                        any_match = True
                        break
            if not any_match:
                classes.append(predicted_class)
                scores.append(prediction[predicted_class])
                boxes.append(prediction[self.n_classes:])
                prev_predictions[predicted_class] = prev_predictions.get(predicted_class, [])
                prev_predictions[predicted_class].append(prediction[self.n_classes:])
        return numpy.array(classes), numpy.array(scores), numpy.array(boxes)

    def encode_input(self, ground_truth_boxes):
        """
        Method to generate anchor boxes based on given ground_truth boxes
        :param ground_truth_boxes: shape=(identified_object_count, 5)
        :return: anchor_boxes in shape=(total_boxes, 1 + n_classes + 4)
        """
        anchor_boxes = numpy.zeros(shape=(self.default_boxes.shape[1], 1 + self.n_classes + 4), dtype='float32')
        anchor_boxes[:, :1 + self.n_classes] = 1 / self.n_classes
        anchor_boxes[:, 0] = 1
        # processing each ground_truth_box
        for ground_truth_box in ground_truth_boxes:
            # process overlapping default_boxes
            g_box = ground_truth_box.copy()
            class_idx = int(g_box[4])
            g_box = numpy.array([g_box[0] - g_box[2] / 2, g_box[0] + g_box[2] / 2, g_box[1] - g_box[3] / 2, g_box[1] + g_box[3] / 2])

            over_lapping_boxes = numpy.argwhere(numpy.any(g_box[0] <= self.default_boxes[0, :, 0:1], axis=-1)
                                                * numpy.any(self.default_boxes[0, :, 0:1] <= g_box[1], axis=-1)
                                                * numpy.any(g_box[2] <= self.default_boxes[0, :, 1:2], axis=-1)
                                                * numpy.any(self.default_boxes[0, :, 1:2] <= g_box[3], axis=-1))
            g_box[:2] *= self.image_shape[:2]
            g_box[2:4] *= self.image_shape[:2]
            over_lapping_boxes = self.find_matching_boxes(g_box, numpy.squeeze(over_lapping_boxes, axis=-1))
            anchor_boxes[over_lapping_boxes, :1 + self.n_classes] = 0
            anchor_boxes[over_lapping_boxes, 1 + class_idx] = 1
            anchor_boxes[over_lapping_boxes, 1 + self.n_classes:] = ground_truth_box[:4]
            anchor_boxes[over_lapping_boxes, 1 + self.n_classes:1 + self.n_classes + 2] -= self.default_boxes[0, over_lapping_boxes, :2]
            anchor_boxes[over_lapping_boxes, 1 + self.n_classes:1 + self.n_classes + 2] /= self.default_boxes[0, over_lapping_boxes, 2:4]
            anchor_boxes[over_lapping_boxes, 1 + self.n_classes + 2:1 + self.n_classes + 4] /= self.default_boxes[0, over_lapping_boxes, 2:4]
            anchor_boxes[over_lapping_boxes, 1 + self.n_classes + 2:1 + self.n_classes + 4] = log(anchor_boxes[over_lapping_boxes, 1 + self.n_classes + 2:1 + self.n_classes + 4])
        return anchor_boxes

    def find_matching_boxes(self, g_box, default_box_idx):
        matches = []
        for d_box_idx in default_box_idx:
            d_cy = self.default_boxes[0, d_box_idx, 0] * self.image_shape[0]
            d_cx = self.default_boxes[0, d_box_idx, 1] * self.image_shape[1]
            d_h = self.default_boxes[0, d_box_idx, 2] * self.image_shape[0]
            d_w = self.default_boxes[0, d_box_idx, 3] * self.image_shape[1]

            # Check whether ground_truth box and g_box match
            if self.intersection_over_union(g_box, (d_cy - 0.5 * d_h, d_cy + 0.5 * d_h, d_cx - 0.5 * d_w, d_cx + 0.5 * d_w)) > self.jaccard_similarity_threshold:
                matches.append(d_box_idx)
        return matches

    @staticmethod
    def intersection_over_union(box_a, box_b):
        """
        Method for Jaccard Similarity calculation
        :param box_a:
        :param box_b:
        :return:
        """
        inter_y_min = max(box_a[0], box_b[0])
        inter_y_max = min(box_a[1], box_b[1])
        inter_x_min = max(box_a[2], box_b[2])
        inter_x_max = min(box_a[3], box_b[3])

        union_y_min = min(box_a[0], box_b[0])
        union_y_max = max(box_a[1], box_b[1])
        union_x_min = min(box_a[2], box_b[2])
        union_x_max = max(box_a[3], box_b[3])

        return (max(inter_y_max - inter_y_min, 0) * max(inter_x_max - inter_x_min, 0)) / (max(union_y_max - union_y_min, 0) * max(union_x_max - union_x_min, 0))

    def plot_boxes(self, img, classes, scores, boxes, file_name, line_width=1, visualize=False):
        """
        Visualize bounding boxes. Largely inspired by SSD-MXNET!
        """
        img_ax = pyplot.imshow(img)
        colors = dict()
        for i in range(classes.shape[0]):
            cls_id = int(classes[i])
            if cls_id >= 0:
                score = scores[i]
                if cls_id not in colors:
                    colors[cls_id] = (random(), random(), random())
                ymin = int(numpy.clip(boxes[i, 0], 0, self.image_shape[0]))
                xmin = int(numpy.clip(boxes[i, 2], 0, self.image_shape[1]))
                ymax = int(numpy.clip(boxes[i, 1], ymin, self.image_shape[0]))
                xmax = int(numpy.clip(boxes[i, 3], xmin, self.image_shape[1]))
                rect = pyplot.Rectangle((xmin, ymin), width=xmax - xmin, height=ymax - ymin, fill=False, edgecolor=colors[cls_id], linewidth=line_width)
                pyplot.gca().add_patch(rect)
                class_name = str(cls_id)
                pyplot.gca().text(xmin + 5, max(ymin - 5, 0), '{:s}|{:.2f}'.format(class_name, score), bbox=dict(facecolor=colors[cls_id], alpha=0.5), fontsize=8, color='white')
        img_ax.figure.savefig(file_name, dpi=500)
        if visualize:
            pyplot.show()
        pyplot.close()


def smooth_l1_loss(y_true, y_pred, delta=1.0):
    """Customized Huber loss method to avoid mean operation

    For each value x in `error = y_true - y_pred`:

    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.

    Returns:
      Tensor with one scalar loss entry per sample.
    """
    y_pred = math_ops.cast(y_pred, dtype=floatx())
    y_true = math_ops.cast(y_true, dtype=floatx())
    error = math_ops.subtract(y_pred, y_true)
    abs_error = math_ops.abs(error)
    quadratic = math_ops.minimum(abs_error, delta)
    linear = math_ops.subtract(abs_error, quadratic)
    return math_ops.add(math_ops.multiply(ops.convert_to_tensor_v2(0.5, dtype=quadratic.dtype), math_ops.multiply(quadratic, quadratic)),
                        math_ops.multiply(delta, linear))
