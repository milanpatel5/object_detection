import numpy
from tensorflow.python import Constant, Min, Max, where
from tensorflow.python.framework import ops
from tensorflow.python.keras.backend import sum, mean, log, zeros, floatx
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Add, Reshape, Softmax, Concatenate
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
            x = BottleneckBlock(self.t, self.c, 1, True)(x)
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

        # Fixed set of default boxes with different scales and aspect ratios (Other random generation method can be used)
        self.default_boxes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2), (1, 2), (2, 1), (2, 2)][:n_boxes]

    def __call__(self, *features):
        """
        Used inbuilt call method to give functional API appearance to this Module
        :param features: all the feature maps collected from MultiScaleFeatureMaps generation module and also from base module
        :return: prediction tensor with size: (batches, total_boxes, n_classes+8)
        """
        predictions = []
        for feature in features:
            # Classification confidence with softmax operation
            class_conf = Conv2D(filters=self.n_boxes * self.n_classes, kernel_size=(3, 3), padding='same')(feature)
            class_conf = Reshape(target_shape=(-1, self.n_classes))(class_conf)
            class_conf = Softmax()(class_conf)

            # Localization layers
            loc_var = Conv2D(filters=self.n_boxes * 4, kernel_size=(3, 3), padding='same')(feature)
            loc_var = Reshape(target_shape=(-1, 4))(loc_var)

            # Default boxes for feature-map reference
            loc_ref = self.make_boxes(feature.shape[0], feature.shape[1], feature.shape[2])

            # concatenation of all the prediction features along last axis
            feature_predictions = Concatenate(axis=-1)([class_conf, loc_var, loc_ref])
            predictions.append(feature_predictions)

        # concatenation of all boxes
        predictions = Concatenate(axis=1)(predictions)
        return predictions

    def make_boxes(self, batch_size, feature_height, feature_width):
        """
        Create AnchorBox for feature-map reference
        :param batch_size:
        :param feature_height:
        :param feature_width:
        :return: Anchor boxes in shape (batch_size, feature_height * feature_width * self.n_boxes, 4)
        """
        boxes = []
        for y in range(feature_height):
            for x in range(feature_width):
                for default_box in self.default_boxes:
                    cy = y / feature_height
                    cx = x / feature_width
                    h = (1 + default_box[0]) / feature_height
                    w = (1 + default_box[1]) / feature_width
                    boxes.append([cy, cx, h, w])
        boxes = numpy.array(boxes, copy=False, dtype=float)
        # broadcast them to all batches
        boxes = numpy.broadcast_to(boxes, shape=(batch_size if batch_size else 0, feature_height * feature_width * self.n_boxes, boxes.shape[1]))
        return boxes


class SingleShotDetector:
    def __init__(self, image_shape, class_conf_threshold=0.1, jaccard_similarity_threshold=0.5, mode='train', n_boxes=4, n_classes=100, loc_loss_weight=1):
        """
        SingleShotDetector model class
        :param image_shape: a tuple of image shape
        :param class_conf_threshold: Threshold value for classification confidence
        :param jaccard_similarity_threshold: Threshold value for jaccard similarity(Intersection Over Union)
        :param mode: Mention whether it is 'train' or 'valid' mode
        :param n_boxes: Number of default boxes per position of feature map
        :param n_classes: Number of classes
        :param loc_loss_weight: Weight(alpha) of localization loss
        """
        self.class_conf_threshold = Constant(class_conf_threshold)
        self.jaccard_similarity_threshold = Constant(jaccard_similarity_threshold)
        self.mode = mode
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        self.loc_loss_weight = loc_loss_weight
        self.image_shape = image_shape

    def __call__(self, base_1, base_2):
        # Extract features of varying scales from input of base model
        conv_layers = MultiScaleFeatureMaps()(base_2)

        # Prepare predictions from extracted features
        predictions = Predictors(n_boxes=self.n_boxes, n_classes=self.n_classes)(base_1, base_2, *conv_layers)

        if self.mode == 'train':
            # if it's training mode then return predictions as they are
            return predictions
        elif self.mode == 'valid':
            # In case of validation decode the predictions and apply Non-Maximum Suppression
            pass

    def loss_fn(self, y_true, y_pred):
        """
        Method to calculate loss during training
        :param y_true: tensor containing the Anchor boxes of corresponding ground truth boxes in shape(batch, total_boxes, n_classes + 5)
        :param y_pred: tensor containing the predicted boxes and default boxes in shape(batch, total_boxes, n_classes + 8)
        :return: mean loss value
        """
        # generate Anchor boxes from given ground truth boxes
        y_true = self.encode_input(y_true, y_pred[:, :, self.n_classes + 4:])

        # identifying matched anchor boxes
        is_match = sum(1 - y_true[:, :, :1], axis=2)
        n_matched_boxes = sum(is_match, axis=1)

        # calculating losses
        classification_loss = categorical_crossentropy(y_true=y_true[:, :, 1:self.n_classes + 1], y_pred=y_pred[:, :, :self.n_classes])
        localization_loss = Add()([huber_loss(y_true=y_true[:, :, self.n_classes + 1 + i], y_pred=y_pred[:, :, self.n_classes + i]) for i in range(4)])

        # weighted total loss
        loss = classification_loss + self.loc_loss_weight * localization_loss

        # we only want to calculate losses for matched anchor boxes
        loss = where(is_match == 0, zeros(shape=is_match.shape), loss)
        # taking mean over matched anchor boxes
        loss = sum(loss, axis=1)
        loss = where(n_matched_boxes > 0, loss / n_matched_boxes, zeros(shape=n_matched_boxes.shape))
        # taking mean over batch
        loss = mean(loss, axis=0)

        return loss

    def encode_input(self, ground_truth_boxes, default_boxes):
        """
        Method to generate anchor boxes based on given ground_truth boxes
        :param ground_truth_boxes: shape=(batch_size, identified_object_count, 5)
        :param default_boxes: shape=(batch_size, total_boxes, 4)
        :return: anchor_boxes in shape=(batch_size, total_boxes, 1 + n_classes + 4)
        """
        batch_size = default_boxes.shape[0]
        total_boxes = default_boxes.shape[1]

        anchor_boxes = zeros(shape=(batch_size, total_boxes, 1 + self.n_classes + 4), dtype=float)
        for image_idx in range(batch_size):
            # processing each default_box
            for default_box_idx, default_box in enumerate(default_boxes[image_idx]):
                # find the overlapping ground_truth box
                matched_box_idx = self.find_matching_box(default_box, ground_truth_boxes[image_idx])
                if 0 <= matched_box_idx < ground_truth_boxes[image_idx].shape[0]:
                    matched_box = ground_truth_boxes[image_idx][matched_box_idx]
                    # assign target class
                    anchor_boxes[image_idx, default_box_idx, matched_box[4] + 1] = 1
                    # calculate g^ vector
                    anchor_boxes[image_idx, default_box_idx, 1 + self.n_classes:] += matched_box[:4]
                    anchor_boxes[image_idx, default_box_idx, 1 + self.n_classes:1 + self.n_classes + 2] -= matched_box[:2]
                    anchor_boxes[image_idx, default_box_idx, 1 + self.n_classes:1 + self.n_classes + 2] /= matched_box[2:]
                    anchor_boxes[image_idx, default_box_idx, 1 + self.n_classes + 2:] /= matched_box[2:]
                    anchor_boxes[image_idx, default_box_idx, 1 + self.n_classes + 2:] = log(anchor_boxes[image_idx, default_box_idx, 1 + self.n_classes + 2:])
                else:
                    anchor_boxes[image_idx, default_box_idx, 0] = 1

        return anchor_boxes

    def find_matching_box(self, default_box, ground_truth_boxes):
        # Scaled (cy, cx, h, w) parameters of default box based on input image dimensions
        d_cy = default_box[0] * self.image_shape[0]
        d_cx = default_box[1] * self.image_shape[1]
        d_h = default_box[2] * self.image_shape[0]
        d_w = default_box[3] * self.image_shape[1]

        for idx, g_box in enumerate(ground_truth_boxes):
            # Scaled (cy, cx, h, w) parameters of ground_truth box based on input image dimensions
            g_cy = g_box[0] * self.image_shape[0]
            g_cx = g_box[1] * self.image_shape[1]
            g_h = g_box[2] * self.image_shape[0]
            g_w = g_box[3] * self.image_shape[1]

            # Check whether ground_truth box and default_box match
            if self.intersection_over_union((d_cy - 0.5 * d_h, d_cy + 0.5 * d_h, d_cx - 0.5 * d_w, d_cx + 0.5 * d_w),
                                            (g_cy - 0.5 * g_h, g_cy + 0.5 * g_h, g_cx - 0.5 * g_w, g_cx + 0.5 * g_w)) > self.jaccard_similarity_threshold:
                return idx
        return -1

    @staticmethod
    def intersection_over_union(default_box, ground_truth_box):
        """
        Method for Jaccard Similarity calculation
        :param default_box:
        :param ground_truth_box:
        :return:
        """
        inter_y_min = Max([default_box[0], ground_truth_box[0]])
        inter_y_max = Min([default_box[1], ground_truth_box[1]])
        inter_x_min = Max([default_box[2], ground_truth_box[2]])
        inter_x_max = Min([default_box[3], ground_truth_box[3]])

        union_y_min = Min([default_box[0], ground_truth_box[0]])
        union_y_max = Max([default_box[1], ground_truth_box[1]])
        union_x_min = Min([default_box[2], ground_truth_box[2]])
        union_x_max = Max([default_box[3], ground_truth_box[3]])

        return ((inter_y_max - inter_y_min) * (inter_x_max - inter_x_min)) / ((union_y_max - union_y_min) * (union_x_max - union_x_min))


def huber_loss(y_true, y_pred, delta=1.0):
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
