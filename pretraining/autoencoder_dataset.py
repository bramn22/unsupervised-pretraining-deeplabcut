'''
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''
import os
import logging
import random as rand
from enum import Enum

import numpy as np
from numpy import array as arr
from numpy import concatenate as cat

import scipy.io as sio
from scipy.misc import imread, imresize
import cv2
import random

class Batch(Enum):
    inputs = 0
    targets = 1
    masks = 2
    data_item = 3


def data_to_input(data):
    return np.expand_dims(data, axis=0).astype(float)


class DataItem:
    pass


class UnsupDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = self.load_dataset()
        self.ref_image = self.get_ref_image()
        self.num_images = len(self.data)
        self.curr_img = 0
        self.set_shuffle(cfg.shuffle)

    def get_ref_image(self):
        return imread(os.path.join(self.cfg.project_path, 'pretrain/training-datasets', self.data[0].im_path), mode='RGB')

    def load_dataset(self):
        cfg = self.cfg
        file_name = os.path.join(self.cfg.project_path, cfg.dataset)
        # Load Matlab file dataset annotation
        mlab = sio.loadmat(file_name)
        self.raw_data = mlab
        mlab = mlab['dataset']

        num_images = mlab.shape[1]
        #        print('Dataset has {} images'.format(num_images))
        data = []
        # TODO: check has_gt is True if there is joint information
        has_gt = False

        for i in range(num_images):
            sample = mlab[0, i]

            item = DataItem()
            item.image_id = i
            item.im_path = sample[0][0]
            item.im_size = sample[1][0]

            data.append(item)

        self.has_gt = has_gt
        return data

    def set_test_mode(self, test_mode):
        self.has_gt = not test_mode

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
        if not shuffle:
            assert not self.cfg.mirror
            self.image_indices = np.arange(self.num_images)

    def shuffle_images(self):
        num_images = self.num_images
        if self.cfg.mirror:
            image_indices = np.random.permutation(num_images * 2)
            self.mirrored = image_indices >= num_images
            image_indices[self.mirrored] = image_indices[self.mirrored] - num_images
            self.image_indices = image_indices
        else:
            self.image_indices = np.random.permutation(num_images)

    def num_training_samples(self):
        num = self.num_images
        if self.cfg.mirror:
            num *= 2
        return num

    def next_training_sample(self):
        if self.curr_img == 0 and self.shuffle:
            self.shuffle_images()

        curr_img = self.curr_img
        self.curr_img = (self.curr_img + 1) % self.num_training_samples()

        imidx = self.image_indices[curr_img]
        mirror = self.cfg.mirror and self.mirrored[curr_img]

        return imidx, mirror

    def get_training_sample(self, imidx):
        return self.data[imidx]

    def get_scale(self):
        cfg = self.cfg
        scale = cfg.global_scale
        if hasattr(cfg, 'scale_jitter_lo') and hasattr(cfg, 'scale_jitter_up'):
            scale_jitter = rand.uniform(cfg.scale_jitter_lo, cfg.scale_jitter_up)
            scale *= scale_jitter
        return scale

    def next_batch(self):
        while True:
            imidx, mirror = self.next_training_sample()
            data_item = self.get_training_sample(imidx)
            scale = self.get_scale()

            if not self.is_valid_size(data_item.im_size, scale):
                continue

            return self.make_batch(data_item, scale, mirror)

    def is_valid_size(self, image_size, scale):
        im_width = image_size[2]
        im_height = image_size[1]

        max_input_size = 100
        if im_height < max_input_size or im_width < max_input_size:
            return False

        if hasattr(self.cfg, 'max_input_size'):
            max_input_size = self.cfg.max_input_size
            input_width = im_width * scale
            input_height = im_height * scale
            if input_height * input_width > max_input_size * max_input_size:
                return False

        return True

    def make_batch(self, data_item, scale, mirror):
        im_file = data_item.im_path
        logging.debug('image %s', im_file)
        logging.debug('mirror %r', mirror)

        # print(im_file, os.getcwd())
        # print(self.cfg.project_path)
        image = imread(os.path.join(self.cfg.project_path, 'pretrain/training-datasets', im_file), mode='RGB')
        img = imresize(image, scale) if scale != 1 else image #np.shape -> rows, cols, depth
        scaled_img_size = arr(img.shape[0:2])
        stride = self.cfg.stride
        size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2
        # scmap = np.zeros(cat([size, 3]))
        #ref_image = imread(os.path.join(self.cfg.project_path, 'pretrain/training-datasets', self.ref_image), mode='RGB') # TODO: only read once!
        ref = imresize(self.ref_image, scale) if scale != 1 else self.ref_image #np.shape -> rows, cols, depth

        target = imresize(cv2.subtract(ref, img), size)
        if mirror:
            target = np.fliplr(img)
        # add noise
        noise = np.random.choice([0, 1], size=(img.shape[0], img.shape[1]), p=[1. / 4, 3. / 4])
        stacked_noise = np.stack((noise, noise, noise), axis=2)
        img = np.multiply(img, stacked_noise)
        # rows, cols, depth = np.shape(img)
        # min_width = 100
        # min_height = 100
        # max_width = 400
        # max_height = 400
        # y1 = random.randint(0, rows - min_height - 1) # -1 because end is inclusive
        # x1 = random.randint(0, cols - min_width - 1)
        # y2 = y1 + random.randint(min_height, min(max_height, rows - y1) - 1)
        # x2 = x1 + random.randint(min_width, min(max_width, cols - y1) - 1)
        # mask = np.zeros(img.shape, dtype="uint8")
        # mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (1, 1, 1), cv2.FILLED) #(x, y)
        # mask = imresize(mask, size)
        #
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED) #(x, y)
        #
        # batch = {Batch.inputs: img, Batch.targets: target, Batch.masks: mask}
        batch = {Batch.inputs: img, Batch.targets: target, Batch.masks: ref}
        batch = {key: data_to_input(data) for (key, data) in batch.items()}

        batch[Batch.data_item] = data_item

        return batch