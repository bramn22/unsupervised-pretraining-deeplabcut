
'''
Source: DeeperCut by Eldatf.divide(heads[pred_layer], 255), tf.divide(batch[Batch.targets], 255r Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
#from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
from pretraining.autoencoder_dataset import Batch
from deeplabcut.pose_estimation_tensorflow.nnet import losses


net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
             'resnet_101': resnet_v1.resnet_v1_101}


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                        activation_fn=None, normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(input, num_outputs,
                                         kernel_size=[3, 3], stride=2,
                                         scope='block4')
            return pred


def get_batch_spec(cfg):
    batch_size = cfg.batch_size
    return {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.targets: [batch_size, None, None, 3],
        Batch.masks: [batch_size, None, None, 3]
    }


class AutoEncoderNet:
    def __init__(self, cfg):
        self.cfg = cfg

    def extract_features(self, inputs):
        net_fun = net_funcs[self.cfg.net_type]

        mean = tf.constant(self.cfg.mean_pixel,
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        im_centered = inputs - mean

        # The next part of the code depends upon which tensorflow version you have.
        vers = tf.__version__
        vers = vers.split(".") #Updated based on https://github.com/AlexEMG/DeepLabCut/issues/44
        if int(vers[0])==1 and int(vers[1]  )<4: #check if lower than version 1.4.
            with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=16)
        else:
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=16,is_training=True)

        return net,end_points

    def prediction_layers(self, features, end_points, reuse=None):
        cfg = self.cfg

        num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
        layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

        out = {}
        rgb_output = 3
        with tf.variable_scope('pose', reuse=reuse):
            out['part_pred'] = prediction_layer(cfg, features, 'part_pred', rgb_output)

        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)

    def test(self, inputs):
        heads = self.get_net(inputs)
        prob = tf.sigmoid(heads['part_pred'])
        return {'part_prob': prob, 'locref': heads['locref']}

    def train(self, batch):
        cfg = self.cfg
        heads = self.get_net(batch[Batch.inputs])
        self.input = batch[Batch.inputs]
        self.target = batch[Batch.targets]
        self.output = heads['part_pred']
        self.mask = batch[Batch.masks]

        self.partial_loss = tf.multiply(batch[Batch.masks], tf.subtract(tf.divide(heads['part_pred'], 255), tf.divide(batch[Batch.targets], 255)))


        def mean_squared_loss(pred_layer):
            return tf.reduce_mean(tf.square(tf.subtract(tf.divide(heads[pred_layer], 255), tf.divide(batch[Batch.targets], 255))))


        def l2_loss(pred_layer):
            res = tf.multiply(batch[Batch.masks], tf.subtract(tf.divide(heads[pred_layer], 255), tf.divide(batch[Batch.targets], 255)))
            #res = tf.Print(res, [tf.is_nan(res)], message="my Z-values:")  #
            #res_reshape = tf.reshape(res, shape=[1, -1])
            #return tf.reduce_mean(tf.square(res))
            return tf.norm(tf.multiply(batch[Batch.masks], tf.subtract(tf.divide(heads[pred_layer], 255), tf.divide(batch[Batch.targets], 255))))

        loss = {}
        loss['total_loss'] = mean_squared_loss('part_pred')
        return loss
