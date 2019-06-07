'''
Source: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
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
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    return {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints],
        Batch.locref_targets: [batch_size, None, None, num_joints * 2],
        Batch.locref_mask: [batch_size, None, None, num_joints * 2]
    }


class PoseNet:
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
        if int(vers[0])==1 and int(vers[1])<4: #check if lower than version 1.4.
            with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=16)
        else:
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=16,is_training=False, reuse=tf.AUTO_REUSE) #TODO: check is_training and reuse

        return net,end_points

    def get_chAT_mask(self, shape, num_joints=None, locref=False):
        if locref:
            # side = shape[2] // 2 - 1
            # print('shape: ', shape)
            # left_dx = tf.zeros((shape[0], shape[1], side, 0))
            # middle_dx = tf.ones((shape[0], shape[1], 1, 0))
            # right_dx = tf.zeros((shape[0], shape[1], side + 1, 0))
            # mask_dx = tf.concat((left_dx, middle_dx, right_dx), axis=2)
            mask_dx = tf.zeros((shape[0], shape[1], shape[2], 1))
            mask_dy = tf.ones((shape[0], shape[1], shape[2], 1))
            mask = tf.concat((mask_dx, mask_dy), axis=3)
            for i in range(1, num_joints):
                mask_dx = tf.zeros((shape[0], shape[1], shape[2], 1))
                mask_dy = tf.ones((shape[0], shape[1], shape[2], 1))
                mask = tf.concat((mask, mask_dx, mask_dy), axis=3)

            return mask

        side = shape[2] // 2 - 1
        print('shape: ', shape)
        left = tf.zeros((shape[0], shape[1], side, shape[3]))
        middle = tf.ones((shape[0], shape[1], 1, shape[3]))
        right = tf.zeros((shape[0], shape[1], side+1, shape[3]))
        mask = tf.concat((left, middle, right), axis=2)
        return mask

    def prediction_layers(self, features, end_points, reuse=tf.AUTO_REUSE): #TODO: check reuse
        cfg = self.cfg

        num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
        layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

        out = {}
        with tf.variable_scope('pose', reuse=reuse):
            out['part_pred'] = prediction_layer(cfg, features, 'part_pred',
                                                cfg.num_joints)
            # TODO:: only when using chAT mode
            out['part_pred'] = tf.multiply(out['part_pred'], self.get_chAT_mask(tf.shape(out['part_pred'])))

            if cfg.location_refinement:
                out['locref'] = prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_joints * 2)
                out['locref'] = tf.multiply(out['locref'], self.get_chAT_mask(tf.shape(out['locref']), cfg.num_joints, locref=True))

            if cfg.intermediate_supervision:
                interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
                block_interm_out = end_points[interm_name]
                out['part_pred_interm'] = prediction_layer(cfg, block_interm_out,
                                                           'intermediate_supervision',
                                                           cfg.num_joints)

        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)

    def test(self, inputs):
        heads = self.get_net(inputs)
        prob = tf.sigmoid(heads['part_pred'])
        prob = tf.multiply(prob, self.get_chAT_mask(tf.shape(prob)))

        return {'part_prob': prob, 'locref': heads['locref']}

    def train(self, batch):
        cfg = self.cfg

        heads = self.get_net(batch[Batch.inputs])

        try:
            print('chAT mode', cfg['chAT'])
            if cfg['chAT']:
                return self.calc_loss_chAT(cfg, heads, batch)
        except KeyError:
            print('chAT mode off')

        weigh_part_predictions = cfg.weigh_part_predictions
        part_score_weights = batch[Batch.part_score_weights] if weigh_part_predictions else 1.0

        def add_part_loss(pred_layer):
            return tf.losses.sigmoid_cross_entropy(batch[Batch.part_score_targets],
                                                   heads[pred_layer],
                                                   part_score_weights)

        loss = {}
        loss['part_loss'] = add_part_loss('part_pred')
        total_loss = loss['part_loss']
        if cfg.intermediate_supervision:
            loss['part_loss_interm'] = add_part_loss('part_pred_interm')
            total_loss = total_loss + loss['part_loss_interm']

        if cfg.location_refinement:
            locref_pred = heads['locref']
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]

            loss_func = losses.huber_loss if cfg.locref_huber_loss else tf.losses.mean_squared_error
            loss['locref_loss'] = cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
            total_loss = total_loss + loss['locref_loss']

        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        loss['total_loss'] = total_loss
        return loss


    def calc_loss_chAT(self, cfg, heads, batch):
        import numpy as np
        weigh_part_predictions = cfg.weigh_part_predictions
        part_score_weights = batch[Batch.part_score_weights] if weigh_part_predictions else 1.0
        def add_part_loss(pred_layer):
            shape = tf.shape(heads[pred_layer])
            side = shape[2]//2 - 1
            print('shape: ', shape)
            left = tf.zeros((shape[0], shape[1], side, shape[3]))
            middle = tf.ones((shape[0], shape[1], 1, shape[3]))
            right = tf.zeros((shape[0], shape[1], side+1, shape[3]))
            mask = tf.concat((left, middle, right), axis=2)
            #mask[:, :, shape[2]//2] = tf.ones((shape[0], 1, shape[1]))
            return tf.losses.sigmoid_cross_entropy(batch[Batch.part_score_targets],
                                                   heads[pred_layer],
                                                   np.multiply(part_score_weights, mask))

        loss = {}
        loss['part_loss'] = add_part_loss('part_pred')
        total_loss = loss['part_loss']
        if cfg.intermediate_supervision:
            loss['part_loss_interm'] = add_part_loss('part_pred_interm')
            total_loss = total_loss + loss['part_loss_interm']

        if cfg.location_refinement:
            locref_pred = heads['locref']
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]
            # shape = tf.shape(locref_pred)
            # side = shape[2] // 2 - 1
            # print('shape: ', shape)
            # left = tf.zeros((shape[0], shape[1], side, shape[3]))
            # middle = tf.ones((shape[0], shape[1], 1, shape[3]))
            # right = tf.zeros((shape[0], shape[1], side + 1, shape[3]))
            # mask = tf.concat((left, middle, right), axis=2)
            # locref_weights = np.multiply(locref_weights, mask)

            loss_func = losses.huber_loss if cfg.locref_huber_loss else tf.losses.mean_squared_error
            loss['locref_loss'] = cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
            total_loss = total_loss + loss['locref_loss']

        self._inp = batch[Batch.inputs]
        self._outp = heads['part_pred']
        self._outp_locref = heads['locref']
        self._targ = batch[Batch.part_score_targets]
        self._targ_locref = batch[Batch.locref_targets]
        self._mask_locref = batch[Batch.locref_mask]
        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        #total_loss = tf.Print(total_loss, [loss['part_loss']])


        loss['total_loss'] = total_loss
        return loss