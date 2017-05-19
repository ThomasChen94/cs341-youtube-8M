#---------------------------------------------------------------------------------------------------------
#                   This file implements the shuffle and learn layer
#                   input: video with frame representations
#                   output: the mapped representation
#---------------------------------------------------------------------------------------------------------


import tensorflow as tf
import numpy as np
from utils import random_pick_3

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    hidden_size1 = 864
    hidden_size2 = 720
    batch_size = 16
    max_grad_norm = 10.0 # max gradients norm for clipping
    lr = 0.001 # learning rate

    # parameters for the first layer
    filter1_size = 5
    conv1_output_channel = 32
    pool1_length = 5

    # parameters for the second layer
    filter1_size = 3
    conv1_output_channel = 8
    pool1_length = 5

    feature_size = 1024
    input_length = 1024

    input_num_once = 3

    num_shuffle_sample = 10

    def __init__(self, args):
        self.batch_size = 16

class shuffleLearnModel():
    def add_placeholders(self):
        self.input_placeholder  = tf.placeholder(tf.int32, [None, Config.input_length, Config.feature_size])
        self.dropout_placeholder = tf.placeholder(tf.float32)
        #self.num_frames = tf.placeholder(tf.int32, [None])

    def create_feed_dict(self, inputs_batch):
        feed_dict = {}
        feed_dict[self.input_placeholder] = inputs_batch
        #feed_dict[self.num_frames] = num_frames
        return feed_dict

    def add_extract_op(self):
        with tf.variable_scope("conv1"):
            # the first convolutional layer
            filter1 = tf.get_variable("f1", [None, Config.filter1_size, Config.conv1_output_channel])
            conv1 = tf.nn.conv1d(self.input_placeholder1, filter1, stride = 2, padding= "SAME")

            # pooling
            pool1 = tf.nn.pool(conv1, [Config.pool1_length], "MAX", "SAME")

            # activate
            activ1 = tf.nn.relu(pool1)

        with tf.variable_scope("conv2"):
            # the first convolutional layer
            filter2 = tf.get_variable("f2", [None, Config.filter2_size, Config.conv2_output_channel])
            conv2 = tf.nn.conv1d(activ1, filter2, stride = 2, padding= "SAME")

            # pooling
            pool2 = tf.nn.pool(conv2, [Config.pool2_length], "MAX", "SAME")

            # activate
            activ2 = tf.nn.relu(pool2)

        with tf.variable_scope("fc1"):
            fc1_input1 = tf.reshape(activ2, [activ2.shape[0], -1])
            self.fc1_output1 = tf.layers.dense(inputs=fc1_input1, units=1024, activation=tf.nn.sigmoid)


        # return the output of the fully connected layer
        return self.fc1_output1


    def add_random_combination(self, input_features, num_frames):
        """
        input features: a list (length: input_length) of element with shape [batch_size, feature_size]
        Returns:

        """
        shuffle_list, label_list = random_pick_3(num_frames, Config.num_shuffle_sample) # 3-D np array [batch_size, num_samples, 3]
        input_embedding_list = tf.unstack(input_features, axis = 0)
        sample_list = []
        label_list = tf.convert_to_tensor(label_list)
        for i in range(input_features.shape[0]):
            # loop over the batch_size
            shuffle_index = tf.convert_to_tensor(shuffle_list[i])
            shuffle_value = tf.nn.embedding_lookup(input_embedding_list[i], shuffle_index)
            shuffle_concat_list = []
            for j in range(shuffle_value.shape[0]):
                shuffle_concat_list.append(tf.concat([shuffle_value[j][0], shuffle_value[j][1], shuffle_value[j][2]],
                                                     axis = 1))
            sample_list.append(shuffle_concat_list)
        return sample_list, label_list


    def add_shuffle_loss(self, sample_list, label_list):
        #!!!!!!!!! Need Discussion!!!!!!!!
        '''

        This function computes the shuffle loss
        Now, we use a simple softmax classifier to do this job
        It needs further discussion

        '''
        sample_list_spread = tf.reshape(sample_list, [-1, sample_list.shape[2]])
        label_list_spread = tf.reshape(label_list, [-1, ])
        self.shuffle_loss = tf.nn.softmax_cross_entropy_with_logits(sample_list_spread, label_list_spread)
        return self.shuffle_loss

    def __init__(self, num_frames):
        self.config = Config()
        self.input_placeholder = None
        self.dropout_placeholder = None
        self.add_placeholders()
        output_feat = self.add_extract_op()
        sample_list, label_list = self.add_random_combination(output_feat, num_frames)
        self.loss = self.add_shuffle_loss(sample_list, label_list)










