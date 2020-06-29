# !/usr/bin/python3.6
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np


class ModelGraph(object):
    def __init__(self, mode='dev', is_training=True, options=None):
        self.options = options
        self.is_training = is_training
        self.vector_dim = int(options.column_count)
        self.mode = mode
        self.float_fmt = getattr(tf, self.options.float_format)
        self.int_fmt = getattr(tf, self.options.int_format)

        self.train_op = []
        self._extra_train_ops = []
        self.learning_rate = None

        self.truth_0 = None
        self.truth_1 = None
        self.loss_1 = 0
        self.loss_0 = 0
        self.vector = None
        self.global_step = None
        self.feature_dim = None

        self.input_vector = None
        self.input_vector_normalized = None
        self.lstm_output = None
        self.lstm_output_normalized = None
        self.share_layer1_output = None
        self.predict_layer1_output_normalized = None

        self.summaries = None
        self.create_placeholders()
        self.create_model_graph()

    def create_placeholders(self):
        if self.options.is_fit:
            self.truth_1 = tf.placeholder(self.float_fmt, [None, self.options.num_fit_target])  # [batch_size]
        if self.options.is_classifier:
            self.truth_0 = tf.placeholder(self.int_fmt, [None])  # [batch_size]
        if not self.options.is_fit and not self.options.is_classifier:
            raise ValueError('Model type should be set, options.is_fit and options.is_classifier cannot be false both!')
        self.vector = tf.placeholder(self.float_fmt,
                                     [None, self.vector_dim])  # [batch_size, time_latent_len * feature_dim]

    def create_feed_dict(self, cur_batch):
        if self.options.is_fit and not self.options.is_classifier:
            feed_dict = {
                self.vector: cur_batch.vector,
                self.truth_1: cur_batch.label_truth_1,
            }
        elif not self.options.is_fit and self.options.is_classifier:
            feed_dict = {
                self.vector: cur_batch.vector,
                self.truth_0: cur_batch.label_truth_0,
            }
        elif self.options.is_fit and self.options.is_classifier:
            feed_dict = {
                self.vector: cur_batch.vector,
                self.truth_0: cur_batch.label_truth_0,
                self.truth_1: cur_batch.label_truth_1,
            }
        else:
            raise ValueError('Model type should be set, options.is_fit and options.is_classifier cannot be false both!')
        return feed_dict

    def create_model_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        if self.options.model == "biLSTM":
            self._build_model_LSTM(self.vector_dim, bi_direction=True)
        elif self.options.model == "CNN":
            self._build_model_CNN(self.vector_dim)
        else:
            self._build_model_LSTM(self.vector_dim, bi_direction=False)

        if self.is_training: self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _build_model_LSTM(self, vector_dim, bi_direction=False):
        num_classes = self.options.num_classes
        num_target = self.options.num_fit_target
        options = self.options
        time_latent_len = options.time_latent
        self.feature_dim = int(vector_dim / time_latent_len)

        if self.is_training or self.mode == 'eval':
            reuse_flag = None
        else:
            reuse_flag = True

        with tf.variable_scope("Model", reuse=reuse_flag):
            # ========Initial input Layer=========
            with tf.variable_scope("Initial_input"):
                self.input_vector_normalized = tf.reshape(self.vector, [-1, time_latent_len, self.feature_dim])
                # self.input_vector_normalized = self.BatchNorm('init_bn', self.input_vector)

            # ========LSTM Layer=========
            with tf.variable_scope("LSTM_layer"):
                if bi_direction:
                    lstm_output, match_dim = self._bidirectional_lstm_nn(self.input_vector_normalized)
                else:
                    lstm_output, match_dim = self._lstm_nn(self.input_vector_normalized)

                if self.options.attention:
                    lstm_output, _ = ModelGraph.attention(lstm_output, self.options.time_latent,
                                                          data_type=self.float_fmt)
                else:
                    lstm_output = lstm_output[:, -1, :]
                lstm_output = tf.reshape(lstm_output, [-1, match_dim])
                self._variable_summaries(lstm_output, name='lstm_output')
                self.lstm_output = lstm_output

            # ========Prediction Layer=========
            with tf.variable_scope("Share_layer"):
                w_0 = tf.get_variable("w_0", [match_dim, match_dim // 2], dtype=self.float_fmt,
                                      initializer=tf.uniform_unit_scaling_initializer(factor=1.0))  # tf.dtypes.float32
                b_0 = tf.get_variable("b_0", [match_dim // 2], dtype=self.float_fmt,
                                      initializer=tf.constant_initializer())
                tf.summary.histogram('share_layer_weight', w_0)
                tf.summary.histogram('share_layer_bias', b_0)
                self.share_layer1_output = tf.matmul(self.lstm_output, w_0) + b_0
                # predict_layer1_output_normalized = self.BatchNorm('final_1_bn', self.share_layer1_output)
                self.share_layer1_output = ModelGraph.swish(self.share_layer1_output)
                if self.is_training:
                    self.share_layer1_output = tf.nn.dropout(self.share_layer1_output, (1 - options.dropout_rate))
                self._variable_summaries(self.share_layer1_output, name='share_layer_output')

            if self.options.is_classifier:
                with tf.variable_scope("classifier_layer_1"):
                    w_1 = tf.get_variable("w_1", [match_dim // 2, num_classes], dtype=self.float_fmt,
                                          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                    b_1 = tf.get_variable("b_1", [num_classes], dtype=self.float_fmt,
                                          initializer=tf.constant_initializer())
                    tf.summary.histogram('classifier_layer_weight', w_1)
                    tf.summary.histogram('classifier_layer_bias', b_1)
                    self.classifier_layer1_output = tf.matmul(self.share_layer1_output, w_1) + b_1
                    self.classifier_layer1_output_normalized = self.BatchNorm('final_1_bn',
                                                                              self.classifier_layer1_output)
                    self._variable_summaries(self.classifier_layer1_output_normalized,
                                             name='classifier_layer_output_normalized')

                with tf.variable_scope("classifier_result"):
                    self.classifier_output_before_softmax = ModelGraph.penalized_tanh(
                        self.classifier_layer1_output_normalized)
                    tf.summary.histogram('classifier_output_before_softmax', self.classifier_output_before_softmax)
                    self.final_output_0 = tf.nn.softmax(self.classifier_output_before_softmax)
                    self._variable_summaries(self.final_output_0, name='classifier_output')
                    self.target_matrix_0 = tf.one_hot(self.truth_0, num_classes, dtype=self.int_fmt)
                    weight_list = [2, 1, 2]
                    target_temp = self.target_matrix_0 * weight_list
                    self.loss_0 = -tf.reduce_mean(tf.cast(target_temp, dtype=self.float_fmt) *
                                                  tf.log(self.final_output_0))
                    tf.summary.scalar('classifier_loss', self.loss_0)
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                    # labels=gold_matrix))

                    correct = tf.nn.in_top_k(tf.cast(self.final_output_0, dtype=tf.float32), self.truth_0, 1)
                    self.correct = correct
                    self.eval_correct = tf.reduce_sum(tf.cast(correct, self.int_fmt))
                    self.predictions = tf.argmax(self.final_output_0, 1)

            if self.options.is_fit:
                with tf.variable_scope("fit_layer_1"):
                    w_1 = tf.get_variable("w_1", [match_dim // 2, match_dim], dtype=self.float_fmt,
                                          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                    b_1 = tf.get_variable("b_1", [match_dim], dtype=self.float_fmt,
                                          initializer=tf.constant_initializer())
                    tf.summary.histogram('fit_layer1_weight', w_1)
                    tf.summary.histogram('fit_layer1_bias', b_1)
                    self.fit_layer1_output = tf.matmul(self.share_layer1_output, w_1) + b_1
                    self.fit_layer1_output = ModelGraph.relu(self.fit_layer1_output)
                    self._variable_summaries(self.fit_layer1_output, name='fit_layer1_output')

                with tf.variable_scope("fit_layer_2"):
                    w_2 = tf.get_variable("w_2", [match_dim, num_target], dtype=self.float_fmt,
                                          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                    b_2 = tf.get_variable("b_2", [num_target], dtype=self.float_fmt,
                                          initializer=tf.constant_initializer())
                    tf.summary.histogram('fit_layer2_weight', w_2)
                    tf.summary.histogram('fit_layer2_bias', b_2)
                    self.final_output_2 = tf.matmul(self.fit_layer1_output, w_2) + b_2
                    self.final_output_2 = tf.tanh(self.BatchNorm('bn', self.final_output_2))

                with tf.variable_scope("fit_result"):
                    self.target_matrix_1 = self.truth_1
                    self._variable_summaries(self.final_output_2, name='fit_layer_output_final')
                    # weight_list = tf.map_fn(
                    #     lambda x:
                    #     1 / tf.distributions.Normal(loc=0., scale=1.).prob(tf.minimum(tf.abs(tf.math.atanh(x)), 2.5)),
                    #     self.target_matrix_1)
                    weight_list = tf.abs(self.target_matrix_1)
                    weight_list = weight_list / tf.reduce_sum(weight_list)
                    self.loss_1 = tf.reduce_sum(weight_list * tf.square(self.final_output_2 - self.target_matrix_1))
                    # self.loss_1 = tf.reduce_mean(tf.square(self.final_output_2 - self.target_matrix_1))
                    tf.summary.scalar('fit_loss', self.loss_1)

            tvars = tf.trainable_variables()
            if self.options.lambda_l2 > 0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1
                                    and not v.op.name.find(r'b_') > 0 and not v.op.name.find(r'beta') > 0
                                    and not v.op.name.find(r'gamma') > 0])
            else:
                l2_loss = 0

            with tf.variable_scope("overall_result"):
                self.loss = self.loss_0 * float(self.options.task_weight) + self.loss_1 * (
                        1 - float(self.options.task_weight)) \
                            + self.options.lambda_l2 * l2_loss

    def _build_model_CNN(self, vector_dim):

        """Build the core model within the graph."""
        num_classes = self.options.num_classes
        num_target = self.options.num_fit_target
        self.feature_dim = int(np.sqrt(vector_dim))
        self.input_vector = tf.reshape(self.vector, [-1, self.feature_dim, self.feature_dim, 1])

        if self.is_training or self.mode == 'eval':
            reuse_flag = None
        else:
            reuse_flag = True

        with tf.variable_scope("Model", reuse=reuse_flag):
            with tf.variable_scope('init'):
                x = self.input_vector
                x = self._conv('init_conv', x, 3, 1, 16, self._stride_arr(1))

            strides = [1, 2, 2]
            activate_before_residual = [True, False, False]
            if self.options.use_bottleneck:
                res_func = self._bottleneck_residual
                filters = [16, 64, 128, 256]
            else:
                res_func = self._residual
                filters = [16, 16, 32, 64]
                # Uncomment the following codes to use w28-10 wide residual network.
                # It is more memory efficient than very deep residual network and has
                # comparably good performance.
                # https://arxiv.org/pdf/1605.07146v1.pdf
                # filters = [16, 160, 320, 640]
                # Update hps.num_residual_units to 4

            with tf.variable_scope('unit_1_0'):
                x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                             activate_before_residual[0])
            for i in range(1, self.options.num_residual_units):
                with tf.variable_scope('unit_1_%d' % i):
                    x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

            with tf.variable_scope('unit_2_0'):
                x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                             activate_before_residual[1])
            for i in range(1, self.options.num_residual_units):
                with tf.variable_scope('unit_2_%d' % i):
                    x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

            with tf.variable_scope('unit_3_0'):
                x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                             activate_before_residual[2])
            for i in range(1, self.options.num_residual_units):
                with tf.variable_scope('unit_3_%d' % i):
                    x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

            with tf.variable_scope('unit_last'):
                x = self.BatchNorm('final_bn', x)
                x = self.swish(x)
                x = self._global_avg_pool(x)

            if self.options.is_classifier:
                with tf.variable_scope("classifier_layer_1"):
                    w_1 = tf.get_variable("w_1", [filters[3], num_classes], dtype=self.float_fmt,
                                          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                    b_1 = tf.get_variable("b_1", [num_classes], dtype=self.float_fmt,
                                          initializer=tf.constant_initializer())
                    tf.summary.histogram('classifier_layer_weight', w_1)
                    tf.summary.histogram('classifier_layer_bias', b_1)
                    self.classifier_layer1_output = tf.matmul(x, w_1) + b_1
                    self.classifier_layer1_output_normalized = self.BatchNorm('final_1_bn',
                                                                              self.classifier_layer1_output)
                    self._variable_summaries(self.classifier_layer1_output_normalized,
                                             name='classifier_layer_output_normalized')

                with tf.variable_scope("classifier_result"):
                    self.classifier_output_before_softmax = ModelGraph.penalized_tanh(
                        self.classifier_layer1_output_normalized)
                    tf.summary.histogram('classifier_output_before_softmax', self.classifier_output_before_softmax)
                    self.final_output_0 = tf.nn.softmax(self.classifier_output_before_softmax)
                    self._variable_summaries(self.final_output_0, name='classifier_output')
                    self.target_matrix_0 = tf.one_hot(self.truth_0, num_classes, dtype=self.float_fmt)
                    weight_list = [2, 1, 2]
                    target_temp = self.target_matrix_0 * weight_list
                    self.loss_0 = -tf.reduce_mean(target_temp * tf.log(self.final_output_0))
                    tf.summary.scalar('classifier_loss', self.loss_0)
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))

                    correct = tf.nn.in_top_k(self.final_output_0, self.truth_0, 1)
                    self.correct = correct
                    self.eval_correct = tf.reduce_sum(tf.cast(correct, self.int_fmt))
                    self.predictions = tf.argmax(self.final_output_0, 1)

            if self.options.is_fit:
                with tf.variable_scope("fit_layer_1"):
                    w_1 = tf.get_variable("w_1", [filters[3], 50], dtype=self.float_fmt,
                                          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                    b_1 = tf.get_variable("b_1", [50], dtype=self.float_fmt,
                                          initializer=tf.constant_initializer())
                    tf.summary.histogram('fit_layer1_weight', w_1)
                    tf.summary.histogram('fit_layer1_bias', b_1)
                    self.fit_layer1_output = tf.matmul(x, w_1) + b_1
                    self.fit_layer1_output = ModelGraph.swish(self.fit_layer1_output, beta=1)
                    self._variable_summaries(self.fit_layer1_output, name='fit_layer1_output')

                with tf.variable_scope("fit_layer_2"):
                    w_2 = tf.get_variable("w_2", [50, num_target], dtype=self.float_fmt,
                                          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                    b_2 = tf.get_variable("b_2", [num_target], dtype=self.float_fmt,
                                          initializer=tf.constant_initializer())
                    tf.summary.histogram('fit_layer2_weight', w_2)
                    tf.summary.histogram('fit_layer2_bias', b_2)
                    self.final_output_1 = tf.matmul(self.fit_layer1_output, w_2) + b_2

                with tf.variable_scope("fit_result"):
                    self.target_matrix_1 = self.truth_1
                    self._variable_summaries(self.final_output_1, name='fit_layer_output_final')
                    # corr = ModelGraph.correlation_coefficient_loss(self.final_output_1, self.target_matrix_1)
                    # self.loss_1 = corr
                    self.loss_1 = tf.reduce_mean(tf.square(self.final_output_1 - self.target_matrix_1))
                    tf.summary.scalar('fit_loss', self.loss_1)

            tvars = tf.trainable_variables()
            if self.options.lambda_l2 > 0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1
                                    and not v.op.name.find(r'b_') > 0 and not v.op.name.find(r'beta') > 0
                                    and not v.op.name.find(r'gamma') > 0])
            else:
                l2_loss = 0

            with tf.variable_scope("overall_result"):
                self.loss = self.loss_0 * float(self.options.task_weight) + self.loss_1 * (
                        1 - float(self.options.task_weight)) \
                            + self.options.lambda_l2 * l2_loss

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.learning_rate = tf.constant(self.options.learning_rate, self.float_fmt)
        tf.summary.scalar('learning_rate', self.learning_rate)

        if self.options.optimizer_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate)
        elif self.options.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)
        elif self.options.optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.options.learning_rate)
        elif self.options.optimizer_type == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.options.learning_rate, 0.9)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.options.learning_rate)

        tvars = tf.trainable_variables()
        grads = ModelGraph.compute_gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.options.grad_clipper)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

        if self.options.with_moving_average:
            # Track the moving averages of all trainable variables.
            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_ops = [self.train_op, variables_averages_op]
            self.train_op = tf.group(*train_ops)

        train_ops = [self.train_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    def _lstm_nn(self, input_vector):
        lstm_dim = self.options.lstm_dim
        layer_num = self.options.layer_num
        with tf.variable_scope('unidirectional_lstm_nn'):
            stacked_rnn = []
            for iiLyr in range(layer_num):
                with tf.variable_scope('unit_%d' % iiLyr):
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, state_is_tuple=True,
                                                        initializer=tf.contrib.layers.xavier_initializer(
                                                            uniform=False, dtype=self.float_fmt))
                    if self.is_training:
                        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                                  output_keep_prob=(1 - self.options.dropout_rate))
                    stacked_rnn.append(lstm_cell)
            MultiLyr_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
            output_vector, _ = tf.nn.dynamic_rnn(
                MultiLyr_cell, input_vector, dtype=self.float_fmt)  # [batch_size, question_len, context_lstm_dim]
        return output_vector, lstm_dim

    def _bidirectional_lstm_nn(self, input_vector):
        lstm_dim = self.options.lstm_dim
        layer_num = self.options.layer_num
        with tf.variable_scope('bidirectional_lstm_nn'):
            stacked_rnn_fw = []
            stacked_rnn_bw = []
            for iiLyr in range(layer_num):
                with tf.variable_scope('unit_fw_%d' % iiLyr):
                    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, state_is_tuple=True,
                                                           initializer=tf.contrib.layers.xavier_initializer(
                                                               uniform=False, dtype=self.float_fmt))
                    if self.is_training:
                        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw,
                                                                     output_keep_prob=(1 - self.options.dropout_rate))
                    stacked_rnn_fw.append(lstm_cell_fw)
                with tf.variable_scope('unit_bw_%d' % iiLyr):
                    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, state_is_tuple=True,
                                                           initializer=tf.contrib.layers.xavier_initializer(
                                                               uniform=False, dtype=self.float_fmt))
                    if self.is_training:
                        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw,
                                                                     output_keep_prob=(1 - self.options.dropout_rate))
                    stacked_rnn_bw.append(lstm_cell_bw)
            MultiLyr_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
            MultiLyr_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                MultiLyr_lstm_cell_fw, MultiLyr_lstm_cell_bw, input_vector,
                dtype=self.float_fmt)  # [batch_size, question_len, context_lstm_dim]
            output_vector = tf.concat(axis=2, values=[f_rep, b_rep])

        output_vector_dim = lstm_dim * 2
        return output_vector, output_vector_dim

    @staticmethod
    def attention(inputs, attention_size, data_type, name='attention'):
        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
        # Trainable parameters
        with tf.variable_scope(name):
            w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name='w_omega',
                                  dtype=data_type)
            b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.001), name='b_omega', dtype=data_type)
            u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.001), name='u_omega', dtype=data_type)

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphaList')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        return output, alphas

    @staticmethod
    def _variable_summaries(var, name='summaries'):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.math.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    @staticmethod
    def swish(x, beta=1.0):
        return x * tf.math.sigmoid(x * beta)

    @staticmethod
    def penalized_tanh(x, alpha=0.25):
        return tf.where(tf.math.less(x, 0.0), alpha * tf.math.tanh(x), tf.math.tanh(x))

    @staticmethod
    def compute_gradients(tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

    def BatchNorm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            axis = list(range(len(x.shape) - 1))

            beta = tf.get_variable(
                'beta', params_shape, self.float_fmt,
                initializer=tf.constant_initializer(0.0, self.float_fmt))
            gamma = tf.get_variable(
                'gamma', params_shape, self.float_fmt,
                initializer=tf.constant_initializer(1.0, self.float_fmt))

            if self.is_training:
                mean, variance = tf.nn.moments(x, axis, name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, self.float_fmt,
                    initializer=tf.constant_initializer(0.0, self.float_fmt),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, self.float_fmt,
                    initializer=tf.constant_initializer(1.0, self.float_fmt),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, self.float_fmt,
                    initializer=tf.constant_initializer(0.0, self.float_fmt),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, self.float_fmt,
                    initializer=tf.constant_initializer(1.0, self.float_fmt),
                    trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        tf.summary.scalar('fit_loss', self.loss_1)
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self.BatchNorm('init_bn', x)
                x = self.swish(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self.BatchNorm('init_bn', x)
                x = self.swish(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self.BatchNorm('bn2', x)
            x = self.swish(x)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        # tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn'):
                x = self.BatchNorm('init_bn', x)
                x = self.swish(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn'):
                orig_x = x
                x = self.BatchNorm('init_bn', x)
                x = self.swish(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self.BatchNorm('bn2', x)
            x = self.swish(x)
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self.BatchNorm('bn3', x)
            x = self.swish(x)
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        # tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    @staticmethod
    def _fully_connected(x, out_dim):
        """FullyConnected layer for final output."""
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    @staticmethod
    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
