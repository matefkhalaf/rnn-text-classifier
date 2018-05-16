import tensorflow as tf


# Creating RNN stacked cell (Layered RNN cells in depth)
def rnn_cell(CONFIGS, dropout):
    # Choosing cell type
    # Default activation is Tanh
    if CONFIGS.rnn_unit == 'rnn':
        rnn_cell_type = tf.nn.rnn_cell.BasicRNNCell
    elif CONFIGS.rnn_unit == 'gru':
        rnn_cell_type = tf.nn.rnn_cell.GRUCell
    elif CONFIGS.rnn_unit == 'lstm':
        rnn_cell_type = tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise Exception("Choose a valid RNN cell type.")

    # Create a single cell
    single_cell = rnn_cell_type(CONFIGS.num_hidden_units)

    # Apply dropoutwrapper to RNN cell (Only output dropout is applied)
    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=1 - dropout)

    # Stack cells on each other (Layers)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell for _ in range(CONFIGS.num_layers)])

    return stacked_cell


# Softmax layer
def rnn_softmax(CONFIGS, outputs):
    # Variable scopes is a way to share variable among different parts of the code
    # helps in initializing variables in one place and reuse them in different parts of code

    with tf.variable_scope('rnn_softmax', reuse=True):
        W_softmax = tf.get_variable("W_softmax", [CONFIGS.num_hidden_units, CONFIGS.num_classes])
        b_softmax = tf.get_variable("b_softmax", [CONFIGS.num_classes])

    logits = tf.matmul(outputs, W_softmax) + b_softmax

    return logits


class model(object):

    def __init__(self, CONFIGS):

        # Placeholders
        self.inputs_X = tf.placeholder(tf.float32, shape=[None, None, CONFIGS.num_hidden_units], name='inputs_X')
        self.targets_y = tf.placeholder(tf.float32, shape=[None, None], name='targets_y')
        self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')
        self.dropout = tf.placeholder(tf.float32)
        with tf.name_scope("rnn"):
            # Create folded RNN network (depth) [RNN cell * num_layers]
            stacked_cell = rnn_cell(CONFIGS, self.dropout)


            # Initial state is zero for each i/p batch as each input example is independent on the other
            initial_state = stacked_cell.zero_state(CONFIGS.batch_size, tf.float32)

        # Unfold RNN cells in time axis

        # sequence_length ->  An int32/int64 vector sized [batch_size]
        # is used to copy-through state and zero-out outputs
        # when past a batch element's sequence length.

        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        # 'state' is a tuple of shape [num_layers, batch_size, cell_state_size]
        #  state[0] is the state from first RNN layer, state[-1] is the state from last RNN layer

            all_outputs, state = tf.nn.dynamic_rnn(cell=stacked_cell, inputs=self.inputs_X, initial_state=initial_state,
                                                sequence_length=self.seq_lens, dtype=tf.float32)

        # Since we are using variable length inputs padded to maximum input length and we are feeding
        # sequence_length to tf.nn.dynamic_rnn, Outputs after input seq. length will be 0, and last state will
        # be propagated, So we can't use output[:,-1,:], instead we will use state[-1]
            if CONFIGS.rnn_unit == 'lstm':
                outputs = state[-1][1]
            else:
                outputs = state[-1]
        # Process RNN outputs
        with tf.variable_scope('rnn_softmax'):
            W_softmax = tf.get_variable("W_softmax", [CONFIGS.num_hidden_units, CONFIGS.num_classes])
            b_softmax = tf.get_variable("b_softmax", [CONFIGS.num_classes])

        # Softmax layer
        # logits [batch_size, num_classes]
        with tf.name_scope("logits"):
            logits = rnn_softmax(CONFIGS, outputs)
            tf.summary.histogram("logits", logits)
            # Convert logits into probabilities
            self.probabilities = tf.nn.softmax(logits)
            tf.summary.histogram("probabilities", self.probabilities)
        with tf.name_scope("accuracy"):
            # Array of boolean
            correct_prediction = tf.equal(tf.argmax(self.targets_y, 1), tf.argmax(self.probabilities, 1))
            # Number of correct examples / batch size
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

        # Loss
        with tf.name_scope("loss"):
            # Multi-class - One label - Mutually exclusive classification, so we use
            # softmax cross entropy cost function
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.targets_y))
            tf.summary.scalar("loss", self.loss)
        ####################################################################


        # Optimization
        with tf.name_scope("optimizer"):
            # Define learning rate (Updated each epoch)
            self.lr = tf.Variable(0.0, trainable=False)
            trainable_vars = tf.trainable_variables()

            # clip the gradient to avoid vanishing or blowing up gradients
            # max_gradient_norm/sqrt(add each element square))
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), CONFIGS.max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_optimizer = optimizer.apply_gradients(zip(grads, trainable_vars))

        ####################################################################
        # for model saving
        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(tf.global_variables())

    def step(self, sess, batch_X, batch_seq_lens, batch_y=None, dropout=0.0, forward_only=True, predict=False, merged_summary=None):

        input_feed = {self.inputs_X: batch_X, self.targets_y: batch_y, self.seq_lens: batch_seq_lens,
                      self.dropout: dropout}

        if forward_only:
            if not predict:
                output_feed = [self.accuracy]
            elif predict:
                input_feed = {self.inputs_X: batch_X, self.seq_lens: batch_seq_lens,
                              self.dropout: dropout}
                output_feed = [self.probabilities]
        else:  # training
            output_feed = [self.train_optimizer, self.loss, self.accuracy, merged_summary]

        outputs = sess.run(output_feed, input_feed)

        if forward_only:
            return outputs[0]
        else:  # training
            return outputs[0], outputs[1], outputs[2], outputs[3]
