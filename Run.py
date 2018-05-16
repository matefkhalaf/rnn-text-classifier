import os
import datetime as dt
import tensorflow as tf
import numpy as np
from Utilities import generate_epoch, load_data_and_labels, load_glove_model, data_to_vectors, split_data
from Model import model
from tensorflow.python import debug as tf_debug

# Configurations
tf.app.flags.DEFINE_string("rnn_unit", 'rnn', "Type of RNN unit: rnn|gru|lstm.")
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.999, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm (clipping ratio).")
tf.app.flags.DEFINE_integer("num_epochs", 65, "Number of epochs during training.")
tf.app.flags.DEFINE_integer("test_num_epochs", 1, "Number of epochs during testing.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_hidden_units", 200, "Number of hidden units in each RNN cell (i/p vector length).")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_float("dropout", 0.5, "Amount to drop during training.")
tf.app.flags.DEFINE_string("ckpt_dir", "checkpoints", "Directory to save the model checkpoints")
tf.app.flags.DEFINE_string("tensorboard_dir", "tensorboard", "Directory to save tensorboard files")
tf.app.flags.DEFINE_integer("num_classes", 6, "Number of classification classes.")
tf.app.flags.DEFINE_float("train_ratio", 0.6, "Percentage of total data to be trained.")
CONFIGS = tf.app.flags.FLAGS


# Create Model

def create_model(sess, CONFIGS):
    text_model = model(CONFIGS)
    print("Created new model.")
    sess.run(tf.global_variables_initializer())

    return text_model


def run_model():
    tf.reset_default_graph()
    X, Y = load_data_and_labels()
    glove_model = load_glove_model()
    data, seqs = data_to_vectors(X, glove_model, CONFIGS.num_hidden_units)
    train_X, train_y, train_seq_lens, test_X, test_y, test_seq_lens = split_data(data, Y, seqs, train_ratio=CONFIGS.train_ratio)

    print("DATA IS LOADED")
    with tf.Session() as sess:
        # Load old model or create new one
        model = create_model(sess, CONFIGS)
        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(CONFIGS.tensorboard_dir) # A python class that writes data for tensorboard
        writer.add_graph(sess.graph)

        print("STARTING TRAINING")
        print("----------------------")
        # summ_count --> counter for tensorboard
        summ_count = 0
        # Train results
        for epoch_num, epoch in enumerate(generate_epoch(train_X, train_y, train_seq_lens,
                                                         CONFIGS.num_epochs, CONFIGS.batch_size)):
            print("EPOCH #%i started:" % (epoch_num + 1))
            print("----------------------")

            # Assign learning rate
            sess.run(tf.assign(model.lr, CONFIGS.learning_rate *
                               (CONFIGS.learning_rate_decay_factor ** epoch_num)))

            train_loss = []
            train_accuracy = []
            curr_time = dt.datetime.now()
            for batch_num, (batch_X, batch_y, batch_seq_lens) in enumerate(epoch):
                _, loss, accuracy, summ = model.step(sess, batch_X, batch_seq_lens, batch_y, dropout=CONFIGS.dropout,
                                                     forward_only=False,
                                                     merged_summary=merged_summary)

                train_loss.append(loss)
                train_accuracy.append(accuracy)
                #print("Epoch {}, Step {}, loss: {:.3f}, accuracy: {:.3f}".format(epoch_num,batch_num, loss, accuracy))

                writer.add_summary(summ, summ_count)
                summ_count = summ_count + 1


            seconds = (float((dt.datetime.now() - curr_time).seconds))
            print()
            print("EPOCH #%i SUMMARY" % (epoch_num + 1))
            print("Total Average Training loss %.3f" % np.mean(train_loss))
            print("Total Average Training accuracy %.3f" % np.mean(train_accuracy))
            print("Time taken (seconds) %.3f" % seconds)
            print("----------------------")
        print("TRAINING ENDED")
        print("----------------------")
        # Save final ckpt.
        if not os.path.isdir(CONFIGS.ckpt_dir):
            os.makedirs(CONFIGS.ckpt_dir)
        checkpoint_path = os.path.join(CONFIGS.ckpt_dir, "model.ckpt")
        print("Saving final model")
        model.saver.save(sess, checkpoint_path)
        print("STARTING TESTING")
        print("----------------------")
        # Test results
        test_accuracy = []
        curr_time = dt.datetime.now()
        for test_epoch_num, test_epoch in enumerate(generate_epoch(test_X, test_y, test_seq_lens,
                                                         CONFIGS.test_num_epochs, CONFIGS.batch_size)):

            for test_batch_num, (test_batch_X, test_batch_y, test_batch_seq_lens) in enumerate(test_epoch):

                accuracy = model.step(sess, test_batch_X, test_batch_seq_lens, test_batch_y, dropout=0.0, forward_only=True)
                test_accuracy.append(accuracy)
        seconds = (float((dt.datetime.now() - curr_time).seconds))
        print("Total Testing accuracy %.3f" % np.mean(test_accuracy))
        print("Time taken (seconds) %.3f" % seconds)
        print("TESTING ENDED")
        print("----------------------")
        writer.close()

if __name__ == '__main__':
    run_model()
