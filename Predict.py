import tensorflow as tf
import numpy as np
from Utilities import load_glove_model
from Model import model
from Run import CONFIGS


def create_model(sess, CONFIGS):

    text_model = model(CONFIGS)

    checkpt = tf.train.get_checkpoint_state(CONFIGS.ckpt_dir)
    if checkpt:
        print("Restoring old model parameters from %s" % checkpt.model_checkpoint_path)
        text_model.saver.restore(sess, checkpt.model_checkpoint_path)

    return text_model


def predict(glove_model, input):
    tf.reset_default_graph()
    sentence = [w for w in input.strip().split()]
    data = [glove_model.wv[word].tolist() if word in glove_model.wv.vocab else [0.0] * CONFIGS.num_hidden_units for word
            in sentence]
    seqs = len(sentence)
    data = np.array(data)
    data = data[np.newaxis, :, :]
    seqs = np.array(seqs)
    seqs = seqs[np.newaxis]

    with tf.Session() as sess:
        # Load old model
        model = create_model(sess, CONFIGS)

        probabilities = model.step(sess, batch_X=data,
                                   batch_seq_lens=seqs,
                                   forward_only=True, predict=True)
        # One-hot [ alarm, application, call, messages, notes, playsong]
        classes = ["alarm", "application", "call", "messages", "notes", "playsong"]
        print("probabilities: ", probabilities)
        predict_class = classes[np.asscalar(np.argmax(probabilities, 1))]
        return predict_class


