import numpy as np
import gensim
import os

def load_data_and_labels():
    # Load the data
    alarm = list(open("data/alarm.txt", "r", encoding="utf-8").readlines())
    alarm = [s.strip() for s in alarm]
    alarm = [[w for w in sent.strip().split()] for sent in alarm]
    application = list(open("data/application.txt", "r", encoding="utf-8").readlines())
    application = [s.strip() for s in application]
    application = [[w for w in sent.strip().split()] for sent in application]
    call = list(open("data/call.txt", "r", encoding="utf-8").readlines())
    call = [s.strip() for s in call]
    call = [[w for w in sent.strip().split()] for sent in call]
    messages = list(open("data/messages.txt", "r", encoding="utf-8").readlines())
    messages = [s.strip() for s in messages]
    messages = [[w for w in sent.strip().split()] for sent in messages]
    notes = list(open("data/notes.txt", "r", encoding="utf-8").readlines())
    notes = [s.strip() for s in notes]
    notes = [[w for w in sent.strip().split()] for sent in notes]
    playsong = list(open("data/playsong.txt", "r", encoding="utf-8").readlines())
    playsong = [s.strip() for s in playsong]
    playsong = [[w for w in sent.strip().split()] for sent in playsong]
    X = alarm + application + call + messages + notes + playsong

    # Labels
    # One-hot [ alarm, application, call, messages, notes, playsong]
    alarm_l = [[1,0,0,0,0,0] for _ in alarm]
    application_l = [[0, 1, 0, 0, 0, 0] for _ in application]
    call_l = [[0, 0, 1, 0, 0, 0] for _ in call]
    messages_l = [[0, 0, 0, 1, 0, 0] for _ in messages]
    notes_l = [[0, 0, 0, 0, 1, 0] for _ in notes]
    playsong_l = [[0, 0, 0, 0, 0, 1] for _ in playsong]
    y = np.concatenate([alarm_l, application_l, call_l, messages_l, notes_l, playsong_l], 0)

    print("Loaded Data Info: Total: %i, alarm: %i, application: %i call: %i messages: %i notes: %i playsong: %i"
          "" % (len(y), np.sum(y[:, 0]), np.sum(y[:, 1]), np.sum(y[:, 2]),
                np.sum(y[:, 3]), np.sum(y[:, 4]), np.sum(y[:, 5])))

    return X, y


def load_glove_model():
    path = os.path.join("data", "Arabic_vectors_64bit_200.txt")
    glove_model = gensim.models.KeyedVectors.load_word2vec_format(path)
    print("GLOVE MODEL LOADED")
    return glove_model


def sentence_to_vectors(sentence, glove_model, num_hidden_units):
    return [glove_model.wv[word].tolist() if word in glove_model.wv.vocab else [0.0] * num_hidden_units for word in sentence]


def data_to_vectors(X, glove_model, num_hidden_units):
    max_len = max(len(sentence) for sentence in X)
    seq_lens = []

    data_as_vectors = []
    for line in X:
        vectors = sentence_to_vectors(line, glove_model, num_hidden_units)
        # Padding
        data_as_vectors.append(vectors + [[0.0] * num_hidden_units] * (max_len - len(line)))
        # Maintain original seq lengths for dynamic RNN
        seq_lens.append(len(line))

    return data_as_vectors, seq_lens


def split_data(X, y, seq_lens, train_ratio=0.8):
    X = np.array(X)
    seq_lens = np.array(seq_lens)
    data_size = len(X)

    # Shuffle the data
    shuffle_indices = np.random.permutation(np.arange(data_size))
    X, y, seq_lens = X[shuffle_indices], y[shuffle_indices], \
                     seq_lens[shuffle_indices]

    # Split into train and test set
    train_end_index = int(train_ratio * data_size)
    train_X = X[:train_end_index]
    train_y = y[:train_end_index]
    train_seq_lens = seq_lens[:train_end_index]
    valid_X = X[train_end_index:]
    valid_y = y[train_end_index:]
    valid_seq_lens = seq_lens[train_end_index:]

    return train_X, train_y, train_seq_lens, valid_X, valid_y, valid_seq_lens


def generate_epoch(X, y, seq_lens, num_epochs, batch_size):
    for epoch_num in range(num_epochs):
        yield generate_batch(X, y, seq_lens, batch_size)


def generate_batch(X, y, seq_lens, batch_size):
    data_size = len(X)

    num_batches = (data_size // batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield X[start_index:end_index], y[start_index:end_index], \
              seq_lens[start_index:end_index]

