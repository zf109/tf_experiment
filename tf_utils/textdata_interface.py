import numpy as np
from collections import Counter
import tensorflow as tf

class TextDataInterface(object):
    def __init__(self, doc_list_w, doc_identifiers, words=None, seq_len=None, labels=None):
        """
            :doc_list_w: list of docs in the form of list of words 
        """

        self.doc_list_w = doc_list_w
        self.seq_len = seq_len if seq_len else len(doc_list_w[0]) + int(len(doc_list_w[0]) * .1)

        self.words = words if words else self.get_words(doc_list_w)
        self.n_words = len(self.words)

        self.features = self.to_features_mat(self.int_encoding(self.doc_list_w), self.seq_len)
        self.targets = labels

    @classmethod
    def from_text(cls, text: str, doc_delimiter='\n', doc_identifiers=None, words=None, seq_len=None, labels=None):
        strlist = text.split(doc_delimiter)
        return cls.from_strlist(strlist=strlist, doc_identifiers=doc_identifiers, words=words, seq_len=seq_len, labels=labels)

    @classmethod
    def from_strlist(cls, strlist, doc_identifiers=None, words=None, seq_len=None, labels=None):
        doc_list_w = [doc.split() for doc in strlist]
        return cls(doc_list_w, doc_identifiers, words, seq_len, labels)

    @staticmethod
    def get_words(doc_list_w):
        return [w for wordlist in doc_list_w for w in wordlist]

    def int_encoding(self, doc_list_w):
        counts = Counter(self.words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

        doc_ints = []
        for doc_words in doc_list_w:
            doc_ints.append([vocab_to_int[word] for word in doc_words])
        return doc_ints

    @staticmethod
    def to_features_mat(doc_ints, seq_len):
        features = np.zeros((len(doc_ints), seq_len), dtype=int)
        for i, row in enumerate(doc_ints):
            features[i, -len(row):] = np.array(row)[:seq_len]
        return features

    def simple_split(self):
        """
            This function does not provide shuffle
        """
        features = self.features
        labels = np.array(self.targets)
        split_frac = 0.8
        split_idx = int(len(features)*0.8)
        train_x, val_x = features[:split_idx], features[split_idx:]
        train_y, val_y = labels[:split_idx], labels[split_idx:]

        test_idx = int(len(val_x)*0.5)
        val_x, test_x = val_x[:test_idx], val_x[test_idx:]
        val_y, test_y = val_y[:test_idx], val_y[test_idx:]

        return train_x, train_y, test_x, test_y


class TextClassificationGraph(object):
    def __init__(self,
        n_words,
        embed_size=300,
        lstm_size=256,
        lstm_layers=1,
        learning_rate=.001,
        batch_size =500
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        graph = tf.Graph()
        with graph.as_default():
            # io layer 
            self.inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
            self.labels_ = tf.placeholder(tf.int32, [None, None], name='labbels')

            # keep probability for drop-out practice
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # embedding layer    
            embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, self.inputs_)

            # lstm layers
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
            self.cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers) # stack multiple number of lstm

            self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

            # output layers
            outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, embed, initial_state=self.initial_state)
            predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
            self.cost = tf.losses.mean_squared_error(self.labels_, predictions)

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
            correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), self.labels_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        self.graph = graph


class LSTMbuilder(object):
    def __init__(self,
                graph,
                accuracy,
                initial_state,
                final_state,
                inputs,
                labels,
                keep_prob,
                cell,
                cost,
                optimizer,
                epochs=10,
                batch_size=500
    ):

        self.graph = graph
        self.epochs = epochs
        self.batch_size = batch_size
        self.accuracy = accuracy
        self.initial_state = initial_state
        self.final_state = final_state
        self.inputs = inputs
        self.labels = labels
        self.keep_prob = keep_prob
        self.cell = cell
        self.cost = cost
        self.optimizer = optimizer

        with self.graph.as_default():
            self.saver = tf.train.Saver()

    @classmethod
    def from_txtclfgraph(cls, tcg: TextClassificationGraph, epochs=10):
        return cls(graph=tcg.graph,
                    accuracy=tcg.accuracy,
                    initial_state=tcg.initial_state,
                    final_state=tcg.final_state,
                    inputs=tcg.inputs_,
                    labels=tcg.labels_,
                    keep_prob=tcg.keep_prob,
                    cell=tcg.cell,
                    cost=tcg.cost,
                    optimizer=tcg.optimizer,
                    epochs=epochs, 
                    batch_size=tcg.batch_size
        )

    def train(self, train_x, train_y):
 
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            for e in range(self.epochs):
                state = sess.run(self.initial_state)
                
                for ii, (x, y) in enumerate(self.get_batches(train_x, train_y, self.batch_size), 1):
                    feed = {self.inputs: x,
                            self.labels: y[:, None],
                            self.keep_prob: 0.5,
                            self.initial_state: state}
                    loss, state, _ = sess.run([self.cost, self.final_state, self.optimizer], feed_dict=feed)
                    
                    if iteration%5==0:
                        print("Epoch: {}/{}".format(e, self.epochs),
                            "Iteration: {}".format(iteration),
                            "Train loss: {:.3f}".format(loss))

                    if iteration%25==0:
                        val_acc = []
                        val_state = sess.run(cell.zero_state(self.batch_size, tf.float32))
                        for x, y in self.get_batches(val_x, val_y, self.batch_size):
                            feed = {self.inputs: x,
                                    self.labels: y[:, None],
                                    self.keep_prob: 1,
                                    self.initial_state: val_state}
                            batch_acc, val_state = sess.run([self.accuracy, self.final_state], feed_dict=feed)
                            val_acc.append(batch_acc)
                        print("Val acc: {:.3f}".format(np.mean(val_acc)))
                    iteration +=1
            self.saver.save(sess, "checkpoints/sentiment.ckpt")        

    @staticmethod
    def get_batches(x, y, batch_size=100):

        n_batches = len(x)//batch_size
        x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
        for ii in range(0, len(x), batch_size):
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]

    def test(self, test_x, test_y):
        test_acc = []
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            test_state = sess.run(self.cell.zero_state(batch_size, tf.float32))
            for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
                feed = {self.inputs: x,
                        self.labels: y[:, None],
                        self.keep_prob: 1,
                        self.initial_state: test_state}
                batch_acc, test_state = sess.run([self.accuracy, self.final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


if __name__ == "__main__":
    # test_texts = [
    #     "this is the first line",
    #     "this is the second line",
    #     "I thought I am the second",
    #     "the quick brown fox jumps over the lazy dog :p",
    #     "That is an old joke in text analysis, haha",
    #     "Yes, this text parsing is meant to be hard",
    #     "so that I can really test how the harder texts are parsed",
    #     "is you a stop word?"
    # ]
    import os
    load_location ='movie_data'
    reviews_file = os.path.join(load_location, 'reviews.txt')
    labbels_file = os.path.join(load_location, 'labels.txt')
    with open(reviews_file, 'r') as f:
        reviews = f.read()
    with open(labbels_file, 'r') as f:
        labels = f.read()

    labels = labels.split('\n')
    labels = np.array([1 if each == 'positive' else 0 for each in labels])

    # convert reviews to string list
    reviews = reviews.split('\n')
    nz_idx = [i for i, r in enumerate(reviews) if len(r) > 0]
    reviews = [reviews[ii] for ii in nz_idx]
    labels = [labels[ii] for ii in nz_idx]

    # ti = TextDataInterface.from_strlist(strlist=test_texts, seq_len=20)
    ti = TextDataInterface.from_strlist(strlist=reviews, labels=labels, seq_len=200)
    print("features are:")
    print(ti.features)

    print("buiding graph...")
    gb = TextClassificationGraph(n_words=ti.n_words)

    print("building model...")
    mb = LSTMbuilder.from_txtclfgraph(gb)

    print("start training...")
    train_x, train_y, test_x, test_y = ti.simple_split()
    mb.train(train_x=train_x, train_y=train_y)

    print("testing")
    mb.test(test_x=test_x, test_y=test_y)











