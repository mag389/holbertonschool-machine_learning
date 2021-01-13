#!/usr/bin/env python3
""" trains a loaded neural network using mini-batch gradient descent """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ trains with minibatch grad descent
        X_train: np.ndarray (m, 784) of training data
            m data points, 784 input features
        Y_train: one-hot np.ndarray (m, 10) of training labels
            10 classes
        X_valid: np.ndarray (m, 784) of validation data
        Y_valid: one-hot np.ndarray (m, 10) of validation labels
        batch_size: number of data points in a batch
        epochs: number of times training should pass through whole dataset
        load_path: path from which to load model
        save_path: path where the model should be saved
        Returns: path where model was saved
    """
    with tf.Session() as sess:
        # first load the past data
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        feed_dict_train = {x: X_train, y: Y_train}
        feed_dict_valid = {x: X_valid, y: Y_valid}

        for i in range(epochs + 1):
            lt = sess.run(loss, feed_dict_train)
            at = sess.run(accuracy, feed_dict_train)
            lv = sess.run(loss, feed_dict_valid)
            av = sess.run(accuracy, feed_dict_valid)

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(lt))
            print("\tTraining Accuracy: {}".format(at))
            print("\tValidation Cost: {}".format(lv))
            print("\tValidation Accuracy: {}".format(av))

            if i == epochs:
                continue
            X_train, Y_train = shuffle_data(X_train, Y_train)

            # then loop through the batch, training on each mini batch
            for j in range(0, int(X_train.shape[0] / batch_size + 1)):
                lower = j * batch_size
                upper = lower + batch_size
                if j == int(X_train.shape[0] / batch_size + 1):
                    upper == lower + X_train.shape[0] % batch_size
                mini_batch = {x: X_train[lower: upper],
                              y: Y_train[lower: upper]}
                sess.run(train_op, mini_batch)
                if j != 0 and j % 100 == 0:
                    ltm = sess.run(loss, feed_dict=mini_batch)
                    atm = sess.run(accuracy, feed_dict=mini_batch)
                    print("\tStep {}:".format(j))
                    print("\t\tCost: {}".format(ltm))
                    print("\t\tAccuracy: {}".format(atm))
        saved = saver.save(sess, save_path)
    return saved
