#!/usr/bin/env python3
""" the train file """


import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """ that builds, trains, and saves a neural network classifier:
        X_train is a numpy.ndarray containing the training input data
        Y_train is a numpy.ndarray containing the training labels
        X_valid is a numpy.ndarray containing the validation input data
        Y_valid is a numpy.ndarray containing the validation labels
        layer_sizes is a list containing the number of nodes
            in each layer of the network
        activations is a list containing the activation functions
            for each layer of the network
        alpha: is the learning rate
        iterations is the number of iterations to train over
        save_path: designates where to save the model
        Returns: path where model was saved
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection("y", y)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    saver = tf.train.Saver()

    # sess = tf.Session()
    init = tf.global_variables_initializer()
    # sess.run()#needs init of some sort?
    feed_dict_train = {x: X_train, y: Y_train}
    feed_dict_valid = {x: X_valid, y: Y_valid}

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            lt = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            at = sess.run(accuracy, feed_dict=feed_dict_train)
            lv = sess.run(loss, feed_dict=feed_dict_valid)
            av = sess.run(accuracy, feed_dict=feed_dict_valid)
            if i % 100 == 0:
                print("after {} iterations:".format(i))
                print("\tTraining Cost: {}".format(lt))
                print("\tTraining Accuracy: {}".format(at))
                print("\tValidation Cost: {}".format(lv))
                print("\tValidation Accuracy: {}".format(av))
            y_pred = forward_prop(x, layer_sizes, activations)
            sess.run(train_op, feed_dict=feed_dict_train)

        lt = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        at = sess.run(accuracy, feed_dict=feed_dict_train)
        lv = sess.run(loss, feed_dict=feed_dict_valid)
        av = sess.run(accuracy, feed_dict=feed_dict_valid)

        print("after {} iterations:".format(iterations))
        print("\tTraining Cost: {}".format(lt))
        print("\tTraining Accuracy: {}".format(at))
        print("\tValidation Cost: {}".format(lv))
        print("\tValidation Accuracy: {}".format(av))

        saved = saver.save(sess, save_path)
    return saved
