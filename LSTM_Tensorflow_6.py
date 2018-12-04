# import the packages
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
import os
import datetime

# Training Parameters
learning_rate = 0.005
# training_steps = 5000
# batch_size = 250
display_step = 100

# Network Parameters
num_input = 1 # the stock price
timesteps = 240 # timesteps
num_hidden = 25 # hidden layer num of features
num_classes = 1 # above or below the median
dropout = 0.1
threshold = tf.constant(0.5)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=0.8)

    # Apply the Dropout
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0 - dropout,
                                              state_keep_prob=1.0 - dropout)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.sigmoid(logits) # for prediction, [0, 1]

# Define loss and optimizer
x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
loss_op = tf.reduce_mean(x_entropy)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
delta = tf.abs((Y - prediction))
correct_pred = tf.cast(tf.less(delta, threshold), tf.int32)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# the tool to save the results
saver = tf.train.Saver()

for i in range(12, -1, -1):
    # read the data
    training_name = os.path.join(os.getcwd(), 'Normalize_Return\\Set_' + str(i) + '_Train.csv')
    testing_name = os.path.join(os.getcwd(), 'Normalize_Return\\Set_' + str(i) + '_Test.csv')
    train_data = pd.read_csv(training_name, index_col=0)
    test_data = pd.read_csv(testing_name, index_col=0)

    training_label = train_data.iloc[:, timesteps]
    training_data = train_data.iloc[:, :timesteps]
    testing_label = test_data.iloc[:, timesteps]
    testing_data = test_data.iloc[:, :timesteps]

    stocks = train_data.ticker.unique()
    training_steps = len(stocks)

    # Start training
    with tf.Session() as sess:
        # print the training info
        print("-------------------------------------------------------------------------------------------------------")
        print("Training the model for Training Set " + str(i) + " from " +
              datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + "...")
        print("-------------------------------------------------------------------------------------------------------")

        # Run the initializer
        sess.run(init)

        # Restore model weights from previously saved model
        if i != 12:
            load_path = saver.restore(sess, log_path)
            print("Model restored from file: %s" % save_path)

        for step in range(training_steps):
            stock = stocks[step]
            batch = train_data[train_data.ticker == stock]
            batch = batch.sort_values('target_date')
            batch_size = len(batch)

            # query the data from the data set
            batch_x = np.array(batch.iloc[:, :timesteps])
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            batch_y = np.array(batch.iloc[:, timesteps])
            batch_y = batch_y.reshape((batch_size, num_classes))

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss = " + \
                      "{:.4f}".format(loss) + ", Training Accuracy = " + \
                      "{:.3f}".format(acc))

        testing_data = np.array(testing_data).reshape((len(testing_data), timesteps, num_input))
        testing_label = np.array(testing_label).reshape((len(testing_label), num_input))
        training_data = np.array(training_data).reshape((len(training_data), timesteps, num_input))
        training_label = np.array(training_label).reshape((len(training_label), num_input))
        print("Overall Training Accuracy:", sess.run(accuracy, feed_dict={X: training_data, Y: training_label}))
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: testing_data, Y: testing_label}))

        log_path = os.path.join(os.getcwd(), 'Logs\\model_for_period_' + str(i))
        save_path = saver.save(sess, log_path)
        print("Model saved in file: %s" % save_path)

        pred = sess.run(prediction, feed_dict={X: testing_data, Y: testing_label})
        pred = pred.reshape((1, len(pred))).tolist()[0]
        output_data = pd.DataFrame({'pred': pred, 'y': test_data['target'], 'ticker': test_data['ticker'],
                                    'date': test_data['target_date']})
        output_path = os.path.join(os.getcwd(), 'Pred\\prediction_period_' + str(i) + '.csv')
        output_data.to_csv(output_path)
        print('Prediction for period ' + str(i) + ' successfully saved.')