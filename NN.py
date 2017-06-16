import tensorflow as tf
from sklearn import cross_validation
import load_data as ld

data_X, data_Y = ld.LoadData()
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data_X, data_Y, test_size=0.2, random_state=0)

iter = 500
size = X_train.shape[0]
batch_size = 50
epoch = int(size/batch_size)

sess = tf.Session()

with sess.as_default():

    x = tf.placeholder("float", shape=(None, X_train.shape[1]))
    y_ = tf.placeholder("float")

    # W = tf.Variable(tf.zeros([X_train.shape[1],1]))
    W = tf.Variable(tf.truncated_normal([X_train.shape[1],1], stddev=0.1))
    b = tf.Variable(0.)
    y = tf.add(tf.matmul(x,W), b)

    Loss = tf.reduce_sum(tf.square(tf.subtract(y,y_)))
    train_step = tf.train.AdamOptimizer(0.00001).minimize(Loss)
    # train_step = tf.train.GradientDescentOptimizer(0.000000000000001).minimize(Loss)

    sess.run(tf.initialize_all_variables())

    for i in range(iter):
        epoch_loss = 0
        for j in range(epoch):
            start = j*batch_size
            end = j*batch_size + 50
            batch_x = X_train[start:end]
            batch_y = Y_train[start:end]
            _,c = sess.run([train_step, Loss], feed_dict={x:batch_x, y_:batch_y})
            epoch_loss += c
        print("Iter: ", i, " Epoch: ", epoch, " Loss: ", epoch_loss)

    accuracy = tf.reduce_mean(tf.abs(tf.subtract(y_,y)))
    print(accuracy.eval(feed_dict={x:X_test, y_:Y_test}))