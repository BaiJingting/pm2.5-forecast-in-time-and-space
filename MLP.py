import tensorflow as tf
from sklearn import cross_validation
import load_data as ld

data_X, data_Y = ld.LoadData()
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data_X, data_Y, test_size=0.2, random_state=0)

size = 100
# sizes = [100,100]
iter = 200
batch_size = 50
epoch = int(X_train.shape[0]/batch_size)

def Layer(input, in_size, out_size, active_function = None):
    W = tf.Variable(tf.zeros([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]))
    output = tf.add(tf.matmul(input, W), b)
    if(active_function!=None):
        output = active_function(output)
    return output

sess = tf.Session()
with sess.as_default():
    x = tf.placeholder("float", shape=[None, X_train.shape[1]])
    y_ = tf.placeholder("float")

    l1 = Layer(x, X_train.shape[1], size)
    y = Layer(l1, size, 1)
    # l2 = Layer(l1, sizes[0], sizes[1])
    # y = Layer(l2, sizes[1], 1)

    loss = tf.reduce_sum(tf.square(y_-y))
    train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

    sess.run( tf.initialize_all_variables())

    for i in range(iter):
        epoch_loss = 0
        for j in range(epoch):
            start = j*batch_size
            end = (j+1)*batch_size
            batch_x = X_train[start:end]
            batch_y = Y_train[start:end]
            _,c = sess.run([train_step, loss], feed_dict={x:batch_x, y_:batch_y})
            epoch_loss += c
        print("Iter: ", i, "loss: ", epoch_loss)

    accuracy = tf.reduce_mean(tf.abs(y_-y))
    print(sess.run(accuracy, feed_dict={x:X_test, y_:Y_test}))