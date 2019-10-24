import os
import urllib
import urllib.request
import argparse
import os
import numpy as np
from azureml.core import Run
import tensorflow as tf

def get_args():
    global input_data_location
    global n_inputs
    global n_h1
    global n_h2
    global n_outputs
    global learning_rate
    global n_epochs
    global batch_size
    global release_id
    global model_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_location", type=str, help="input data directory")
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
    parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                        help='# of neurons in the first layer')
    parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                        help='# of neurons in the second layer')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
    parser.add_argument('--release_id', type=int, dest='release_id', default=0, help='Release ID')
    parser.add_argument('--model_name', type=str, dest='model_name', default='tf_mnist_pipeline.model', help='Model Name')
    
    args = parser.parse_args()

    n_inputs = 28 * 28
    n_outputs = 10
    n_epochs = 20
    
    input_data_location = args.input_data_location
    n_h1 = args.n_hidden_1
    n_h2 = args.n_hidden_2
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    release_id = args.release_id
    model_name = args.model_name

    print('Argument input_data_location: {}'.format(input_data_location))
    print('Argument n_h1: {}'.format(n_h1))
    print('Argument n_h2: {}'.format(n_h2))
    print('Argument learning_rate: {}'.format(learning_rate))
    print('Argument batch_size: {}'.format(batch_size))
    print('Argument release_id: {}'.format(release_id))
    print('Argument model_name: {}'.format(model_name))

def load_data():
    global training_set_size
    global X_train
    global y_train
    global X_test
    global y_test

    # note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the neural network converge faster.
    X_train = np.load('{}/X_train.npy'.format(input_data_location))
    y_train = np.load('{}/y_train.npy'.format(input_data_location))
    X_test = np.load('{}/X_test.npy'.format(input_data_location))
    y_test = np.load('{}/y_test.npy'.format(input_data_location))

    training_set_size = X_train.shape[0]

    print('Numpy binary data is loaded')

def train():
    with tf.name_scope('network'):
        # construct the DNN
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
        y = tf.placeholder(tf.int64, shape=(None), name='y')
        h1 = tf.layers.dense(X, n_h1, activation=tf.nn.relu, name='h1')
        h2 = tf.layers.dense(h1, n_h2, activation=tf.nn.relu, name='h2')
        output = tf.layers.dense(h2, n_outputs, name='output')

    with tf.name_scope('train'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(output, y, 1)
        acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    run_details = run.get_details()
    experiment_name = run_details['runDefinition']['environment']['name'].split()[1]

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):

            # randomly shuffle training set
            indices = np.random.permutation(training_set_size)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]

            # batch index
            b_start = 0
            b_end = b_start + batch_size
            for _ in range(training_set_size // batch_size):
                # get a batch
                X_batch, y_batch = X_train_sample[b_start: b_end], y_train_sample[b_start: b_end]

                # update batch index for the next batch
                b_start = b_start + batch_size
                b_end = min(b_start + batch_size, training_set_size)

                # train
                sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
            # evaluate training set
            acc_train = acc_op.eval(feed_dict={X: X_batch, y: y_batch})
            # evaluate validation set
            acc_val = acc_op.eval(feed_dict={X: X_test, y: y_test})

            # log accuracies
            run.log('accuracy-train', np.float(acc_train))
            run.log('accuracy-val', np.float(acc_val))
            print(epoch, '-- Training accuracy:', acc_train, '\b Validation accuracy:', acc_val)
            y_hat = np.argmax(output.eval(feed_dict={X: X_test}), axis=1)

        run.log('final-accuracy', np.float(acc_val))

        os.makedirs('./outputs/model', exist_ok=True)
        # files saved in the "./outputs" folder are automatically uploaded into run history
        saver.save(sess, './outputs/model/{}'.format(model_name))

        # run.upload_file(name="./outputs/" + 'model/mnist-tf.model', path_or_stream='./outputs/model/mnist-tf.model')
        # print("Uploaded the model {} to experiment {}".format('mnist-tf.model', run.experiment.name))
        # dirpath = os.getcwd()
        # print(dirpath)
        # print("Following files are uploaded ")
        # print(run.get_file_names())

        run.add_properties({"release_id": release_id, "run_type": "train"})
        print(f"added properties: {run.properties}")

        # tags = {}
        # tags['run_id'] = run.id
        # tags['final_accuracy'] = np.float(acc_val)
        # 
        # run.register_model(model_name='mnist_model_with_pipeline',
        #                    model_path='outputs/model',
        #                    tags=tags)

if __name__ == '__main__':
    global run
    run = Run.get_context()
    get_args()
    load_data()
    train()
