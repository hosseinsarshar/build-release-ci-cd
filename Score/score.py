import json
import numpy as np
import os
import tensorflow as tf
import time
from azureml.core.model import Model

def init():
    global X, output, sess
    print ("model initializing at " + time.strftime("%H:%M:%S"))
    model_name = "tf_mnist_pipeline_devops.model"
    tf.reset_default_graph()
    model_root = Model.get_model_path(model_name)
    saver = tf.train.import_meta_graph(os.path.join(model_root, f'{model_name}.meta'))
    X = tf.get_default_graph().get_tensor_by_name("network/X:0")
    output = tf.get_default_graph().get_tensor_by_name("network/output/MatMul:0")
    
    sess = tf.Session()
    saver.restore(sess, os.path.join(model_root, model_name))
    print ("model initialized at " + time.strftime("%H:%M:%S"))

def run(raw_data):
    print ("data scoring at " + time.strftime("%H:%M:%S"))
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    out = output.eval(session=sess, feed_dict={X: data})
    y_hat = np.argmax(out, axis=1)
    print ("data scored at " + time.strftime("%H:%M:%S"))
    return y_hat.tolist()
