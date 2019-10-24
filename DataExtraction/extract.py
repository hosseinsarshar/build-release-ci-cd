import os
import urllib
import urllib.request
import argparse
import os
import numpy as np
from azureml.core import Run
# import matplotlib
# import matplotlib.pyplot as plt

def get_args():
    global output_extract
    global raw_folder_address

    parser = argparse.ArgumentParser("extract")
    parser.add_argument("--output_extract", type=str, help="output_extract directory")

    args = parser.parse_args()
    output_extract = args.output_extract

    print("Argument 1: %s" % output_extract)
    raw_folder_address = '{}/raw_files/'.format(output_extract)


def download_files():
    os.makedirs(output_extract, exist_ok=True)
    os.makedirs(raw_folder_address, exist_ok=True)
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename = '{}/train-images.gz'.format(raw_folder_address))
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename = '{}/train-labels.gz'.format(raw_folder_address))
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename = '{}/test-images.gz'.format(raw_folder_address))
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename = '{}/test-labels.gz'.format(raw_folder_address))

    print("Files are downloaded to {}".format(raw_folder_address))
    return raw_folder_address


def normalize_inuputs():
    global X_train

    from utils import load_data

    # note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the neural network converge faster.
    X_train = load_data('{}/train-images.gz'.format(raw_folder_address), False) / 255.0
    y_train = load_data('{}/train-labels.gz'.format(raw_folder_address), True).reshape(-1)
    X_test = load_data('{}/test-images.gz'.format(raw_folder_address), False) / 255.0
    y_test = load_data('{}/test-labels.gz'.format(raw_folder_address), True).reshape(-1)

    run.log('train_dataset_size', X_train.shape[0])
    run.log('test_dataset_size', X_test.shape[0])

    np.save('{}/X_train'.format(output_extract), X_train)
    np.save('{}/y_train'.format(output_extract), y_train)
    np.save('{}/X_test'.format(output_extract), X_test)
    np.save('{}/y_test'.format(output_extract), y_test)

    print('Normalized numpy binary files are saved at: {}'.format(output_extract))

# def log_sample_image():
#     count = 0
#     sample_size = 30
#     plt.figure(figsize = (16, 6))
#     for i in np.random.permutation(X_train.shape[0])[:sample_size]:
#         count = count + 1
#         plt.subplot(1, sample_size, count)
#         plt.axhline('')
#         plt.axvline('')
#         plt.text(x = 10, y = -10, s = y_train[i], fontsize = 18)
#         plt.imshow(X_train[i].reshape(28, 28), cmap = plt.cm.Greys)
#         
#     run.log_image(name='{}-samples-of-input-dataset'.format(sample_size), plot=plt)
#     plt.close()

## Todo: Add sample image log

if __name__ == '__main__':
    global run
    run = Run.get_context()
    get_args()
    download_files()
    normalize_inuputs()
    # log_sample_image()
