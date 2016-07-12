import cPickle
import os
from lib.htkmfc import HTKFeat_read
import numpy as np

__author__ = 'yossiadi'

# globals
rho = 360


# TODO also look at https://github.com/fchollet/keras/issues/68 for big datasets
def get_data(path):
    """
    Reads the data as numpy d-array
    :param path: the path to the data
    :return: Numpy d-array for x and y
    """
    # validation
    if not path:
        return None

    f = file(path, 'rb')
    loaded_objects = []
    for i in range(2):
        loaded_objects.append(cPickle.load(f))
    f.close()

    # x = split_2_time_steps(loaded_objects[0])  # get the features first
    # y = split_2_time_steps(loaded_objects[1])  # get the labels

    return loaded_objects[0], loaded_objects[1]


# split to time steps for the recurrent steps (BPTT)
def split_2_time_steps(data):
    # get the features first
    t = 1
    x_raw = []
    time_step_x = []
    for i in xrange(len(data)):
        x_raw.append(data[i])
        if t == rho:
            time_step_x.append(x_raw)
            x_raw = []
            t = 0
        t += 1
    x = np.array(time_step_x)

    return x


def get_data_4_predict(x_dir, y_dir, is_y=True):
    if is_y:
        x_tmp = []
        y_tmp = []
        f_tmp = []
        for item in os.listdir(x_dir):
            if item.endswith(".htk"):
                # read the mfcc features
                reader = HTKFeat_read(x_dir + item)
                matrix = reader.getall()
                x_tmp.append(matrix)
                labels = np.loadtxt(y_dir + item.replace("_16.htk", '.txt'))
                y_tmp.append(labels)
                f_tmp.append([item, len(labels)])
        x = np.array(x_tmp)
        y = np.array(y_tmp)
        f_names = np.array(f_tmp)
        return x, y, f_names
    else:
        x_tmp = []
        f_tmp = []
        for item in os.listdir(x_dir):
            if item.endswith(".htk"):
                # read the mfcc features
                reader = HTKFeat_read(x_dir + item)
                matrix = reader.getall()
                x_tmp.append(matrix)
                f_tmp.append([item, len(matrix)])
        x = np.array(x_tmp)
        f_names = np.array(f_tmp)
        return x, f_names


def get_data_set(train_path, test_path):
    """
    loads the train and test sets together
    :param train_path: the path to the train data
    :param test_path: the path to the test data
    :return: 4 numpy d-arrays for the train and test features and labels
    """
    x_train, y_train = get_data(train_path)
    x_test, y_test = get_data(test_path)

    s = 0
    count = 0
    for i in range(len(y_train)):
        s += sum(y_train[i][:, 0])
        count += len(y_train[i][:, 0])
    prec_1 = (s/count)
    print "========================"
    print "Labels balance:"
    print "The percentage of label 0: %.2f" % (1 - prec_1)
    print "The percentage of label 1: %.2f" % prec_1
    print "========================"
    return x_train, y_train, x_test, y_test


# helper function
def convert_label2vec(path, output):
    # read the features
    fid = open(path)
    lines = fid.readlines()
    dim = int(lines[0].split()[0])
    y = np.zeros([dim, 2])
    y[:, 1] = 1
    accumulate = 0
    for i in xrange(1, len(lines)):
        values = lines[i].split()
        y[:, 0][int(values[0]) + accumulate:int(values[1]) + accumulate] = 1
        y[:, 1][int(values[0]) + accumulate:int(values[1]) + accumulate] = 0
        accumulate += int(values[2])
    fid.close()

    np.savetxt(output, y)
