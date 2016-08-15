import argparse
import os
import sys

import numpy as np


def read_labels_files(path):
    suffix = '.labels'
    if not os.path.isdir(path):
        print >> sys.stderr, "Directory does not exists"
        return 0
    labels = list()
    for i in os.listdir(path):
        if i.endswith(suffix):
            with open(path+i) as f:
                l = f.readlines()[-1:][0].split()
                l.append(i)
                labels.append(l)
    return labels


def calc_durations(files):
    durations = list()
    for i in files:
        d = float(i[1]) - float(i[0])
        durations.append(d)
    return durations


def print_outliers(files, mean, std, m=3):
    c = 0
    s = ''
    for i in files:
        d = float(i[1]) - float(i[0])
        if abs(d - mean) > m*std:
            print('File name: %s, duration %d ' % (i[2], d))
            c += 1
    s = 'Number of files that do not fall into %d std is %d' % (m, c)
    print(s)
    return s


def run_stats(path, output_path):
    dirs = ['train/', 'test/', 'val/']
    names = ['train', 'test', 'validation']
    str = 'WORD DURATION DATA STATISTICS:'

    for i, d in enumerate(dirs):
        files = read_labels_files(os.path.join(path + d))
        durations = calc_durations(files)

        mean = np.mean(durations)
        median = np.median(durations)
        std = np.std(durations)
        max = np.max(durations)
        min = np.min(durations)
        count = len(durations)

        s = '\n\nDuration statistics for %s files: ' % names[i]
        str += s + '\n'
        print(s)

        s = 'Mean: %.3f' % mean
        str += s + '\n'
        print(s)

        s = 'Median: %.3f' % median
        str += s + '\n'
        print(s)

        s = 'Std: %.3f' % std
        str += s + '\n'
        print(s)

        s = 'Max: %.3f' % max
        str += s + '\n'
        print(s)

        s = 'Min: %.3f' % min
        str += s + '\n'
        print(s)

        s = 'Number of files: %d' % count
        str += s + '\n'
        print(s)

        s = print_outliers(files, mean, std, m=4)
        str += s + '\n'

    f = open(output_path, 'w')
    f.write(str)
    f.close()


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description="Calculates data statistics.")
    parser.add_argument("--in_path", help="The path to the data files, this dir should contains the val"
                                          ", test and train dirs with the labels files inside them",
                        default='/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/amanda_pos/')
    parser.add_argument("--out_path", help="The path to the save the data stats",
                        default='/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/docs/'
                                'vot_duration_amanda_stats.txt')
    args = parser.parse_args()

    run_stats(args.in_path, args.out_path)
