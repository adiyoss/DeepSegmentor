import argparse
import sys
import os
import numpy as np

from lib import utility as utils
from lib import textgrid as tg
from lib.utility import generate_tmp_filename

__author__ = 'yossiadi'

# globals
NO_STRING = '$$'


def extract_single_mfcc(in_path, out_path):
    """
    Extract mfcc features from one audio file
    :param in_path: the path to the audio file
    :param out_path: the path to save the mfcc features
    """
    import platform
    plat = platform.system().lower()
    if plat is 'darwin':
        sox_path = 'sbin/osx/sox'
        htk_path = 'sbin/osx'
    elif 'linux' in plat:
        sox_path = 'sox'
        htk_path = 'sbin/linux'
    else:
        sox_path = 'sbin/osx/sox'
        htk_path = 'sbin/osx'

    tmp_file = generate_tmp_filename('wav')
    cmd = "%s %s -r 16000 -b 16 %s" % (sox_path, in_path, tmp_file)
    utils.easy_call(cmd)
    cmd = "%s/HCopy -C config/htk.config %s %s" % (htk_path, tmp_file, out_path)
    utils.easy_call(cmd)
    os.remove(tmp_file)


def extract_single_acoustic(in_path, out_path, times):
    """
    Extract mfcc features from one audio file
    :param in_path: the path to the audio file
    :param out_path: the path to save the mfcc features
    """
    import platform
    plat = platform.system().lower()
    if plat == 'darwin':
        sox_path = 'sbin/osx/sox'
    elif 'linux' in plat:
        sox_path = 'sox'
    else:
        sox_path = 'sbin/osx/sox'

    tmp_file = generate_tmp_filename('wav')
    cmd = "%s %s -r 16000 -b 16 %s" % (sox_path, in_path, tmp_file)
    utils.easy_call(cmd)

    input_file = generate_tmp_filename('input')  # open the input file for the feature extraction
    features_file = generate_tmp_filename('features')  # open file for the feature list path
    labels_file = generate_tmp_filename('labels')  # open file for the labels

    with open(input_file, 'w') as f:
        # write the data
        f.write(
            '"' + tmp_file + '" ' + str(0) + ' ' + str(times[1]) + ' ' + str(0) + ' ' + str(
                times[1]))

    with open(features_file, 'w') as f:
        f.write(out_path)

    command = "sbin/osx/fea_extract %s %s %s" % (input_file, features_file, labels_file)
    utils.easy_call(command)

    with open(out_path, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(out_path, 'w') as fout:
        fout.writelines(data[1:])

    # remove leftovers
    os.remove(input_file)
    os.remove(features_file)
    os.remove(labels_file)
    os.remove(tmp_file)


def extract_acoustic_dir(in_path, out_path, extract_time):
    """
    Extract acoustic features from directory
    :param in_path: the path to the audio files directory
    :param out_path: the path to save the features (should be directory)
    """
    if not os.path.exists(in_path):
        print >> sys.stderr, "Directory does not exists"

    if not os.path.exists(out_path):
        print "Output directory does not exists"
        print "Creating output directory"
        os.mkdir(out_path)

    for item in os.listdir(in_path):
        if item.endswith(".wav"):
            abs_path = os.path.abspath(in_path + item)
            times = extract_time[item]
            extract_single_acoustic(abs_path, out_path + item.replace(".wav", ".data"), times)


def create_labels(in_path, out_path):
    """
    Extract the labels from the text grid files
    :param in_path: the path to the text grid files directory
    :param out_path: the path to save the labels
    """
    extract_time = dict()
    if not os.path.exists(in_path):
        print >> sys.stderr, "Directory does not exists"

    if not os.path.exists(out_path):
        print "Output directory does not exists"
        print "Creating output directory"
        os.mkdir(out_path)

    for item in os.listdir(in_path):
        if item.endswith(".TextGrid"):
            abs_path = os.path.abspath(in_path + item)
            textgrid = tg.TextGrid()
            textgrid.read(abs_path)
            onset_voicing = np.ceil(textgrid.tiers[4].intervals[1].minTime * 1000)
            onset_burst = np.ceil(textgrid.tiers[2].intervals[1].minTime * 1000)
            offset = np.ceil(textgrid.tiers[2].intervals[1].maxTime * 1000)

            f = open(out_path + item.replace(".TextGrid", ".labels"), 'w')
            f.write('1 2\n')
            f.write(str(onset_voicing) + ' ' + str(onset_burst) + ' ' + str(offset))
            f.close()

            gap = 0.08
            extract_time[item.replace('.TextGrid', '.wav')] = [
                max(float(onset_voicing / 1000) - gap, 0),
                float(offset / 1000) + gap]
    return extract_time


# the main function
def main(in_path_x, in_path_y, out_path):
    """
    Extract the mfcc features and labels from the audio and text grid files
    Both the TextGrid and audio files should be at the same directory
    :param in_path_x: the path to the audio files directory
    :param in_path_y: the path to the text grid files directory
    :param out_path: the path to save the files
    """
    if not os.path.exists(in_path_x):
        print >> sys.stderr, "X Directory does not exists"
        return

    extract_time = create_labels(in_path_y, out_path)  # create the labels
    extract_acoustic_dir(in_path_x, out_path, extract_time)  # extract the features


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description="This script extract the mfcc features from a given audio file.")
    parser.add_argument("in_path_x", help="The path to the audio files")
    parser.add_argument("in_path_y", help="The path to the text grid files if possible")
    parser.add_argument("out_path", help="The path to save the features and labels")
    args = parser.parse_args()

    main(args.in_path_x, args.in_path_y, args.out_path)
