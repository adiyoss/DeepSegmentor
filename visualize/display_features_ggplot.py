import sys
from optparse import OptionParser
import numpy as np
from ggplot import *
import pandas as pd

__author__ = 'adiyoss'

NUM_OF_FEATURES = 22


def display_features(filename, frame_begin_and_end_real, frame_begin_and_end_predict):
    labels = frame_begin_and_end_real.split('-')
    m = np.loadtxt(filename)
    index = 0
    l = np.arange(0, len(m[:, index]))
    df = pd.DataFrame(
        {'Frames': l, 'Short-Term Energy': m[:, index], 'Total Energy': m[:, index + 1], 'Low Energy': m[:, index + 2], 'High Energy': m[:, index + 3]})    
    p = ggplot(pd.melt(df, id_vars=['Frames']),
               aes(x='Frames', y='value', color='variable')) + geom_line() + geom_vline(
        xintercept=[labels[0], labels[1]], color="black") + theme_bw()
    ggsave(p, "features_dash_acus_1.pdf", width = 13, height = 6)

    df = pd.DataFrame(
        {'Frames': l, 'Wiener Entropy': m[:, index + 4], 'Auto Correlation': m[:, index + 5], 'Zero Crossing': m[:, index + 8]})
    p = ggplot(pd.melt(df, id_vars=['Frames']),
               aes(x='Frames', y='value', color='variable')) + geom_line() + geom_vline(
        xintercept=[labels[0], labels[1]], color="black") + theme_bw()
    ggsave(p, "features_dash_acus_2.pdf", width = 13, height = 6)

    # df = pd.DataFrame(
    #     {'Frames': l, 'Vowel Indicator': m[:, index + 9], 'Nasal Indicator': m[:, index + 10], 'Sum vowel': m[:, index + 13]})
    # p = ggplot(pd.melt(df, id_vars=['Frames']),
    #            aes(x='Frames', y='value', color='variable')) + geom_line() + geom_vline(
    #     xintercept=[labels[0], labels[1]], color="black") + theme_bw()
    # ggsave(p, "features_dash_phoneme.eps", width = 13, height = 6)

    # df = pd.DataFrame(
    #     {'Frames': l, 'Delta 1': m[:, index + 16], 'Delta 2': m[:, index + 17], 'Delta 3': m[:, index + 18], 'Delta 4': m[:, index + 19]})
    # p = ggplot(pd.melt(df, id_vars=['Frames']),
    #            aes(x='Frames', y='value', color='variable')) + geom_line() + geom_vline(
    #     xintercept=[labels[0], labels[1]], color="black") + theme_bw()
    # ggsave(p, "features_dash_delta.eps", width = 13, height = 6)
    
# parse the parameters
# the first argument should be the labels file from the intellij
# the second argument should be the path to the directory in which the textGrid files are located
# #-------------MENU--------------#
parser = OptionParser()
parser.add_option("-f", "--file", dest="file", help="The name of the data file", metavar="FILE")
parser.add_option("-l", "--label", dest="label", help="The onset and offset of the vowel, Example: 100-138",
                  metavar="FILE")
parser.add_option("-p", "--predict", dest="predict", help="The predicted onset and offset of the vowel, same as before",
                  metavar="FILE")
(options, args) = parser.parse_args()

# validation
if options.file is None or options.label is None or options.predict is None:
    sys.stderr.write("Invalid number of arguments.")
else:
    # run the script
    display_features(options.file, options.label, options.predict)
