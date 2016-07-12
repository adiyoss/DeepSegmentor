__author__ = 'adiyoss'

import os
from optparse import OptionParser
from lib.textgrid import *


def create_text_grid(text_grid_path, label, num_of_frames):
    """
    create TextGrid files from the labels_path file
    """

    # should be different for every file format
    FRAME_RATE = 10
    MILLISEC_2_SEC = 0.001
    FRAME_CONCAT_ONSET = 0
    FRAME_CONCAT_OFFSET = 0
    v_onset = (label[0] + FRAME_CONCAT_ONSET)
    v_offset = (label[1] + FRAME_CONCAT_OFFSET)
    length = num_of_frames*FRAME_RATE*MILLISEC_2_SEC

    # build TextGrid
    text_grid = TextGrid()
    vowels_tier = IntervalTier(name='Duration', minTime=0.0, maxTime=float(length))
    vowels_tier.addInterval(Interval(0, float(v_onset) * FRAME_RATE * MILLISEC_2_SEC, ""))
    vowels_tier.addInterval(Interval(float(v_onset) * FRAME_RATE * MILLISEC_2_SEC, float(v_offset) * FRAME_RATE * MILLISEC_2_SEC, ""))
    vowels_tier.addInterval(Interval(float(v_offset) * FRAME_RATE * MILLISEC_2_SEC, float(length), ""))

    text_grid.append(vowels_tier)
    text_grid.write(text_grid_path)