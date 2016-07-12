__author__ = 'Yossi'

import os
from optparse import OptionParser

from lib.textgrid import *

# this script shorten the wav file and its TextGrid
# it's shorten the file according to the vowel onset and offset
# 150 millisecond from the onset and 400 millisecond from the offset
if __name__ == "__main__":
    # parse the parameters
    # the first place is the labels path
    # the second place is the wave files path
    # -------------MENU-------------- #
    parser = OptionParser()
    parser.add_option("-s", "--source", dest="source_path", help="The path to the audio files", metavar="FILE")
    parser.add_option("-d", "--dest", dest="destination_path", help="The output path for the files", metavar="FILE")
    parser.add_option("-t", "--textGrid", dest="TextGrid_path", help="The path to the relevant TextGrid files",
                      metavar="FILE")

    (options, args) = parser.parse_args()

    # validation
    if options.source_path is None or options.destination_path is None or options.TextGrid_path is None:
        print "Invalid number of arguments."

    else:
        source_path = options.source_path
        dest_path = options.destination_path
        text_grid_path = options.TextGrid_path
        gap_start = 0.3
        gap_end = 0.4

        # convert all the files in source_path that ends with .wav to 16 bit 16k and to mono
        # write the new files to dest_path
        for i in os.listdir(source_path):
            if i.endswith(".wav"):
                txtGridFile = i.replace("_16.wav", ".TextGrid")
                textGrid = TextGrid()
                interval = IntervalTier()
                textGrid.read(text_grid_path + txtGridFile)

                start = textGrid.tiers[1].intervals[1].minTime - gap_start
                end = textGrid.tiers[1].intervals[1].maxTime + gap_end
                duration = end - start

                print "Start = " + str(start)
                print "End = " + str(end)

                command = "../sbin/sox " + source_path + i + " " + dest_path + i + " trim " + str(start) + " " + str(
                    duration)
                os.system(command)

                # build TextGrid
                duration -= 0.01
                textgrid = TextGrid()
                vowels_tier = IntervalTier(name='vowel', minTime=0.0, maxTime=float(duration))
                vowels_tier.addInterval(Interval(0, float(gap_start), ""))
                vowels_tier.addInterval(Interval(float(gap_start), float(duration) - float(gap_end), ""))
                vowels_tier.addInterval(Interval(float(duration) - float(gap_end), float(duration), ""))

                textgrid.append(vowels_tier)
                textgrid.write(dest_path + txtGridFile)
