import argparse
import os
import numpy as np
from lib.textgrid import TextGrid


def main(man_path, wdm_path):
    ms_2 = 0
    ms_5 = 0
    ms_10 = 0
    ms_15 = 0
    ms_25 = 0
    ms_50 = 0
    total = 0
    with open(wdm_path) as f:
        for l in f.readlines():
            vals = l.split()
            duration_hat = float(vals[2]) - float(vals[1])
            duration = 0
            is_first = True
            with open(man_path+vals[0].replace('.data', '.labels')) as f_man:
                for l_man in f_man.readlines():
                    if is_first:
                        is_first = False
                    else:
                        v_man = l_man.split()
                        duration = float(v_man[1]) - float(v_man[0])
            abs_off = abs(duration_hat - duration)

            total += 1
            if abs_off <= 2:
                ms_2 += 1
            if abs_off <= 5:
                ms_5 += 1
            if abs_off <= 10:
                ms_10 += 1
            if abs_off <= 15:
                ms_15 += 1
            if abs_off <= 25:
                ms_25 += 1
            if abs_off <= 50:
                ms_50 += 1

    print('Percentage of <= 2ms: %f' % (ms_2 / float(total)))
    print('Percentage of <= 5ms: %f' % (ms_5 / float(total)))
    print('Percentage of <= 10ms: %f' % (ms_10 / float(total)))
    print('Percentage of <= 15ms: %f' % (ms_15 / float(total)))
    print('Percentage of <= 25ms: %f' % (ms_25 / float(total)))
    print('Percentage of <= 50ms: %f' % (ms_50 / float(total)))

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description="Analyze results.")
    parser.add_argument("--manually_label", help="The path to the manually labeled TextGrid",
                        default='/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/natalia_pos'
                                '/test/')
    parser.add_argument("--deep_vot", help="The path to the TextGrid prediction of the wdm algorithm",
                        default='/Users/yossiadi/Projects/deep_vot/back_end_new_structure/results/natalia_pos/pred.txt')
    args = parser.parse_args()
    main(args.manually_label, args.deep_vot)
