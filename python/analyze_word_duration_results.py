import argparse
import os
import numpy as np
from lib.textgrid import TextGrid


def main(man_path, wdm_path, seg_path):
    dur_wdm = list()
    dur_seg = list()
    dur_man = list()
    for item in os.listdir(man_path):
        if item.endswith('.TextGrid') and os.path.exists(seg_path+item) and os.path.exists(wdm_path+item):
            wdm_text = TextGrid()
            wdm_text.read(wdm_path+item)
            if len(wdm_text.tiers[0].intervals) < 2:
                dur_wdm.append([0, 0])
            else:
                onset = wdm_text.tiers[0].intervals[1].minTime
                offset = wdm_text.tiers[0].intervals[1].maxTime
                dur_wdm.append([onset, offset])

            seg_text = TextGrid()
            seg_text.read(seg_path + item)
            onset = seg_text.tiers[0].intervals[1].minTime
            offset = seg_text.tiers[0].intervals[1].maxTime
            dur_seg.append([onset, offset])

            man_text = TextGrid()
            man_text.read(man_path + item)
            onset = man_text.tiers[1].intervals[1].minTime
            offset = man_text.tiers[1].intervals[1].maxTime
            dur_man.append([onset, offset])

    loss_durations_wdm = 0
    loss_durations_seg = 0
    loss_offsets_wdm = np.zeros(3)
    loss_offsets_seg = np.zeros(3)
    e = 0

    for i in range(len(dur_wdm)):
        y = dur_man[i][1] - dur_man[i][0]
        y_hat_wdm = dur_wdm[i][1] - dur_wdm[i][0]
        y_hat_seg = dur_seg[i][1] - dur_seg[i][0]

        loss_durations_wdm += max(np.abs(y - y_hat_wdm) - e, 0)
        loss_durations_seg += max(np.abs(y - y_hat_seg) - e, 0)

        loss_offsets_wdm[0] += np.abs(dur_wdm[i][0] - dur_man[i][0])
        loss_offsets_wdm[1] += np.abs(dur_wdm[i][1] - dur_man[i][1])
        loss_offsets_wdm[2] += np.abs(dur_wdm[i][0] - dur_man[i][0]) + np.abs(dur_wdm[i][1] - dur_man[i][1])

        loss_offsets_seg[0] += np.abs(dur_seg[i][0] - dur_man[i][0])
        loss_offsets_seg[1] += np.abs(dur_seg[i][1] - dur_man[i][1])
        loss_offsets_seg[2] += np.abs(dur_seg[i][0] - dur_man[i][0]) + np.abs(dur_seg[i][1] - dur_man[i][1])

    loss_durations_wdm /= len(dur_wdm)
    loss_durations_seg /= len(dur_wdm)
    loss_offsets_wdm /= len(dur_wdm)
    loss_offsets_seg /= len(dur_wdm)
    print 'duration loss wdm: %s' % (loss_durations_wdm * 100)
    print 'duration loss seg: %s' % (loss_durations_seg * 100)
    print '====================='
    print 'loss offsets - onset wdm: %s, offset wdm: %s, total: %s' % (loss_offsets_wdm[0] * 100, loss_offsets_wdm[1] * 100, loss_offsets_wdm[2] * 100)
    print 'loss offsets - onset seg: %s, offset seg: %s, total: %s' % (loss_offsets_seg[0] * 100, loss_offsets_seg[1] * 100, loss_offsets_seg[2] * 100)

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description="Analyze results.")
    parser.add_argument("--manually_label", help="The path to the manually labeled TextGrid",
                        default='/Users/yossiadi/Projects/deep_audio_segmentation'
                                '/code/segmentor/preductions/wd/manually/')
    parser.add_argument("--wdm_pred", help="The path to the TextGrid prediction of the wdm algorithm",
                        default='/Users/yossiadi/Projects/deep_audio_segmentation/'
                                'code/segmentor/preductions/wd/deep_wdm_2rnn/')
    parser.add_argument("--segmentor_pred", help="The path to the TextGrid prediction of the segmentor algorithm",
                        default='/Users/yossiadi/Projects/deep_audio_segmentation'
                                '/code/segmentor/preductions/wd/deep_segmentor/')
    args = parser.parse_args()
    main(args.manually_label, args.wdm_pred, args.segmentor_pred)
