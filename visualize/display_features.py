import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'adiyoss'

graphIndex = 0
graphTitle = 0
predict_line = 1
label_line = 1


def ontype(event):
    global graphIndex
    global graphTitle
    global predict_line
    global label_line

    if event.key == 'q':
        sys.exit(0)

    elif event.key == 'right':
        graphIndex -= 1
        graphTitle -= 1
    elif event.key == 'left':
        graphIndex += 1
        graphTitle += 1
    elif event.key == 'p':
        if predict_line == 1:
            predict_line = 0
        else:
            predict_line = 1
    elif event.key == 'l':
        if label_line == 1:
            label_line = 0
        else:
            label_line = 1
    else:
        return

    plt.close()


def display_features(filename, frame_begin_and_end_real, frame_begin_and_end_predict):
    global graphIndex, predict_plot, labels_plot

    if os.path.isfile(filename) is False:
        sys.stderr.write("WARNING: file not found, " + str(filename))

    labels = frame_begin_and_end_real.split('-')
    predict = frame_begin_and_end_predict.split('-')
    m = np.loadtxt(filename)
    feature_names = ['Short Term Energy', 'Total Energy', 'Low Energy', 'High Energy', 'Wiener Entropy',
                     'Auto Correlation', 'Pitch', 'Voicing', 'Zero Crossing',
                     '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                     '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                     '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                     '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                     '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                     '51', '52', '53', '54']
    while True:
        index = graphIndex % len(m[0])
        title_index = graphTitle % len(feature_names)

        fig = plt.figure(1, figsize=(20, 10))
        fig.canvas.mpl_connect('key_press_event', ontype)
        fig.suptitle(feature_names[title_index], fontsize='x-large', style='italic', fontweight='bold')
        max_m = np.max(m[:, index])
        min_m = np.min(m[:, index])
        width = float(0.6)
        plt.plot((m[:, index]), linestyle='-', linewidth=width, color='#006699')
        if label_line == 1:
            labels_plot, = plt.plot([labels[0], labels[0]], [min_m, max_m], linestyle='-.', color="#730A0A", lw=2)
            plt.plot([labels[1], labels[1]], [min_m, max_m], linestyle='-.', color="#730A0A", lw=2)
        if predict_line == 1:
            predict_plot, = plt.plot([predict[0], predict[0]], [min_m, max_m], linestyle='-.', color='#335C09', lw=2)
            plt.plot([predict[1], predict[1]], [min_m, max_m], linestyle='-.', color='#335C09', lw=2)
        plt.xlim(xmin=0, xmax=len(m))

        # plot the legend
        plt.figtext(0.13, 0.05, 'Q: quit', style='italic')
        plt.figtext(0.2, 0.05, 'P: Enable/disable prediction marks', style='italic')
        plt.figtext(0.38, 0.05, "L: Enable/disable label marks", style='italic')
        plt.figtext(0.13, 0.02, 'Left arrow: Next figure', style='italic')
        plt.figtext(0.38, 0.02, 'Right arrow: Previous figure', style='italic')
        l2 = plt.legend([labels_plot, predict_plot], ["Real Label", "Predict Label"])
        plt.gca().add_artist(l2)  # add l1 as a separate artist to the axes
        plt.show()


# ------------- MENU -------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Features Visualization")
    parser.add_argument("features_file", help="The path to features file")
    parser.add_argument("labels", help="The vot onset and offset, separated by dash, e.g. 100-108")
    parser.add_argument("prediction", help="The vot onset and offset prediction, separated by dash, e.g. 100-108")
    args = parser.parse_args()

    # run the script
    display_features(args.features_file, args.labels, args.prediction)
