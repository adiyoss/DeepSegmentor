import os


def create_label_files(
        path_data='/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/amanda_pos/4/',
        path_labels='/Users/yossiadi/Datasets/vot/amanda/labels/amanda_fe_labels_4.txt'):
    labels = list()
    is_first = True
    with open(path_labels) as f:
        for line in f.readlines():
            if is_first:
                is_first = False
            else:
                v = line.split()
                t = list()
                t.append(v[0])
                t.append(v[1])
                labels.append(t)

    i = 0
    for item in os.listdir(path_data):
        try:
            if item.endswith('.txt'):
                abs_path = path_data + item
                f = open(abs_path.replace('.txt', '.labels'), 'w')
                f.write('1 2\n')
                f.write(str(labels[i][0]) + ' ' + str(labels[i][1]))
                f.close()
                i += 1
        except:
            print(item)

    # remove the header file from the features
    for item in os.listdir(path_data):
        if item.endswith('.txt'):
            abs_path = path_data + item
            with open(abs_path, 'r') as fin:
                data = fin.read().splitlines(True)
            with open(abs_path, 'w') as fout:
                fout.writelines(data[1:])

path_data = '/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/amanda_pos/1/'
path_labels = '/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/amanda_pos/1/'
path_outliers = '/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/amanda_pos/outliers/'

for item in os.listdir(path_data):
    if item.endswith('.txt'):
        count = 0
        with open(path_data+item) as f:
            for line in f.readlines():
                count += 1

        f = open(path_labels + item.replace('.txt', '.labels'))
        v = f.readlines()[-1:][0].split()
        if count < int(v[1]):
            os.rename(path_data+item, path_outliers+item)
            os.rename(path_labels + item.replace('.txt', '.labels'), path_outliers + item.replace('.txt', '.labels'))
            print(item)
        f.close()
