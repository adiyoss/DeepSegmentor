import shutil

path_data = '/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/bb_pos/bb_features_list.test.txt'
path_labels = '/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/bb_pos/bb_labels.test.txt'
path_data_dir = '/Users/yossiadi/Datasets/vot/bb/data_bb/'
output_data = '/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/vot/bb_pos/test/'

# parse the labels
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

# parse the file names
file_names = list()
with open(path_data) as f:
    for line in f.readlines():
        v = line.split('/')
        file_names.append(v[len(v) - 1][:-1])

# copy the files to new place
# and remove the header file from the features
for item in file_names:
    shutil.copy(path_data_dir + item, output_data + item)
    abs_path = output_data + item
    with open(abs_path, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(abs_path, 'w') as fout:
        fout.writelines(data[1:])

# create label files
for i, item in enumerate(file_names):
    try:
        abs_path = output_data + item
        f = open(abs_path.replace('.txt', '.labels'), 'w')
        f.write('1 2\n')
        f.write(str(labels[i][0]) + ' ' + str(labels[i][1]))
        f.close()
    except:
        print(item)
