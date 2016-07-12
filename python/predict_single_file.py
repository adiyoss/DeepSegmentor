import argparse
from lib.htkmfc import HTKFeat_read
from run_front_end import extract_single_mfcc
import numpy as np
import os

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description="This script extract the mfcc features from a given audio file.")
    parser.add_argument("in_path_x", help="The path to the audio files")
    parser.add_argument("out_path", help="The path to save the mfcc's and labels, the saved file will be a "
                                         "pickle file for both features and labels")
    args = parser.parse_args()


# extract the features
extract_single_mfcc(args.in_path_x, args.in_path_x.replace(".wav", ".htk"))

# convert them to .txt file
reader = HTKFeat_read(args.in_path_x.replace(".wav", ".htk"))
matrix = reader.getall()

# write them to the desired output
f_handle = file(args.out_path, 'a')
np.savetxt(f_handle, matrix)
f_handle.close()

# remove leftovers
os.remove(args.in_path_x.replace(".wav", ".htk"))
