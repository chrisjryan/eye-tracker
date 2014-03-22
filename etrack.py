#!/usr/bin/env python

"""Webcam Eye-tracker 
	CJ Ryan, Hacker School Winter 2014

	Implemenatation reverse engineered from 'Eye pupil localization with an ensemble of randomized trees' (Markus, Frljaka, Pandzia, Ahlbergb, Forchheimer)

	guided also by Tom Ballinger's `gazer' (see Github) """

import glob
import argparse

from face_finder import * # is this bad style, since it's unclear later in the code where those functions are scoped?
from training_data_gen import *
from tree_ensemble import *


parser = argparse.ArgumentParser(description='An eye tracker, developed from the method described by Markus et al 2014 (doi: 10.1016/j.patcog.2013.08.008).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tree-depth', type=int, default=3, help='The depth of the tree (=10 in the paper).')
parser.add_argument('--training-data-folder', metavar='TDF',  dest='training_data_folder_processed', default='./training_data_processed', help='Folder containing the training data.')
parser.add_argument('--training-data-folder-raw', metavar='TDFr',  dest='training_data_folder_raw', default='./training_data_raw/BioID/BioID-FaceDatabase-V1.2', help='Folder containing the raw training data (faces), which is to be processed into usable training data (just eyes).')
parser.add_argument('--ntrees', type=int, default=10, help='The number of trees in the random ensemble (=100 in the paper).')
parser.add_argument('--shrinkage', metavar='NU', type=float, default=0.4, help='Shrinkage parameter for the boosted ensemble.')

args = parser.parse_args()
globals().update(vars(args)) # is this bad style?




if __name__ == '__main__':

    # if the training data folder is empty, make the training data:
    if os.listdir(training_data_folder_processed) == []:
        # this doesn't need to be a class/object
        tdg = training_data_gen()
        tdg.make_training_data(training_data_folder_raw, training_data_folder_processed)

    training_data_filelist = glob.glob(training_data_folder_processed+'/BioID_????_eye?_????.jpg')

    print 'number of trees:\t\t', ntrees
    print 'number of splittings per tree:\t', sum([2**d for d in range(tree_depth-1)])
    tree_ens = TreeEnsemble(ntrees, tree_depth, training_data_filelist)
    tree_ens.train()

    # print the trained results so you know they look okay:



    sys.stdout.write('\n')