#!/usr/bin/env python

"""Webcam Eye-tracker 
	CJ Ryan, Hacker School Winter 2014

	Implemenatation reverse engineered from 'Eye pupil localization with an ensemble of randomized trees' (Markus, Frljaka, Pandzia, Ahlbergb, Forchheimer)

	guided also by Tom Ballinger's `gazer' (see Github) """

# TODO:
#   why does this hang when subsample_size = 0?    

import glob
import argparse
import sys

from face_finder import * # is 'import *' bad style, since it's unclear later in the code where those functions are scoped?
from tree_ensemble import *
from eye_tracker import *


parser = argparse.ArgumentParser(description='An eye tracker, developed from the method described by Markus et al 2014 (doi: 10.1016/j.patcog.2013.08.008).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tree-depth', type=int, default=3, help='The depth of the tree (=10 in the paper).')
parser.add_argument('--training-data-folder', metavar='TDF',  dest='training_data_folder_processed', default='./training_data_processed', help='Folder containing the training data.')
parser.add_argument('--training-data-folder-raw', metavar='TDFr',  dest='training_data_folder_raw', default='./training_data_raw/BioID/BioID-FaceDatabase-V1.2', help='Folder containing the raw training data (faces), which is to be processed into usable training data (just eyes).')
parser.add_argument('--ntrees', type=int, default=10, help='The number of trees in the random ensemble (=100 in the paper).')
parser.add_argument('--shrinkage', metavar='NU', type=float, default=0.4, help='Shrinkage parameter for the boosted ensemble.')
parser.add_argument('--exportgv', type=bool, default=False, help='(True/False) Export results of each tree in a file that can be used to produce tree graphs using GraphViz.')
parser.add_argument('--subsample', metavar='FRAC', type=float, default=1.0, help='The subsample fraction of the entire dataset used for creating each tree in the ensemble (i.e., no subsampling done if subsample=1.0).')

args = parser.parse_args()
globals().update(vars(args)) # is this bad style?

assert subsample<=1.0 and subsample>=0.0, "subsample parameter %g is not in [0,1] interval" % subsample

if __name__ == '__main__':

    # if the training data folder is empty, make the training data:
    if os.listdir(training_data_folder_processed) == []:
        sys.exit("Error: No training data found in", training_data_folder)

    training_data_filelist = glob.glob(training_data_folder_processed+'/BioID_????_eye?_????.jpg')

    print 'number of trees:\t\t', ntrees
    print 'number of splittings per tree:\t', sum([2**d for d in range(tree_depth-1)])
    tree_ens = TreeEnsemble(ntrees, tree_depth, subsample, exportgv)

    # TODO: make it so that you can either train the eye tracker, or load pre-trained parameters (with some metadata in the header):
    tree_ens.train(training_data_filelist)
 

    # (a) test on some other images, rather than the webcam:



    ff = FaceFinder()
    cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE)

    while True:
        # get image from webcam:
        capture = cv.CaptureFromCAM(0)
        image = cv.QueryFrame(capture)

        # detect the eyes from the image:
        eyes = ff.detect_eyes(image)
        # if eyes:
        #     left_pupil  = self.tree_ens.predict(left_eye[0])
        #     right_pupil = self.tree_ens.predict(right_eye[1])

        # et.draw_pupils(image, left_pupil, right_pupil)

        # display the image, which now contains boxes drawn on the face and eyes:
        # cv.ShowImage('asdf', image)
        # cv.WaitKey(0)






    sys.stdout.write('\n')