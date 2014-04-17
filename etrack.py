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
import matplotlib.pylab as plt

from face_finder import * # is 'import *' bad style, since it's unclear later in the code where those functions are scoped?
from tree_ensemble import *
#from eye_tracker import *


parser = argparse.ArgumentParser(description='An eye tracker, developed from the method described by Markus et al 2014 (doi: 10.1016/j.patcog.2013.08.008).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tree-depth', type=int, default=3, help='The depth of the tree (=10 in the paper).')
parser.add_argument('--training-data-folder', metavar='TDF',  dest='training_data_folder_processed', default='./training_data_processed', help='Folder containing the training data.')
parser.add_argument('--training-data-folder-raw', metavar='TDFr',  dest='training_data_folder_raw', default='./training_data_raw/BioID/BioID-FaceDatabase-V1.2', help='Folder containing the raw training data (faces), which is to be processed into usable training data (just eyes).')
parser.add_argument('--ntrees', type=int, default=10, help='The number of trees in the random ensemble (=100 in the paper).')
parser.add_argument('--shrinkage', metavar='NU', type=float, default=0.4, help='Shrinkage parameter for the boosted ensemble.')
parser.add_argument('--exportgv', type=bool, default=False, help='(True/False) Export results of each tree in a file that can be used to produce tree graphs using GraphViz.')
parser.add_argument('--subsample', metavar='FRAC', type=float, default=1.0, help='The subsample fraction of the entire dataset used for creating each tree in the ensemble (i.e., no subsampling done if subsample=1.0).')
parser.add_argument('--min_neighbors', metavar='neigh', type=int, default=3, help='This parameter for the Haar classifier generally sets how strictly to look for eyes (still kind of mysterious). Setting it higher makes this classifier more discriminating, and might avoid detection of non-eye things things like nostrils.')
parser.add_argument('--min_eyeface_ratio', metavar='RAT', type=float, default=1.0/6.0, help="The minimum size ratio between the eyes and face. Used so that things much smaller than eyes (e.g., nostrils) aren't accidentally classified as eyes.")
parser.add_argument('--max_eyeface_ratio', metavar='RAT', type=float, default=5.0/12.0, help="The maximum size ratio between the eyes and face. Used so that things much larger than eyes aren't accidentally classified as eyes.")
parser.add_argument('--cross-validation-test', metavar='cvTF', type=bool, default=False, help="(True/False) Use the learned tree to predict image files, rather than the webcam, for cross validation tests.")
parser.add_argument('--treeparams-folder', metavar='TPF', default=None, help='Folder containing a set of XML files that define the (trained) tree ensemble. Note: this makes user-specified parameters like ntrees & tree_depth irrelevant.')
parser.add_argument('--saveXML', type=bool, default=False, help="(True/False) If you train random tree ensemble, save the result as XML files.")

args = parser.parse_args()
#globals().update(vars(args)) # is this bad style?

assert args.subsample<=1.0 and args.subsample>=0.0, "subsample parameter %g is not in [0,1] interval" % subsample


# TODO: read these parameters from the command line rather than hardcoding them.
def crossval_test(training_files, tree_ens, tree_depth_min = 3, tree_depth_max = 10, ntrees_min = 1, ntrees_max = 20):
    print """Cross validation test.
             This will generate plots of the mean squared prediction error as a 
             function of tree_depth (from range %i to %i) and ntrees (from 
             range %i to %i). User specifications of these parameters at 
             the command line will be ignored.""" % (tree_depth_min , tree_depth_max, ntrees_min, ntrees_max)

    results = [[tree_ens.crossval(training_files, ff, depth, ntrees) \
                for depth in range(tree_depth_min, tree_depth_max+1)] \
                for ntrees in range(ntrees_min, ntrees_max+1)]

    # output results to a file:
    cv_results = open('cv_results.txt','w')
    nimages = len(training_data_filelist)
    subsample_size = int(round(tree_ens.subsample_frac*nimages))
    cv_results.write('# using sets of %i of %i training files per tree\n' % (subsample_size, nimages))
    cv_results.write('# rows: tree_depth (from range %i to %i), cols: ntrees (from range %i to %i)\n' % (tree_depth_min , tree_depth_max, ntrees_min, ntrees_max))
    numpy.savetxt(cv_results, results)
    cv_results.close()

    # TODO: automate plotting.


def detect_webcam(tree_ens, ff):
    while True:
        # get image from webcam & get the subimages containing eyes:
        capture = cv.CaptureFromCAM(0)
        image = cv.QueryFrame(capture)
        eyes = ff.detect_eyes(image)

        # use the trained random forest to predict the pupil coordinates:
        if eyes and len(eyes[0]) == 2:
            eyes_imgs, eyes_loc = eyes
            left_pupil  = tree_ens.predict_forest(eyes_imgs[0])
            right_pupil = tree_ens.predict_forest(eyes_imgs[1])

            ff.draw_pupil(image, left_pupil, eyes_loc[0])
            ff.draw_pupil(image, right_pupil, eyes_loc[1])

            # display the image, which now contains boxes drawn on the face and eyes:
            cv.ShowImage('asdf', image)
            cv.WaitKey(0)



if __name__ == '__main__':

    # TODO: in the following functions, pass 'args' then unpack it inside.
    ff = FaceFinder(args.min_neighbors, args.min_eyeface_ratio, args.max_eyeface_ratio)
    cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE)
    tree_ens = TreeEnsemble(args.subsample, args.exportgv, args.saveXML)

    if args.cross_validation_test:
        training_data_filelist = glob.glob(args.training_data_folder_processed+'/BioID_????_eye?_????.jpg')
        crossval_test(training_data_filelist, tree_ens)

    else:
        # TODO: abstract some of this a bit better:
        if args.treeparams_folder:
            print 'loading params from '+ args.treeparams_folder+'...'
            tree_param_filelist = glob.glob(args.treeparams_folder+'/*.xml')
            tree_ens.loadparams(tree_param_filelist)

        elif os.listdir(args.training_data_folder_processed) != []:
            print 'building trees using training data in '+args.training_data_folder_processed+'...'
            print 'number of trees:\t\t', args.ntrees
            print 'number of splittings per tree:\t', sum([2**d for d in range(args.tree_depth-1)])
            training_data_filelist = glob.glob(args.training_data_folder_processed+'/BioID_????_eye?_????.jpg')
            tree_ens.train(training_data_filelist, args.tree_depth, args.ntrees)
        else:
            sys.exit("Error: No processed training data found in", args.training_data_folder_processed)


        detect_webcam(tree_ens, ff)


    sys.stdout.write('\n')