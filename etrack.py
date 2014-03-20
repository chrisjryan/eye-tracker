#!/usr/bin/env python

"""Webcam Eye-tracker 
	CJ Ryan, Hacker School Winter 2014

	Implemenatation reverse engineered from 'Eye pupil localization with an ensemble of randomized trees' (Markusa, Frljaka, Pandzia, Ahlbergb, Forchheimer)

	guided also by Tom Ballinger's `gazer' (see Github) """

import warnings
warnings.simplefilter('error')

import os
import cv, cv2
import numpy
import random
import glob
import argparse
import sys


from face_finder import * # is this bad style, since it's unclear later in the code where those functions are scoped?
from training_data_gen import *

parser = argparse.ArgumentParser(description='An eye tracker, developed from the method described by Markus et al 2014 (doi: 10.1016/j.patcog.2013.08.008).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tree-depth', type=int, default=3, help='The depth of the tree (=10 in the paper).')
parser.add_argument('--training-data-folder', metavar='TDF',  dest='training_data_folder_processed', default='./training_data_processed', help='Folder containing the training data.')
parser.add_argument('--training-data-folder-raw', metavar='TDFr',  dest='training_data_folder_raw', default='./training_data_raw/BioID/BioID-FaceDatabase-V1.2', help='Folder containing the raw training data (faces), which is to be processed into usable training data (just eyes).')
parser.add_argument('--ntrees', type=int, default=100, help='The number of trees in the random ensemble.')
parser.add_argument('--shrinkage', metavar='NU', type=float, default=0.4, help='Shrinkage parameter for the boosted ensemble.')

args = parser.parse_args()
globals().update(vars(args)) # is this bad style?


class EyeTracker(object):
    def __init__(self, tree_depth):
        self.storage = cv.CreateMemStorage(0)
        self.last_face_position = None
        self.face_cascade = cv.Load(os.path.expanduser('/usr/local/Cellar/opencv/2.4.7.1//share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'))
        self.eyepair_cascade = cv.Load(os.path.expanduser('./parojosG45x11.xml'))
        self.eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.7.1/share/OpenCV/haarcascades/haarcascade_eye.xml')

        # Tom:
        self.detect_times = []
        self.eye_pair_history = []
        self.xamount_histories = [[], []]
        self.yamount_histories = [[], []]
        self.xpos_history = []
        self.ypos_history = []

        self.tree_depth = tree_depth


#    def get_pixel_samples(self, image, nsamples):
#        w, h = cv.GetSize(image)
            # it's uninuitive how height should be indexed first when referring to pixel values later...
#        return [ (randint(0, h - 1), randint(0, w - 1)) ] for _ in range(nsamples) ]
        # check for a numpy funtion that does this too

#        rand_pixel_indices = [(randint(0, h - 1), randint(0, w - 1)) for _ in range(nsamples)]



    # this is kind of a trivial use of a generator:
    def rand_feature_gen(self):
        while True:
            yield ( (random.uniform(-1,1), random.uniform(-1,1)), (random.uniform(-1,1), random.uniform(-1,1)) )

    def get_best_clustering(self, splitting_candidates):

        # TODO: eliminate (or minimize) redundant usages of genfromtxt() here in this subroutine.
        Qlist = []
        for (cluster0, cluster1, feature) in splitting_candidates:
            # get the eye coordinate file list from the image file list:
            # TODO: see if you can make this all a little cleaner
            cluster0_pupilfiles = [os.path.splitext(f)[0]+'.eye' for f in cluster0]
            cluster1_pupilfiles = [os.path.splitext(f)[0]+'.eye' for f in cluster1]
            pupcoodlist0 = [numpy.genfromtxt(pupilcoord) for pupilcoord in cluster0_pupilfiles]
            pupcoodlist1 = [numpy.genfromtxt(pupilcoord) for pupilcoord in cluster1_pupilfiles]

            # is this different from the standard deviation? can you just use a standard subroutine?
            avgpupilcoord0 = numpy.mean(pupcoodlist0, axis=0)
            avgpupilcoord1 = numpy.mean(pupcoodlist1, axis=0)
            # TODO: probably don't need the lambda function here
            Q  =  sum(map(lambda x0, x0avg=avgpupilcoord0: sum((x0-x0avg)**2), pupcoodlist0))
            Q +=  sum(map(lambda x1, x1avg=avgpupilcoord1: sum((x1-x1avg)**2), pupcoodlist1))
            Qlist.append(Q)

        # I think this is working, because Qs get smaller as you get deeper into the tree:
#        print Qlist

        return numpy.argmin(Qlist)



    def cluster_images(self, image_list, d=0): # d is the depth of this node

        if d < self.tree_depth:

            # generate random features on [(-1,+1), (-1,+1)]:
            # you might make this a list of (doubled) named tuples, to make the "pixelcoord_ =" lines more readble
            ncandidate_features = 4*d + 4
            minclustersize = 2.0**(self.tree_depth - d) # this might still slow things down, if min cluster size is achieved near the root of the tree and the min cluster sizes need to be found at each level on the way down

            splitting_candidates = []
            done_clustering = False

            while done_clustering == False:
                # kind of trivial use of a generator:
                feature = self.rand_feature_gen().next()

                # calculate the intensity difference for all training eye images clustered at this node:
                cluster0 = []
                cluster1 = []
                for imgfile in image_list:
                    image = cv2.imread(imgfile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    w, h = image.shape # TODO: are these the correct respective height and width values?
                    image = numpy.int_(image) # since cv2 images are numpy arrays of unsigned ints, make them signed
                    pixelcoord1 = ( round((feature[0][0]+1)/2*(w-1)), round((feature[0][1]+1)/2*(h-1)) )
                    pixelcoord2 = ( round((feature[1][0]+1)/2*(w-1)), round((feature[1][1]+1)/2*(h-1)) )
                    intensity_diff = image[pixelcoord1] - image[pixelcoord2]
                    if intensity_diff < 0:
                        cluster0.append(imgfile)
                    else:
                        cluster1.append(imgfile)

                if len(cluster0) >= minclustersize and len(cluster1) >= minclustersize:
                    splitting_candidates.append([cluster0, cluster1, feature])
                if len(splitting_candidates) == ncandidate_features:
                    done_clustering = True


            optimal = self.get_best_clustering(splitting_candidates)

            # recurse:
            child0 = self.cluster_images(splitting_candidates[optimal][0], d+1)
            child1 = self.cluster_images(splitting_candidates[optimal][1], d+1)

            sys.stdout.write('.')
            sys.stdout.flush()

            return [ child1, child0, splitting_candidates[optimal][2] ]

        else:
            return


if __name__ == '__main__':

    # if the training data folder is empty, make the training data:
    if os.listdir(training_data_folder_processed) == []:
        tdg = training_data_gen()
        tdg.make_training_data(training_data_folder_raw, training_data_folder_processed)

    training_data_filelist = glob.glob(training_data_folder_processed+'/BioID_????_eye?_????.jpg')


    # initialize (instantiate?) the eye tracker object
    et = EyeTracker(tree_depth)

    print 'number of splitting to calculate:', sum([2**d for d in range(tree_depth-1)])
    tree = et.cluster_images(training_data_filelist)

    sys.stdout.write('\n')



    ##### train the random forest classifier #####
#    et.train(bioid_database)


