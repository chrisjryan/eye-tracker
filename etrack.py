#!/usr/bin/env python

"""Webcam Eye-tracker 
	CJ Ryan, Hacker School Winter 2014

	Implemenatation reverse engineered from 'Eye pupil localization with an ensemble of randomized trees' (Markusa, Frljaka, Pandzia, Ahlbergb, Forchheimer)

	guided also by Tom Ballinger's `gazer' (see Github) """


import os
import time
import csv

import cv
import cv2 # do you need both cv and cv2? Apparently cv2 returns everything in Numpy, that's the main difference
import numpy
import glob

from sklearn.ensemble import RandomForestRegressor # can you use the Q splitting criterion instead of the information gain one?
import random
from itertools import combinations

import cvnumpyconvert
from mouse import setMousePosition

from face_finder import * # is this bad style, since it's unclear later in the code where those functions are scoped?

# for GraphViz tree visualization, testing purposes only:
import StringIO
import pydot
from sklearn.tree import export_graphviz # can you use the Q splitting criterion instead of the information gain one?
import copy

# Parameters from the paper:
# n trees: 100 ('n_estimators')
# tree depth: 10 ('max_depth')
# n_features: 1, since binary? I think pixel intensity if the only feature ('max_features')
# 
# shrinkage, nu = 0.4 ('learning_rate'?)
# 
# TODO: 
#   put classes into files.


# note: the current face/eye detection scheme is a combintation of the one from Tom's 'gazer' and the OpenCV tutorial "Face Detection using Haar Cascades".


# take these as command line parameters, via argparse, eventually (right?):
ntrees = 100 # value from markus paper
nsamples = 100 # value from markus paper, (# sample sets?)
tree_depth = 10 # value from markus paper
nu = 0.4 # shrinkage parameter, value from markus paper






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



    def cluster_images(image_list, d): # d is node depth

        if d < self.tree_depth:

            # generate random features on [(-1,+1), (-1,+1)]:
            ncandidate_features = 4*d + 4
            # you might make this a list of (doubled) named tuples, to make the "pixelcoord_ =" lines more readble
            rand_feature_list = [( (random.uniform(-1,1), random.uniform(-1,1)), (random.uniform(-1,1), random.uniform(-1,1)))
                             for r in range(ncandidate_features)]

            # for each feature, calculate the intensity differences for all training eye images clustered at this node:
            # (there's probably a crafty functional-ish way to do this):
            cluster0 = []
            cluster1 = []
            for imgfile in image_list:
                for feature in rand_feature_list:
                    image = cv2.imread(imgfile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    w, h = image.shape # TODO: is this the correct respective height and width values?
                    pixelcoord1 = ( round((feature[0][0])*w), round((feature[0][1])*h) )
                    pixelcoord2 = ( round((feature[1][0])*w), round((feature[1][1])*h) )
                    intensity_diff = pixelcoord1 - pixelcoord2
                    if intensity_diff < 0:
                        cluster0.append(image)
                    else:
                        cluster1.append(image)


            # recurse:
            child0 = cluster_images(cluster0, d-1)
            child1 = cluster_images(cluster1, d-1)

            return [child1, child0]

        else:
            # is this ok, or should these be empty lists?
            return





class training_data_gen():

    def __init__(self):
        self.ff = FaceFinder()


    def draw_plus(self, image, coord, width=20, color=(0,0,255)):    
        img_size = numpy.shape(image)

        # note: aguments of the CvPoint type must be tuples and not lists
        cv2.line(image, tuple(map(int, numpy.around((coord[0], coord[1]-width/2)))), tuple(map(int, numpy.around((coord[0], coord[1]+width/2)))), color)
        cv2.line(image, tuple(map(int, numpy.around((coord[0]-width/2, coord[1])))), tuple(map(int, numpy.around((coord[0]+width/2, coord[1])))), color)


    def in_eye_box(self, (pupx, pupy), eyes):
        for (ex,ey,ew,eh) in eyes:
            if pupx > ex and pupx < ex+ew and pupy > ey and pupy < ey+eh:
                return True
        if True:
            return False


    def write_eye_data(self, eyes, trainingfacefile, image, pupil_coords, subimg_num=''):
        for idx,(ex,ey,ew,eh) in enumerate(eyes):

            # deepcopying a separate object for annotation seems to be necessary:
            image_annotated = copy.deepcopy(image)

            filename = './training_data_folder_processed/'+os.path.splitext(os.path.basename(trainingfacefile))[0]+'_eye'+str(idx)+'_'+subimg_num
            # write eye image without annotation:
            eye_imfile = filename+'.jpg'
            cv2.imwrite(eye_imfile, image[ey:ey+eh, ex:ex+ew])

            # draw pre-annotated pupil coordinates:
            # (use named tuples eventually)
            self.draw_plus(image_annotated, pupil_coords[0])
            self.draw_plus(image_annotated, pupil_coords[1])

            # write eye image with annotation:
            eye_clean_imfile = filename+'_landmarked.jpg'
            cv2.imwrite(eye_clean_imfile, image_annotated[ey:ey+eh, ex:ex+ew])

            # convert the pupil coords to ([-1, +1], [-1, +1]) interval & write to file:
            pupil_coords_mapped = ((pupil_coords[idx][0]-ex)/ew*2-1,  (pupil_coords[idx][1]-ey)/eh*2-1)
            pupcoordfile = open(filename+'.eye','w')
            pupcoordfile.write('# pupil coords on [-1,+1], [-1,+1] interval \n')
            writer = csv.writer(pupcoordfile, delimiter='\t')
            writer.writerow(pupil_coords_mapped)
            pupcoordfile.close()



    def make_training_data(self, training_data_filelist, training_data_folder_processed, maxjitx=10):

        maxjity = 0.5*maxjitx

        for trainingfacefile in training_data_filelist:
            # (not sure why the second arg is necessary, since the images are gray to start with)
            image = cv2.imread(trainingfacefile, cv2.CV_LOAD_IMAGE_GRAYSCALE)

            # read in landmarked pupil coords, reverse them so 0=left eye on image, 1=right eye on image:
            pupil_coords = numpy.genfromtxt(os.path.splitext(trainingfacefile)[0]+'.eye')
            pupil_coords = [pupil_coords[2:4], pupil_coords[0:2]]

            # search for a face on the image using OpenCV subroutines:
            f = self.ff.find_face(cv.fromarray(image))
            if f:
                # search for eyes on the face using OpenCV subroutines:
                eyes = self.ff.find_eyes(cv.fromarray(image), f)

                # for training, only use the images where 2 eye boxes are found, and check that the "landmarked" pupil coordinates lie within these boxes
                # TODO: allow for 1 eye only to be found (though, note that sometimes 2 boxes find the same eye. can this be specified against in OpenCV coords?)
                num_eyes_found = numpy.shape(eyes)[0]
                if num_eyes_found == 2:

                    if self.in_eye_box(pupil_coords[0], eyes) and self.in_eye_box(pupil_coords[1], eyes):

                        # sort the eyes, so you can index left and right:
                        eyes = eyes[numpy.argsort(eyes[:,0])]

                        # write the un-jittered, original eye images:
                        self.write_eye_data(eyes, trainingfacefile, image, pupil_coords, ('%04d' % 1))

                        # generate 99 jitterings of the eye to artificially expand the data set:
                        for i in range(2,11):
                            jitterleft  = eyes[0] + numpy.array((random.randint(-maxjitx,maxjitx), random.randint(-maxjity,maxjity), 0, 0))
                            jitterright = eyes[1] + numpy.array((random.randint(-maxjitx,maxjitx), random.randint(-maxjity,maxjity), 0, 0))
                            self.write_eye_data((jitterleft, jitterright), trainingfacefile, image, pupil_coords, ('%04d' % i))

                    else:
                        print 'image not found, or not usable' 

        # print a running status to the terminal of what face you are on



if __name__ == '__main__':

    training_data_folder_processed = './training_data_folder_processed'

    # if the training data folder is empty, make the training data
    if os.listdir(training_data_folder_processed) == []:
        # identify the indices of good training face data, along with eye coords
        tdg = training_data_gen()
        training_data_filelist = glob.glob("./training_data_raw/BioID/BioID-FaceDatabase-V1.2/*pgm")

        tdg.make_training_data(training_data_filelist, training_data_folder_processed)

    # TODO:
    # if training data folder does not exist, generate training data
    #   loop over all face images for usable eyes (either 1 or 2 eyes is ok)
    #       for each eye box cotaining a landmarked pupil:
    #           generate 100 random "jitterings" of this eye box
    #           (note that the authors used a "basic rectangle" since the eye is roughly rectangular, and you'll have to estimate this hueristically in your choice of sampling window)
    #               save each random jittering as a file along with the scaled pupil coord (check this by saving the iamge with and without the plus sign adeed)


#    cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE)

    # initialize (instantiate?) the eye tracker object
#    et = EyeTracker()



    ##### train the random forest classifier #####
#    et.train(bioid_database)


