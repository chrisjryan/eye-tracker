#!/usr/bin/env python

"""Webcam Eye-tracker 
	CJ Ryan, Hacker School Winter 2014

	Implemenatation reverse engineered from 'Eye pupil localization with an ensemble of randomized trees' (Markusa, Frljaka, Pandzia, Ahlbergb, Forchheimer)

	guided also by Tom Ballinger's `gazer' (see Github) """


import os
import time

import cv
import cv2 # do you need both cv and cv2? Apparently cv2 returns everything in Numpy, that's the main difference
import numpy
import glob

from sklearn.ensemble import RandomForestRegressor # can you use the Q splitting criterion instead of the information gain one?
from random import randint
from itertools import combinations

import cvnumpyconvert
from mouse import setMousePosition


# for GraphViz tree visualization, testing purposes only:
import StringIO
import pydot
from sklearn.tree import export_graphviz # can you use the Q splitting criterion instead of the information gain one?


# Parameters from the paper:
# n trees: 100 ('n_estimators')
# tree depth: 10 ('max_depth')
# n_features: 1, since binary? I think pixel intensity if the only feature ('max_features')
# 
# shrinkage, nu = 0.4 ('learning_rate'?)
# 


# note: the current face/eye detection scheme is a combintation of the one from Tom's 'gazer' and the OpenCV tutorial "Face Detection using Haar Cascades".


# take these as command line parameters, via argparse, eventually (right?):
ntrees = 100 # value from markus paper
nsamples = 100 # value from markus paper, (# sample sets?)
tree_depth = 10 # value from markus paper
nu = 0.4 # shrinkage parameter, value from markus paper



class EyeTracker(object):
    def __init__(self):
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

        # CJR:
        self.rfr = RandomForestRegressor(n_estimators=ntrees, max_depth=tree_depth)
#        add the classifier stuff here



    def detect(self, image):

        f = self.find_face(image)
        if f:
            eyes = self.find_eyes(image, f)
            num_eyes_found = numpy.shape(eyes)[0]
            if num_eyes_found == 2:

                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(numpy.asarray(image),(f[0][0]+ex,f[0][1]+ey),(f[0][0]+ex+ew,f[0][1]+ey+eh),(0,255,0),2)

                cv.ShowImage('a_window', image)
                #cv2.imshow('a_window', image)
                cv.WaitKey(0)


    # self is needed here?
    def get_pixel_samples(self, image, nsamples):
        w, h = cv.GetSize(image)
            # it's uninuitive how height should be indexed first when referring to pixel values later...
#        return [ (randint(0, h - 1), randint(0, w - 1)) ] for _ in range(nsamples) ]
        # check for a numpy funtion that does this too

        rand_pixel_indices = [(randint(0, h - 1), randint(0, w - 1)) for _ in range(nsamples)]




    def train(self, training_data_files):


        rfr = RandomForestRegressor(max_depth=tree_depth)

        # for each regression tree...:
        for t in range(ntrees):


            # "at each node"
            # (set up a recursive scheme for this)

            # uniformly at random, choose the set of 100 pixel samples:
            samples = []
            samples_coords = self.get_pixel_samples(image, nsamples)
            for sc in samples_coords:
               samples.append(image[sc[0], sc[1]])


            for trainingface in training_data_files:
                # the second option below might be unnecessary (check this), the images are gray
                image = cv2.imread(trainingface, cv2.CV_LOAD_IMAGE_GRAYSCALE)

                # convert from numpy array to CvArr for use with cv instead of cv2 (for now, but use either cv or cv2 consistently at some point)
                image = cv.fromarray(image)

                # if you find 2 eye boxes in the image with 2 landmarked pupils inside...:
                pupil_coords_file = numpy.genfromtxt(os.path.splitext(trainingface)[0]+'.eye')
                if et.check_usability(image, pupil_coords_file):

                    # calculate the intensities between each pixel pair (feature set size is (nsamples choose 2):
                    # intensity_features := [(xcood, ycoord), intensity_difference] for each sampled pixel pairing
#                    intensity_features = []
#                    for p1, p2 in combinations(range(0,nsamples), 2):
                        # this should be absolute value, right?:
#                        intensity_features.append([(p1, p2), abs(samples[p1] - samples[p2])])

                    # fit the image data to a random forest
                    # can X and y be unpacked in 1 line?
                    y = [coord for (coord, idiff) in intensity_features]
                    X_list = [idiff for (coord, idiff) in intensity_features]
                    X = numpy.array(X_list)[:,None] # this is needed to change the shape of X from (nfeatures,) to (nfeatures,1), but you should find a cleaner way to do this (maybe just work with arrays earlier on)
                    rfr.fit(X, y) 






                    # test: make graph images of the forest:
                    # for idx,tr in enumerate(rfr.estimators_):
                    #     out = StringIO.StringIO()
                    #     export_graphviz(tr, out_file = out)
                    #     graph = pydot.graph_from_dot_data(out.getvalue())

                    #     graphviz_dir = './temp_graphviz/tree' + str(idx)
                    #     os.makedirs(graphviz_dir)
                    #     graph.write_png(graphviz_dir+"/decision_tree.png")





#               Notes:
#               * rfr.get_params(deep=True) doesn't return anything extra compared to rint rfr.get_params(deep=True), I think becuase it considers all relevant params to be in each individual tree and then averages the om the fly later when called upon to.

                # cluster, in a regression tree, based on the intensity differences

                # 1) for each training set, grow a regression tree based on the MSE criterion where X is the pixel intensity difference
                # 2) at each leaf, get the mean coordinates of the associated pixels
                # 2a) plot each cluster. does it look okay?
                # 3) what mean pupil cluster location is closest to the real one? Save this "threshold window" 


                # average the coordinates that are clustered in each leaf node

                # compare the mean coord of each leaf to the real pupil coord.

                # what's the best threshold [window]? Save these parameters (add them to a running average) and then train another tree. On a new set of pixels.





    # this is somewhat reduundant with detect(), find some way to avoid code reuse
    def check_usability(self, image, pupil_coords):

        f = self.find_face(image)
        if f:
            eyes = self.find_eyes(image, f)

            # for training the classifier, only use the images where 2 eye boxes are found, and check that the "landmarked" pupil coordinates lie within these boxes
            num_eyes_found = numpy.shape(eyes)[0]

            if num_eyes_found == 2:

                # make the coordinates of the eye boxes refer to the image frame and not the face box:
                eyes[:,0] += f[0][0]
                eyes[:,1] += f[0][1]

                # draw a rectangle around each eye:
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(numpy.asarray(image),(ex, ey),(ex+ew, ey+eh),(0,255,0),2)

                # draw pre-annotated pupil coordinates for training set:
                pupil_left, pupil_right = pupil_coords[0:2], pupil_coords[2:4]
                draw_plus(image, pupil_left)
                draw_plus(image, pupil_right)


                if self.in_eye_box(pupil_left, eyes) and self.in_eye_box(pupil_right, eyes):
                    return True
                    ##cv2.imshow('a_window', image)
                    #cv.ShowImage('a_window', image)
                    #cv.WaitKey(0)
                else:
                    return False



    def in_eye_box(self, (pupx, pupy), eyes):
        for (ex,ey,ew,eh) in eyes:
            if pupx > ex and pupx < ex+ew and pupy > ey and pupy < ey+eh:
                return True
        if True:
            return False


    def find_eyes(self, image, f):
        # get the total image size:
        w, h = cv.GetSize(image)

        # CreateImage((width, 2/3*(height)),8 bits, 1 channel) # see rect() to better understand what f is
        faceimg = cv.CreateImage((f[0][2], f[0][3],), 8, 1)
        src_region = cv.GetSubRect(image, (f[0][0], f[0][1], f[0][2], f[0][3]))
        cv.Copy(src_region, faceimg)

        eyes = self.eye_cascade.detectMultiScale(numpy.asarray(faceimg[:,:]))

        return eyes


    def find_face(self, image):

        w, h = cv.GetSize(image)
        # I actually think cv.CreateImage returns a BGR image, but it gets gray-scaled in the line after:
        grayscale = cv.CreateImage((w, h), 8, 1)
        #print 'num channels =',image.channels
        #cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)


        # OpenCV method to detect faces, already trained machine (decision stump [1-leve decision tree] with adaboost)
        # note: if face detecton seems off you might have to tweak the minimum object size argument
        faces = cv.HaarDetectObjects(grayscale, self.face_cascade, self.storage, 1.2, 2, 0, (100, 100))

        if faces:
            print 'face detected!'
            # if it found more than 1 face it will cycle through each:
            for f in faces:
                rect(image, f, (0, 255, 0)) # this draws a green (black in grayscale) rectangle to frame the object that was found
                self.frames_since_face = 0 # a hack, mainly for the no-face-detected case
                self.last_face_position = f # remember this face as the last one, again for the no-face-detected case
                # won't this only return the 1st face found, exiting the function? I think you'd need to return a list of faces outside of the for loop to return both

                # show the image (box will be black since `image' is 1-channel)
                return f

        # if it didn't find a face it will draw one where the last one was, so there's no blank. this is a good guess anyway
        # (BUG (maybe): I think if 2 or more faces were detected in the last frame, this will only draw the most recent of them)
        elif self.last_face_position:
            # print 'can\'t find face, using old postion'
            self.frames_since_face += 1
            f = self.last_face_position
            rect(image, f, (0, 100, 200)) # gray in grayscale
            return f
        else:
            print 'no face'




def rect(image, result, color=(0,0,255)):
    f = result
    cv.Rectangle(image, (f[0][0], f[0][1]),
            (f[0][0]+f[0][2], f[0][1]+f[0][3]),
            cv.RGB(*color), 3, 8, 0)


def draw_plus(image, coord, width=20, color=(0,0,255)):    
    img_size = numpy.shape(image)
    # note: aguments of the CvPoint type must be tuples and not lists
    cv.Line(image, tuple(map(int, numpy.around((coord[0], coord[1]-width/2)))), tuple(map(int, numpy.around((coord[0], coord[1]+width/2)))), color)
    cv.Line(image, tuple(map(int, numpy.around((coord[0]-width/2, coord[1])))), tuple(map(int, numpy.around((coord[0]+width/2, coord[1])))), color)



# each node of the regression tree is binary, and classifies the eye based on pixel intensity
# this means that 
# coordinates are normalzied, so this does not depend on box size at all
#def find_pupils():


def make_training_data():



if __name__ == '__main__':

    

    # TODO:
    # if training data folder does not exist, generate training data
    #   loop over all face images for usable eyes (either 1 or 2 eyes is ok)
    #       for each eye box cotaining a landmarked pupil:
    #           generate 100 random "jitterings" of this eye box
    #           (note that the authors used a "basic rectangle" since the eye is roughly rectangular, and you'll have to estimate this hueristically in your choice of sampling window)
    #               save each random jittering as a file along with the scaled pupil coord (check this by saving the iamge with and without the plus sign adeed)


    cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE)

    # initialize (instantiate?) the eye tracker object
    et = EyeTracker()

    # identify the indices of good training face data, along with eye coords

    ##### train the random forest classifier #####
    bioid_database = glob.glob("./training_data/BioID/BioID-FaceDatabase-V1.2/*pgm")
    et.train(bioid_database)


        # detect the face and eyes from the image
#        et.detect(image)

        # display the image, which now contains boxes drawn on the face and eyes:
#        cv.ShowImage('asdf', image)
#        cv2.imshow('a_window', image)
#        cv.WaitKey(0)

	# train the random forest regressor:
	# for each training image:
	# a) find the face
	# b) find the eyes
	# c) 
	# function: get previously landmarked eyecoords


	# 1. Obtain a face bounding box using a face detector (i.e., OpenCV's mmthod).

	# open the image with imread(filename)




	# 2. Estimate eye regions using simple anthropometric relations (also OpenCV?).


	# 3. Estimate pupil location for each eye region using a chain of multi-scale tree ensembles ().

