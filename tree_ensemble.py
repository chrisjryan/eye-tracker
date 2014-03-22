from regression_tree import *


class TreeEnsemble(object):
    def __init__(self, ntrees, tree_depth, training_data_filelist):

        self.tree_depth = tree_depth
        self.training_data_filelist = training_data_filelist
        self.tree_list = [RegressionTree(tree_depth) for _ in range(ntrees)]


    def train(self):
#        for t in range(self.ntrees):
        for idx,t in enumerate(self.tree_list):
            # TODO: feed in a random subset of images instead of all, like Markus et al do, to make trees more random:

            # clsuter the images:
            t.cluster_images(self.training_data_filelist)

            # output data for graph visualization (TODO: make this an option)
            t.export_graphviz_file(idx)


##### much of the following adapted from Tom Ballinger's 'gazer' repo #####
# TODO: put this in a separate class, with a pointer to the trained tree ensemble?

    def find_eyes(self, image, f):
        w, h = cv.GetSize(image)
        # CreateImage((width, 2/3*(height)),8 bits, 1 channel) # see rect() to better understand what f is
        small = cv.CreateImage((f[0][2], f[0][3]*2/3,), 8, 3)
        # get the slightly cropped image of the face, roughly centering the eyes and removing some chin:
        src_region = cv.GetSubRect(image, (f[0][0], f[0][1],
            f[0][2], f[0][3]*2/3))
        # ... then copy it into the image in the window
        cv.Copy(src_region, small)
        grayscale = cv.CreateImage((f[0][2], f[0][3]*2/3), 8, 1)
        cv.CvtColor(small, grayscale, cv.CV_BGR2GRAY)
#        grayscale = cv.CreateImage((f[0][2], f[0][3]*2/3), 8, 1)
#        cv.CvtColor(small, grayscale, cv.CV_BGR2GRAY)
        eye_pairs = cv.HaarDetectObjects(grayscale, self.eye_cascade, self.storage, 1.2, 2, 0, (10, 10))
        for eye_pair in eye_pairs:
            eye_pair = ((eye_pair[0][0]+f[0][0], eye_pair[0][1]+f[0][1],
                eye_pair[0][2], eye_pair[0][3]), eye_pair[1])
            rect(image, eye_pair, (255,0,0))
            return eye_pair


    def find_face(self, image):

        w, h = cv.GetSize(image)
        grayscale = cv.CreateImage((w, h), 8, 1)
        cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

        cv.ShowImage('a_window', grayscale)
        cv.WaitKey(0)

        faces = cv.HaarDetectObjects(grayscale, self.face_cascade, self.storage, 1.2, 2, 0, (300, 250))

        if faces:
            #print 'face detected!'
            for f in faces:
                rect(image, f, (0, 255, 0)) # this draws a green rectangle to frame the object that was found
                self.frames_since_face = 0
                self.last_face_position = f
                return f

        elif self.last_face_position:
            # print 'can\'t find face, using old postion'
            self.frames_since_face += 1
            f = self.last_face_position
            rect(image, f, (0, 100, 200))
            return f
        else:
            print 'no face'


    def detect(self, image):

        f = self.find_face(image)
        if f:
            # call the find eyes funciton that Tom wrote (use this too, this is just OpenCV's method):
            eyes = self.find_eyes(image, f)
            if eyes:
                # roll back the eye history:
                rolling_eyes = self.rolling_eye_pair(eyes, samples=5)
                # find black portions of outside thirds of image
                return eyes
            else:
                return None # it does this even without this else statement


    def track_eyes():
        # CJR: creates a window named 'a_window', the size of which will automatically fit the image
        cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE)

        while True:

            # return a capture obecct from the camera, which is index 0 by default:
            capture = cv.CaptureFromCAM(0)

            # get the image from the video capture object
            image = cv.QueryFrame(capture)

            # detect the face, eyes, then pupil from the image
            # (note that "left" refers to the image's left eye, not the person's left eye, etc)
            eyes = et.detect(image)
            if eyes:
                left_pupil  = et.predict(left_eye[0])
                right_pupil = et.predict(right_eye[1])

#                et.draw_pupils(image, left_pupil, right_pupil)

            cv.ShowImage('asdf', image)


    def printParams():
        for idx, tree in enumerate(self.tree_list):
            print 'tree:', idx 
            tree.printParams()
















