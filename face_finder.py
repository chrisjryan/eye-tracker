
import cv, cv2
import os
import numpy


class FaceFinder(object):
    def __init__(self):
        self.storage = cv.CreateMemStorage(0)
        self.last_face_position = None
        self.face_cascade = cv.Load(os.path.expanduser('/usr/local/Cellar/opencv/2.4.7.1//share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'))
        self.eyepair_cascade = cv.Load(os.path.expanduser('./parojosG45x11.xml'))
        self.eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.7.1/share/OpenCV/haarcascades/haarcascade_eye.xml')

        # Tom uses these:
        self.detect_times = []
        self.eye_pair_history = []
        self.xamount_histories = [[], []]
        self.yamount_histories = [[], []]
        self.xpos_history = []
        self.ypos_history = []


    def rect(self, image, result, color=(0,0,255)):
        f = result
        cv.Rectangle(image, (f[0][0], f[0][1]), (f[0][0]+f[0][2], f[0][1]+f[0][3]), cv.RGB(*color), 3, 8, 0)

    # write cv2 versions of this eventually:
    def find_eyes(self, image, f):
        # get the total image size:
        w, h = cv.GetSize(image)

        # CreateImage((width, 2/3*(height)),8 bits, 1 channel) # see rect() to better understand what f is
        faceimg = cv.CreateImage((f[0][2], f[0][3],), 8, 1)
        src_region = cv.GetSubRect(image, (f[0][0], f[0][1], f[0][2], f[0][3]))
        cv.Copy(src_region, faceimg)

        eyes = self.eye_cascade.detectMultiScale(numpy.asarray(faceimg[:,:]))

        # if eyes are found, make their coordinates refer to the image frame and not the face box:
        if eyes != ():
            eyes[:,0] += f[0][0]
            eyes[:,1] += f[0][1]

        return eyes


    # write cv2 versions of this eventually:
    def find_face(self, image):

        w, h = cv.GetSize(image)
        # I actually think cv.CreateImage returns a BGR image, but it gets gray-scaled in the line after:
        grayscale = cv.CreateImage((w, h), 8, 1)
        #print 'num channels =',image.channels
        #cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

        # note: if face detecton seems off you might have to tweak the minimum object size argument
        faces = cv.HaarDetectObjects(grayscale, self.face_cascade, self.storage, 1.2, 2, 0, (100, 100))

        if faces:
            print 'face detected!'
            # TODO: if it found more than 1 face it will cycle through each (only partially done here):
            for f in faces:
                self.rect(image, f, (0, 255, 0)) # this draws a green (black in grayscale) rectangle to frame the object that was found
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
            self.rect(image, f, (0, 100, 200)) # gray in grayscale
            return f
        else:
            print 'no face'


    # write cv2 versions of this eventually:
    # def detect(self, image):

    #     f = self.find_face(image)
    #     if f:
    #         eyes = self.find_eyes(image, f)
    #         num_eyes_found = numpy.shape(eyes)[0]
    #         if num_eyes_found == 2:

    #             for (ex,ey,ew,eh) in eyes:
    #                 cv2.rectangle(numpy.asarray(image),(f[0][0]+ex,f[0][1]+ey),(f[0][0]+ex+ew,f[0][1]+ey+eh),(0,255,0),2)

    #             cv.ShowImage('a_window', image)
    #             #cv2.imshow('a_window', image)
    #             cv.WaitKey(0)
