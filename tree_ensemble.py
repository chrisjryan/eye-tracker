
import cv, cv2
import os
import random
import numpy
import sys




class Node(object):
    def __init__(self, clustersize, splitting_feature = None, children = None, trained_pupil_avg = None, leaf = False):
        self.splitting_feature = splitting_feature
        self.trained_pupil_avg = trained_pupil_avg
        self.leaf = leaf
        self.children = children
        self.clustersize = clustersize


    # TODO: fix this __repr__().
    # def __repr__(self):
    #     if not self.leaf:
    #         self.children[0]
    #         self.children[1]
    #         return 'splitting_feature:', str(self.splitting_feature)
    #     else:
    #         return 'average pupil coordinates:', str(self.trained_pupil_avg)


    def graphviz_print_node_lines(self, f, idx=0):
        if not self.leaf:
            parentidx = idx
            f.write(('n%03d' % idx)+(' [label="clustersize = '+str(self.clustersize)+'\nsplitting feature:\n((%.3g,%.3g)' % (self.splitting_feature[0]))+('-(%.3g,%.3g))"] ;\n'% (self.splitting_feature[1])))
            f.write(('n%03d' % idx)+' -- '+('n%03d' % (idx+1))+' ;\n')
            idx = self.children[0].graphviz_print_node_lines(f,idx+1)
            f.write(('n%03d' % parentidx)+' -- '+('n%03d' % (idx+1))+' ;\n')
            idx = self.children[1].graphviz_print_node_lines(f,idx+1)
            return idx
        else:
            f.write(('n%03d' % idx)+' [label="clustersize = '+str(self.clustersize)+'\npupil avg:\n'+str(self.trained_pupil_avg)+'"] ;\n')
            return idx



class RegressionTree(object):
    def __init__(self, tree_depth):

        self.tree_depth = tree_depth
        self.rootnode = None


    # generate random features on [(-1,+1), (-1,+1)]:
    # (this is kind of a trivial use of a generator)
    def rand_feature_gen(self):
        while True:
            yield ( (random.uniform(-1,1), random.uniform(-1,1)), (random.uniform(-1,1), random.uniform(-1,1)) )


    def get_best_clustering(self, splitting_candidates):

        # TODO: eliminate (or minimize) redundant usages of genfromtxt() in this subroutine.
        Qlist = []
        for (cluster0, cluster1, feature) in splitting_candidates:
            # get the eye coordinate file list from the image file list:
            # TODO: see if you can make this all a little cleaner
            cluster0_pupilfiles = [os.path.splitext(f)[0]+'.eye' for f in cluster0]
            cluster1_pupilfiles = [os.path.splitext(f)[0]+'.eye' for f in cluster1]
            pupcoodlist0 = [numpy.genfromtxt(pupilcoord) for pupilcoord in cluster0_pupilfiles]
            pupcoodlist1 = [numpy.genfromtxt(pupilcoord) for pupilcoord in cluster1_pupilfiles]

            # is this different from the standard deviation? can you just use a library?
            avgpupilcoord0 = numpy.mean(pupcoodlist0, axis=0)
            avgpupilcoord1 = numpy.mean(pupcoodlist1, axis=0)
            # TODO: probably don't need the lambda function here
            Q  =  sum(map(lambda x0, x0avg=avgpupilcoord0: sum((x0-x0avg)**2), pupcoodlist0))
            Q +=  sum(map(lambda x1, x1avg=avgpupilcoord1: sum((x1-x1avg)**2), pupcoodlist1))
            Qlist.append(Q)

        # TODO: Test that sqrt(Q)/Nimages gets smaller as you go further down the tree (i.e., less variance in the landmarked pupil coordinates in each cluster)

        return numpy.argmin(Qlist)



    def cluster_images(self, image_list, d=0): # d is the depth of this node

        clustersize = len(image_list)

        if d < self.tree_depth-1:

            # you might make this a list of (doubled) named tuples, to make the "pixelcoord_ =" lines more readble
            ncandidate_features = 4*d + 4
            minclustersize = 2.0**(self.tree_depth - d) # this might still slow things down, if min cluster size is achieved near the root of the tree and the min cluster sizes need to be found at each level on the way down

            splitting_candidates = []
            done_clustering = False
            while done_clustering == False:
                # kind of trivial use of a generator:
                feature = self.rand_feature_gen().next()

                # calculate the intensity difference for all training eye images clustered at this node:
                # TODO: wrap all this clustering into a function, such that image_list and feature go in and cluster0, cluster1 comes out.
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

            # TODO: consider whether you really need to save the image list for each cluster (you probably just need the params, maybe a report of Q and the cluster sizes)
            n = Node(clustersize, children = (child0, child1), splitting_feature = splitting_candidates[optimal][2])
            if d==0:
                self.rootnode = n
            return n

        else:
            # TODO: caculate the  standard error of the pupil coords, too
            # TODO: optimize code requse between this and get_best_clustering() (i.e., make a get_avg_pupilcood() function)
            pupilfiles = [os.path.splitext(f)[0]+'.eye' for f in image_list]
            pupcoodlist = [numpy.genfromtxt(pupilcoord) for pupilcoord in pupilfiles]
            return Node(clustersize, leaf = True, trained_pupil_avg = numpy.mean(pupcoodlist, axis=0))



    def export_graphviz_file(self, i):
        outfile = open('tree'+str(i)+'.dot', 'w')
        outfile.write('## [header material...]\n')
        outfile.write('## Command to get the layout: "dot -Teps thisfile > thisfile.eps"\n')
        outfile.write('graph "test"\n')
        outfile.write('{\n')
        outfile.write('node [shape="rectangle", fontsize=10, width=".2", height=".2", margin=0];\n')
        outfile.write('graph[fontsize=8];\n\n')
        self.rootnode.graphviz_print_node_lines(outfile)
        outfile.write('}')
        outfile.close()


    def predict_tree(self, eye_img, (w,h), node):
        if not node.leaf:
            feat = node.splitting_feature
            pixelcoord1 = ( round((feat[0][0]+1)/2*(w-1)), round((feat[0][1]+1)/2*(h-1)) )
            pixelcoord2 = ( round((feat[1][0]+1)/2*(w-1)), round((feat[1][1]+1)/2*(h-1)) )
            intensity_diff = eye_img[pixelcoord1] - eye_img[pixelcoord2]
            if intensity_diff < 0:
                return self.predict_tree(eye_img, (w,h), node.children[0])
#                cluster0.append(imgfile)
            else:
                return self.predict_tree(eye_img, (w,h), node.children[1])
#                cluster1.append(imgfile)
        else:
            return node.trained_pupil_avg


class TreeEnsemble(object):
    def __init__(self, ntrees, tree_depth, subsample_frac, exportgv):
        self.tree_depth = tree_depth
        self.tree_list = [RegressionTree(tree_depth) for _ in range(ntrees)]
        self.exportgv = exportgv

        self.subsample_frac = subsample_frac


    def train(self, filelist):

        Nimages = len(filelist)
        subsample_size = int(round(self.subsample_frac*Nimages))

        for idx,t in enumerate(self.tree_list):
            if self.subsample_frac == 1.0:
                t.cluster_images(filelist)
            else:
                filelist_subsamp = random.sample(filelist, subsample_size)
                t.cluster_images(filelist_subsamp)

            # output trained tree result for graph visualization:
            if self.exportgv:
                t.export_graphviz_file(idx)


    def read_trained_params(paramfile):
        sys.exit('read_trained_params() function not made yet.')
        # might be neater just tp put this into __init__()


    def predict_forest(self, eye_img): #, face_img_size): # 3rd argument will help get the correct coordinate later on

        # convert the eye image to grayscale
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY);

        w, h = eye_img.shape # TODO: are these the correct respective height and width values?

        # calculate the prediction of each tree:
        pupil_predictions = [ t.predict_tree(numpy.int_(eye_img), (w,h), t.rootnode) for t in self.tree_list ]

        # return pupil_coord, avg of predictions of all trees?
        return numpy.mean(pupil_predictions, axis=0)







