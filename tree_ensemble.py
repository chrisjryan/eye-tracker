
import cv, cv2
import os
import random
import numpy
# from scipy import stats
import sys
import xml.etree.ElementTree as ET
from ElementTree_pretty import prettify




class Node(object):
    def __init__(self, clustersize = None, splitting_feature = None, children = [], trained_pupil_avg = None, leaf = False):
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
    def __init__(self, tree_depth = None):

        self.tree_depth = tree_depth # TODO: this is parameter optional since it is not needed if parameters are loaded instead of trained, and it can be gotten from XML metadata if you need to report to the terminal.
        self.rootnode = None

        # right now this object is constructed in parallel to the tree I created, and contains the same information. It is only used for saving and loading XML parameter files. Eventually I should use only this object, rather than the tree class I made.
        self.rootnode_elemtree = None

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

                # print 'len(cluster0) ', len(cluster0) , ', len(cluster1) ', len(cluster1) 
                if len(cluster0) >= minclustersize and len(cluster1) >= minclustersize:
                    splitting_candidates.append([cluster0, cluster1, feature])
                if len(splitting_candidates) == ncandidate_features:
                    done_clustering = True


            optimal = self.get_best_clustering(splitting_candidates)

            # recurse:
            child0, child0_et = self.cluster_images(splitting_candidates[optimal][0], d+1)
            child1, child1_et = self.cluster_images(splitting_candidates[optimal][1], d+1)

            sys.stdout.write('.')
            sys.stdout.flush()

            # save the node in your format:
            # TODO: consider whether you really need to save the image list for each cluster (you probably just need the params, maybe a report of Q and the cluster sizes)
            n = Node(clustersize, children = [child0, child1], splitting_feature = splitting_candidates[optimal][2])

            # save the node in the ElementTree format as well:
            et_node = ET.Element('midnode')
            sc = ET.SubElement(et_node, 'splitting_feature')
            sc.text = str(splitting_candidates[optimal][2])
            cl = ET.SubElement(et_node, 'clustersize')
            cl.text = str(clustersize)
            et_node.extend([child0_et, child1_et, sc, cl])

            if d==0:
                self.rootnode = n
                et_node.tag = 'root'
                self.elemtree = ET.ElementTree(et_node)

            return n, et_node

        else:
            # TODO: caculate the  standard error of the pupil coords, too
            # TODO: optimize code requse between this and get_best_clustering() (i.e., make a get_avg_pupilcood() function)
            pupilfiles = [os.path.splitext(f)[0]+'.eye' for f in image_list]
            pupcoodlist = [numpy.genfromtxt(pupilcoord) for pupilcoord in pupilfiles]

            # save the node in your format:
            n = Node(clustersize, leaf = True, trained_pupil_avg = numpy.mean(pupcoodlist, axis=0))

            # save the node the element tree format:
#            et_node = ET.Element('temp', {'trained_pupil_avg': numpy.mean(pupcoodlist, axis=0), 'clustersize': clustersize})
            et_node = ET.Element('leaf')
            avg = ET.SubElement(et_node,'trained_pupil_avg')
            avg.text = str(numpy.mean(pupcoodlist, axis=0))
            cl = ET.SubElement(et_node, 'clustersize') 
            cl.text = str(clustersize)

            return n, et_node


    def export_graphviz_file(self, i, folder):
        outfile = open(folder+'/tree'+str(i)+'.dot', 'w')
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
            else:
                return self.predict_tree(eye_img, (w,h), node.children[1])
        else:
            return node.trained_pupil_avg


class TreeEnsemble(object):
    def __init__(self, subsample_frac, exportgv, saveXML):
        self.subsample_frac = subsample_frac
        self.exportgv = exportgv
        self.saveXML = saveXML

        # these are set either in train() or loadparams():
        self.tree_depth = None
        self.tree_list = None
        self.training_result_dir = ''

        # if self.saveXML:
        #     self.training_result_dir = './random_forest_params/'
        #     if not os.path.exists(self.training_result_dir):
        #         os.mkdir(self.training_result_dir)


    def train(self, training_data_filelist, tree_depth, ntrees):

        self.tree_depth = tree_depth
        self.tree_list = [RegressionTree(tree_depth) for _ in range(ntrees)]

        nimages = len(training_data_filelist)
        subsample_size = int(round(self.subsample_frac*nimages))

        if self.saveXML:
            # make a directory to contain trained tree info:
            # TODO: ask the user if they want to overwrite files in this dir or not.
            self.training_result_dir = './random_forest_params/depth%intrees%inimages%i' % (tree_depth, ntrees, subsample_size)
            if not os.path.exists(self.training_result_dir):
                os.makedirs(self.training_result_dir)


        for idx,t in enumerate(self.tree_list):
            # create each tree w/ training data:
            if self.subsample_frac == 1.0:
                t.cluster_images(training_data_filelist)
            else:
                filelist_subsamp = random.sample(training_data_filelist, subsample_size)
                t.cluster_images(filelist_subsamp)

            # write XML parameter file for each tree (more human-readable than elemtree.write()):
            if self.saveXML:
                with open(self.training_result_dir+'/tree%i.xml' % (idx),'w') as f:
                    f.write(prettify(t.elemtree.getroot()))

            # output trained tree result for graph visualization:
            if self.exportgv:
                t.export_graphviz_file(idx, self.training_result_dir)


    # TODO: This is a little cunky, since it maps the element tree onto my own, more specialized RegressionTree object; Eventually I should use just Element trees so no conversion is necessary. Or it's probably easier to just write my own XML file writer for my object.
    def map_node_params(self, elem):

        n = Node()

        # this loop iterates over ElementTree subelements for each node, which in our case includes child nodes, features, avgs, and clsuters sizes.
        for e in list(elem):
            if e.tag == 'clustersize':
                n.clustersize = int(e.text)
            elif e.tag == 'splitting_feature':
                n.splitting_feature = eval(e.text)
            elif e.tag == 'trained_pupil_avg':
                n.trained_pupil_avg = tuple(map(float, e.text[1:-1].split())) # TODO: do this more cleanly
                n.leaf = True
            elif e.tag == 'leaf' or e.tag == 'midnode':
                n.children.append(self.map_node_params(e))
            else:
                sys.exit("Error: misformatted XML parameter file (unknown subelement).")

        return n


    def loadparams(self, tree_param_filelist):

        Ntrees = len(tree_param_filelist)
        self.tree_list = [RegressionTree() for _ in range(Ntrees)]

        for t, f in zip(self.tree_list, tree_param_filelist):
            etree = ET.ElementTree()
            etree.parse(f)
            t.rootnode = self.map_node_params(etree.getroot())


    def predict_forest(self, eye_img): #, face_img_size): # 3rd argument will help get the correct coordinate later on

        # convert the eye image to grayscale
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY);

        w, h = eye_img.shape # TODO: are these the correct respective height and width values?

        # calculate the prediction of each tree:
        pupil_predictions = [ t.predict_tree(numpy.int_(eye_img), (w,h), t.rootnode) for t in self.tree_list ]

        # return avg of predictions of all trees:
        return numpy.mean(pupil_predictions, axis=0)


    # def partitiondata_cv(self, filelist, trainingset_frac = 0.5):
    #     # Nimages = len(filelist)
    #     # subsample_size = int(round(self.subsample_frac*Nimages))
    #     # filelist_subsamp = random.sample(filelist, subsample_size)
    #     random.shuffle(filelist)
    #     filelist_len = len(filelist)
    #     divider = int(filelist_len*trainingset_frac)
    #     return filelist_subsamp[:divider], filelist_subsamp[divider:]


    def partitiondata_cv(self, filelist, npredictions = 500):
        random.shuffle(filelist)
        return filelist[npredictions:], filelist[:npredictions]


    def crossval(self, filelist, ff, depth, ntrees):

        # use the subsample_frac to get the "full" file list (i.e., not every file but before taking the 50% CV sample), then shuffle it
        training_set, prediction_set = self.partitiondata_cv(filelist)

        self.train(training_set, depth, ntrees)

        errorlist = []
        for img_file in prediction_set:

            eye_img = cv2.imread(img_file)#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            pupil_predicted = self.predict_forest(eye_img)
            pupil_landmarked = numpy.genfromtxt(os.path.splitext(img_file)[0]+'.eye')
            # print 'predicted:', pupil_predicted
            # print 'landmarked:', pupil_landmarked
            errorlist.append(pupil_predicted - pupil_landmarked)

            # draw & show eye, with landmark and prediction
            # w, h = eye_img.shape[:2]
            # ff.draw_pupil(eye_img, pupil_predicted, (0,0,w,h))
            # ff.draw_plus(eye_img, pupil_landmarked, (0,0,w,h), map_to_pixeldims=True)
            # cv.ShowImage('asdf', cv.fromarray(eye_img))
            # cv.WaitKey(0)


        # return the average prediction error:
        return numpy.mean([abs(e) for e in errorlist])











