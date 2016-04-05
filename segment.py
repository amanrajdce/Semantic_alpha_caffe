import numpy as np
import sys
import scipy.io
import cv2
#from scipy.misc import imread
#from scipy.misc import imsave
from skimage.feature import hog
#import mahotas

#caffe setup 
caffe_root='../'
sys.path.insert(0,caffe_root+'python')
sys.path.append('/home/amanraj/caffe-master/python')
import caffe
MODEL_FILE='/home/amanraj/caffe-master/examples/ml_task/deploy.prototxt'
PRETRAINED='/home/amanraj/caffe-master/examples/ml_task/iter_60000.caffemodel'
net=caffe.Classifier(MODEL_FILE,PRETRAINED,image_dims=(28,28))
net.set_phase_test()
net.set_mode_cpu()


#reading list of rgb file names
f1 = open('run.txt','r')
lines1=[line.strip() for line in open('run.txt')]
f1.close()

fsize = 28
ds = 20
# function to create lbp image
def lbp(src):
    dst=np.zeros((src.shape[0]-2,src.shape[1]-2))
    for i in xrange(1,src.shape[0]-1):
        #count=0
        for j in xrange(1,src.shape[1]-1):
            center=src[i][j]
            code=0
            code |= (src[i-1][j-1] > center) << 7
            code |= (src[i-1][j] > center) << 6
            code |= (src[i-1][j+1] > center ) << 5
            code |= (src[i][j+1] > center ) << 4
            code |= (src[i+1][j+1] > center ) << 3
            code |= (src[i+1][j] > center ) << 2
            code |= (src[i+1][j-1] > center ) << 1
            code |= (src[i][j-1] > center ) << 0
            dst[i-1][j-1] = code           
    return dst

#for making a channel model
nOfFiles=lines1.__len__()
for f in xrange(0,nOfFiles):
    rgb = cv2.imread(lines1[f])
#creating luv color chanel
    luv_img = cv2.cvtColor(rgb,cv2.COLOR_BGR2LUV)
    print 'luv image shape:',luv_img.shape
#creating gradient image
    grad_img = cv2.Laplacian(rgb,cv2.CV_8U)
    grad_img1 = cv2.cvtColor(grad_img,cv2.COLOR_BGR2GRAY)
    print 'grad image shape:',grad_img1.shape
#creating HOG image and scaling values
    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    fdhog,hog_img1 = hog(gray,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=True,normalise=True)
    print 'hog image shape:',hog_img1.shape
    hog_img2 = 300*hog_img1
#creating lbp image
    lbp_img=lbp(gray)
    print 'lbp image shape:',lbp_img.shape
    print 'RGB file being processed:', lines1[f]
    m,n = lbp_img.shape
    patch=0
#getting no of patches
    for i in xrange(0,m,ds):
        for j in xrange(0,n,ds):
            im=i+fsize
            jm=j+fsize
            if ((im<=m) and (jm<=n)):
                patch=patch+1

    print 'pathces formed:',patch
#making fsize*fsize 2-D array to store values of each type
    c_l = np.zeros((fsize,fsize))
    c_u = np.zeros((fsize,fsize))
    c_v = np.zeros((fsize,fsize))
    c_g = np.zeros((fsize,fsize))
    c_hog = np.zeros((fsize,fsize))
    c_lbp = np.zeros((fsize,fsize))
    D_new = np.zeros((fsize,fsize,6))
    #p=0
    for i in xrange(0,m,ds):
        for j in xrange(0,n,ds):
            ik = i + fsize
            jk = j + fsize
            if ((ik <= m) and (jk <= n)):
                for a, c in zip(xrange(i,ik), xrange(0,fsize)):
                    for b, d in zip(xrange(j,jk), xrange(0,fsize)):
                        #print c,d
                        c_l[c][d] = 0.00390625*luv_img[a][b][0]
                        c_u[c][d] = 0.00390625*luv_img[a][b][1]
                        c_v[c][d] = 0.00390625*luv_img[a][a][2]
                        c_g[c][d] = 0.00390625*grad_img1[a][b]
                        c_hog[c][d] = 0.00390625*hog_img2[a][b]
                        c_lbp[c][d] = 0.00390625*lbp_img[a][b]
            D=[]
            D.append(c_hog)
            D.append(c_lbp)
            D.append(c_g)
            D.append(c_l)
            D.append(c_u)
            D.append(c_v)
            D=np.asarray(D)
            print 'shape',D.shape
            for i1 in xrange(0,D.shape[0]):
                for j1 in xrange(0,D.shape[1]):
                    for k1 in xrange(0,D.shape[2]):
                        D_new[j1][k1][i1]=D[i1][j1][k1]
            print 'D_new shape:',D_new.shape
            prediction = net.predict([D_new])
            print 'prediction shape:',prediction[0].shape
            print 'prediction class:',prediction[0].argmax()
            #Cl=prediction[0].argmax()
            print 'prediction:',prediction[0]
            #print 'class',Cl
