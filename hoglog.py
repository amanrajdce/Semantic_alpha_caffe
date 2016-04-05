import numpy as np
import scipy.io
from scipy.misc import imread
from scipy.misc import imsave
import cv2
from skimage.feature import hog
import mahotas


fsize = 16
ds = 16

#reading list of rgb file names
f1 = open('left.txt','r')
lines1=[line.strip() for line in open('left.txt')]
f1.close()
#reading list of disparity images
f2 = open('label.txt','r')
lines2=[line.strip() for line in open('label.txt')]
f2.close

D=[]
L=[]
nOfFiles=lines1.__len__()
for f in xrange(0,nOfFiles):
    rbg = cv2.imread(lines1[f])
    img1 = cv2.cvtColor(rbg,cv2.COLOR_BGR2GRAY)
    lbl = cv2.imread(lines2[f])
    img2 = cv2.cvtColor(lbl,cv2.COLOR_BGR2GRAY)
    print 'RGB File being processed',lines1[f]
    print 'Label File being processed',lines2[f]
    #since dimension of both type images same
    m,n=img2.shape
    patch=0
#getting no of patches
    for i in xrange(0,m,ds):
        for j in xrange(0,n,ds):
            im=i+fsize
            jm=j+fsize
            if ((im<=m) and (jm<=n)):
                patch=patch+1

    print 'pathces formed:',patch
    print 'shape Of label:',img2.shape
    v1 = np.zeros((fsize,fsize))
    v2 = np.zeros((fsize,fsize))
    p=0
    for i in xrange(0,m,ds):
        for j in xrange(0,n,ds):
            ik = i + fsize
            jk = j + fsize
            if ((ik <= m) and (jk <= n)):
                for a, c in zip(xrange(i,ik), xrange(0,fsize)):
                    for b, d in zip(xrange(j,jk), xrange(0,fsize)):
                        #print c,d
                        v1[c][d] = img1[a][b]
                        v2[c][d] = img2[a][b]
			            

            #print 'Patches shape:', v1.shape,'and',v2.shape
            #cv2.imwrite('patchd'+str(p)+'.jpeg',v1)
            #cv2.imwrite('patchl'+str(p)+'.png',v2)
            #p=p+1	
#Extracting HOG features
            fdhog = hog(v1,orientations=9,pixels_per_cell=(8,8),cells_per_block=(1,1),visualise=False,normalise=True) 
#Extracting LBP features
            fdlbp=mahotas.features.lbp(v1,1,8,ignore_zeros=True) 
            #print fdhog.shape
            #print fdhog
            D.append(fdhog) 
            #print fdlbp 
            D.append(fdlbp) 
            L.append(v2[8][8]) 
            print 'current label:',v2[8][8]
	    

D=np.asarray(D)
L=np.asarray(L)
print 'Feature shape:',D.shape
print 'label shape:',L.shape
print 'Total files processed:', nOfFiles
#Saving the file to mat to convert it to hdf5
scipy.io.savemat('data.mat',{'D':D})
scipy.io.savemat('label.mat',{'L':L})
