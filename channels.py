import numpy as np
import scipy.io
from scipy.misc import imread
from scipy.misc import imsave
import cv2
from skimage.feature import hog
#import mahotas

#size of patch
fsize = 28
#step size
ds = 28
#reading list of rgb file names
f1 = open('left1.txt','r')
lines1=[line.strip() for line in open('left1.txt')]
f1.close()
#reading list of labelled images
f2 = open('label1.txt','r')
lines2=[line.strip() for line in open('label1.txt')]
f2.close

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
            '''
            print 'Pixel :',count+1,'lbp code:',code
            array=np.zeros((1,8))

            for k in xrange(0,8):
                if (code & (1 << k)):
                    array[0][k]=1
            print 'bcode:',array
            '''
    return dst

#creating list to store data and labels
D=[]
L=[]
nOfFiles=lines1.__len__()
for f in xrange(0,nOfFiles):
    rgb = cv2.imread(lines1[f])
#creating luv color chanel
    luv_img = cv2.cvtColor(rgb,cv2.COLOR_BGR2LUV)
    print 'luv image shape:',luv_img.shape
#creating gradient image
    grad_img = cv2.Laplacian(rgb,cv2.CV_8U)
    grad_img1 = cv2.cvtColor(grad_img,cv2.COLOR_BGR2GRAY)
    print 'grad image shape:',grad_img.shape
#creating HOG image and scaling values
    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    fdhog,hog_img1 = hog(gray,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=True,normalise=True)
    print 'hog image shape:',hog_img1.shape
    hog_img2 = 300*hog_img1
#creating lbp image
    lbp_img=lbp(gray)
    print 'lbp image shape:',lbp_img.shape
#reading labels
    img = cv2.imread(lines2[f])
    lbl = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print 'RGB File being processed',lines1[f]
    print 'Label File being processed',lines2[f]
    #since dimension of lbp deciding factor
    m,n=lbp_img.shape
    patch=0
#getting no of patches
    for i in xrange(0,m,ds):
        for j in xrange(0,n,ds):
            im=i+fsize
            jm=j+fsize
            if ((im<=m) and (jm<=n)):
                patch=patch+1

    print 'pathces formed:',patch
    print 'shape Of label:',lbl.shape
#making fsize*fsize 2-D array to store values of each type
    c_l = np.zeros((fsize,fsize))
    c_u = np.zeros((fsize,fsize))
    c_v = np.zeros((fsize,fsize))
    c_g = np.zeros((fsize,fsize))
    c_hog = np.zeros((fsize,fsize))
    c_lbp = np.zeros((fsize,fsize))
    c_lbl = np.zeros((fsize,fsize))
    #p=0
    for i in xrange(0,m,ds):
        for j in xrange(0,n,ds):
            ik = i + fsize
            jk = j + fsize
            if ((ik <= m) and (jk <= n)):
                for a, c in zip(xrange(i,ik), xrange(0,fsize)):
                    for b, d in zip(xrange(j,jk), xrange(0,fsize)):
                        #print c,d
                        c_l[c][d] = luv_img[a][b][0]
                        c_u[c][d] = luv_img[a][b][1]
                        c_v[c][d] = luv_img[a][a][2]
                        c_g[c][d] = grad_img1[a][b]
                        c_hog[c][d] = hog_img2[a][b]
                        c_lbp[c][d] = lbp_img[a][b]
                        c_lbl[c][d] = lbl[a][b]
            '''
            cv2.imwrite('patch_a'+str(p)+'.png',c_l)
            cv2.imwrite('patch_b'+str(p)+'.png',c_u)
            cv2.imwrite('patch_c'+str(p)+'.png',c_v)
            cv2.imwrite('patch_d'+str(p)+'.png',c_g1)
            cv2.imwrite('patch_e'+str(p)+'.png',c_g2)
            cv2.imwrite('patch_f'+str(p)+'.png',c_g3)
            cv2.imwrite('patch_g'+str(p)+'.png',c_hog)
            cv2.imwrite('patch_h'+str(p)+'.png',c_lbp)
            cv2.imwrite('patch_i'+str(p)+'.png',c_lbl)
            p=p+1
            #print 'current label:',c_lbl[8][8]
            '''
            D.append(c_hog)
            D.append(c_lbp)
            D.append(c_g)
            D.append(c_l)
            D.append(c_u)
            D.append(c_v)
            L.append(c_lbl[8][8])

D=np.asarray(D)
L=np.asarray(L)
print 'Feature shape:',D.shape
print 'label shape:',L.shape
print 'Total files processed:', nOfFiles
#Saving the file to mat to convert it to hdf5
scipy.io.savemat('data1.mat',{'D':D})
scipy.io.savemat('label1.mat',{'L':L})
