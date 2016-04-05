import h5py
import scipy.io as sio
import numpy as np

mat1=sio.loadmat('data.mat') #training data
mat2=sio.loadmat('label.mat') #labels
f5=mat1["D"]
t5=mat2["L"]
#list for column major of data
images1=[]

for i in xrange(0,f5.shape[0]):
    for j in xrange(0,f5.shape[1]):
        for k in xrange(0,f5.shape[2]):
            images1.append(f5[i][j][k])

#list for column major of labels

labels=[]

for i in xrange(0,t5.shape[0]):
    for j in xrange(0,t5.shape[1]):
        labels.append(t5[i][j])

data_len=images1.__len__()
lbel_len=labels.__len__()
#coverting to int type

labels=[int(a) for a in labels]
print 'lenght of data:',data_len
print 'lenght of labels:',lbel_len
#print 'data shape:',f5.shape
#creating hdf5 file now
'''
with h5py.File('test.h5', 'w') as f:
    f['data'] = f5
    f['label'] = labels.astype(np.float32)

'''
fid1=h5py.File("train.h5",'w')
data=fid1.create_dataset("images1",(data_len,),'u1')
data[0:data_len]=images1
#creating label inside file
lbl=fid1.create_dataset("labels",(lbel_len,),'u1')
lbl[0:lbel_len]=labels
fid1.close()
