### Pixe-wise classification to indentify various regions like tree, ground, sky, water etc. in outdoor images ###
### Machine learnig framework used = caffe with other libraries ###

### Total classes = 10 ###
The mapping is as follows:

{'building': 1,
 'dirt': 2,
 'foliage': 3,
 'grass': 4,
 'human': 5, 
 'pole': 6,
 'rails': 7,
 'road': 8,
 'sign': 9,
 'sky': 10}
0 means unlabelled and can be ignored. 
Note: Unfortunately complete data can't be open-sourced as it belongs to AirLab CMU, but I do have included a sample with permission


`channels.py`
Extracting the six channels of RGB image written in python

`makehdf5.py`
File to convert the stored data and label in mat to hdf5 format is

`solver.prototxt`
`trainer.prototxt`
`deploy.prototxt`
caffe files that contains a new network architecture for the current approach, still improving

`segment.py`
implements caffe model

`runtrain.sh`
run training using which creates log also

`A short report.pdf`
a brief idea of the work and things accomplished


