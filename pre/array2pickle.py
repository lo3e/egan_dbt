import pickle
from tif2array import imgarray

filename = 'imgarray2pickle'
pimgs = open(filename, 'wb')

pickle.dump(imgarray, pimgs)
pimgs.close()
