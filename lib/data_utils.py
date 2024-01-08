import numpy as np
from sklearn import utils as skutils
from lib.rng import np_rng, py_rng

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], str):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n // size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])

def convert_img(img):
    H,W,C=img.shape
    con_img = np.zeros((C,H,W),dtype=img.dtype)
    for i in range(C):
        con_img[i,:,:] = img[:,:,i]
    return con_img

def convert_img_back(img):
    C,H,W=img.shape
    con_img = np.zeros((H,W,C),dtype=img.dtype)
    for i in range(C):
        con_img[:,:,i] = img[i,:,:]
    return con_img

def processing_img(img, center=True, scale=True, convert=True):
    img = np.array(img)
    img = np.cast['float32'](img)
    if convert is True:
        img = convert_img(img)
    if  center and scale:
        img[:] -= 127.5
        img[:] /= 127.5
    elif center:
        img[:] -= 127.5
    elif scale:
        img[:] /= 255.
    return img

def ImgRescale(img,center=True,scale=True, convert_back=False):
    img = np.array(img)
    img = np.cast['float32'](img)
    if convert_back is True:
        img = convert_img_back(img)
    if center and scale:
        img = ((img+1) / 2 * 255).astype(np.uint8)
    elif center:
	    img = (img + 127.5).astype(np.uint8)
    elif scale:
	    img = (img * 255).astype(np.uint8)
    return img