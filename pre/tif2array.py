import os
from PIL import Image
import numpy as np

nimg = len([name for name in os.listdir('DBT_card_256') if os.path.isfile(os.path.join('DBT_card_256', name))])
imgarray = np.zeros((nimg, 256, 256), dtype=np.ndarray)

for (filename, i) in zip(os.listdir('DBT_card_256'), range(nimg)):
    if filename.endswith(".tif"):
        image = Image.open(os.path.join(os.path.join('DBT_card_256', filename)))
        data = np.asarray(image)
        imgarray[i] = data

