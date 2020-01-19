"""
Created on Tue Mar 12 01:27:39 2019

@author: soumi
"""
from ast import literal_eval
import numpy as np
from skimage.draw import line_aa
from skimage.transform import resize
import imageio

#get bound of the image
def get_bounds(strokes):
    min_x, max_x, min_y, max_y = (1000, 0, 1000, 0)

    for stroke in strokes:
        for x in stroke[0]:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
        
        for y in stroke[1]:
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    return (min_x, max_x, min_y, max_y)

# convert strokes to bitmap
def strokes_to_npy(strokes):

    # if no stroke- Convert to a black image of dimension 150 by 150
    if len(strokes)==0:
        dims=(150,150)
        img = np.zeros(dims, dtype=np.uint8)
    else: 
        min_x, max_x, min_y, max_y = get_bounds(strokes)
        # Add boundary of 20 pixels
        dims = (20 + max_x - min_x, 20 + max_y - min_y)
        img = np.zeros(dims, dtype=np.uint8)
        #fix according to binary
        abs_x = min_x - 10
        abs_y = min_y - 10
        for stroke in strokes:
            if len(stroke[0]) >1:
                prev_x = stroke[0][0]-abs_x
                prev_y = stroke[1][0]-abs_y         
                for i in range(len(stroke[0])-1):
                    dx = stroke[0][i+1]-abs_x
                    dy = stroke[1][i+1]-abs_y 
                    rr, cc, val = line_aa(prev_x, prev_y, dx, dy)
                    img[rr, cc] = (val * 255).astype(np.uint8)
 
                    prev_x = dx
                    prev_y = dy
    return img.T

# fit in square box
def reshape_to_square(img, size=512):
    img_resize = resize(img, (size, size))
    return img_resize

# crate a square image of dimension 100 by 100
def strokeToSquareImage(strokes, size=100):
    strokes = np.asarray(strokes)
    img = strokes_to_npy(strokes)
    img_resize = resize(img, (size, size))
    return img_resize


# convert the image and create a outfile.jpg in local directory
def getImage(content):
    img = strokeToSquareImage(strokes)    
    imageio.imwrite('outfile.jpg', 255-img)
#    scipy.misc.imsave('outfile.jpg', 255-img)

    return img



# provide full path of the file or save it locally
def readStrokes(fileName):
    f = open(fileName, 'r')
    x = f.read()
    f.close()
    strokes = literal_eval(str(x))
    return strokes

if __name__ == '__main__':
    strokes = readStrokes("temp")
    getImage(strokes)