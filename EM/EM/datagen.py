import random
import csv

def create_data(kind,N,filePath):
    distribute = []
    x1,x2 = random.randint(-100,0),random.randint(-100,100)
    y1,y2 = random.randint(-100,0),random.randint(-100,100)
    for j in range(N):
        a = random.randint(min(x1,x2),max(x1,x2))
        b = random.randint(min(y1,y2),max(y1,y2))
        distribute.append([a,b])

    x1,x2 = random.randint(0,100),random.randint(-100,100)
    y1,y2 = random.randint(0,100),random.randint(-100,100)
    for j in range(N):
        a = random.randint(min(x1,x2),max(x1,x2))
        b = random.randint(min(y1,y2),max(y1,y2))
        distribute.append([a,b])

    handle = open(filePath,"w")
    writer = csv.writer(handle)
    distribute.insert(0,[kind,N])
    writer.writerows(distribute)
    handle.close()

import numpy as np

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

if __name__ == "__main__":
    create_data(2,100,"./test.csv")

