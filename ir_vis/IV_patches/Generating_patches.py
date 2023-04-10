import os
import random
import numpy as np
from scipy.misc import imread, imsave, imresize

PATCH_SIZE = 128;
PATCH_STRIDE = 12;

#Original image pairs
num_image = 50;

#Put the original images in the "IVIF_source/IR" & "IVIF_source/VIS" directories.
#The augmented data will be put in the "./IR" and "./VIS" directories.

prepath = "./"
patchesIR = [];
patchesVIS = [];
picidx = 0;
for idx in range(0 + 1, num_image + 1):
    print("Decomposing " + str(idx) + "-th images...");
    imageIR = imread(prepath + '/IVIF_source/IR/' + str(idx) + '.jpg', mode='L');
    imageVIS = imread(prepath + '/IVIF_source/VIS/' + str(idx) + '.jpg', mode='L');
    h = imageIR.shape[0];
    w = imageIR.shape[1];
    for i in range(0, h - PATCH_SIZE + 1, PATCH_STRIDE):
        for j in range(0, w - PATCH_SIZE + 1, PATCH_STRIDE):
            picidx += 1;
            patchImageIR = imageIR[i:i + PATCH_SIZE, j:j + PATCH_SIZE];
            patchImageVIS = imageVIS[i:i + PATCH_SIZE, j:j + PATCH_SIZE];
            imsave('./IR/' + str(picidx) + '.png', patchImageIR);
            imsave('./VIS/' + str(picidx) + '.png', patchImageVIS);
    print(picidx);
    
