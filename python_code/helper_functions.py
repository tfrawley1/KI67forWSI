########################################################################################
# Tim Frawley
# July 2023
# 
# Functions used by Ki67 pipeline and other programs related to Ki67 index project
########################################################################################

import numpy as np
import math
from aicspylibczi import CziFile
import os
import cv2
from skimage.color import rgb2hed, rgb2gray, hed2rgb
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from PIL import Image,ImageFilter 
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pandas as pd



#function to generate PNG tiles of step_size from mosaic tile
def make_jpg_tiles(m, tile_image, tile_directory_path, step_size = 256):
    if(os.path.exists(tile_directory_path) == False) :
        os.mkdir(tile_directory_path)
    x_range = math.floor(len(tile_image[0])/step_size)
    y_range = math.floor(len(tile_image)/step_size)

    for x in range(0,x_range):
        for y in range(0,y_range):
            x_start = x * (step_size)
            y_start = y * (step_size)
            x_end = x_start + step_size
            y_end = y_start + step_size

            tile_name = "tile{}_{}{}_{}.png".format(m,x,y,step_size)
            image = tile_image[y_start:y_end,x_start:x_end,:]
            tile_path = os.path.join(tile_directory_path, tile_name)
            cv2.imwrite(tile_path, image)

#generate PNG tiels from CZI file
def czi_to_tiles(czi_file, mouse_ID, parent_directory_path):
    tile_path = os.path.join(parent_directory_path,"Tiles",mouse_ID)

    num_mosaic = czi_file.get_dims_shape()[0]['M'][1]
    for i in range(num_mosaic):
        mosaic_tile, mosaic_shape = czi_file.read_image(M = i)
        mosaic_image = mosaic_tile[0,0,0]
        #not needed if writing to png
        #mosaic_image = np.stack((mosaic_image[:,:,2],mosaic_image[:,:,1],mosaic_image[:,:,0]), axis = 2)

        if(len(mosaic_image)%400 == 0): 
            step_size = 400
        else: 
            step_size = 256
        make_jpg_tiles(i, mosaic_image, tile_path, step_size)

#detect positive and total cell counts from stained tissue image
#input: image as np array with 3 color channels
#output: list of positive and all cell locations in image
def detect_cells(image):
    image_hed = rgb2hed(image.copy())
    mask_positive = np.zeros(image_hed[:,:,2].shape)
    mask_positive[image_hed[:,:,2] >= 0.04] = 255

    positive_cells = ws_cells(mask_positive)
    positive_cells_clean = clean_overlapping_cells(positive_cells)

    mask_all = get_all_mask(image_hed)
    all_cells = ws_cells(mask_all)
    all_cells_clean = clean_overlapping_cells(all_cells)
    return(positive_cells_clean, all_cells_clean)

# watershed method
# input: np array of BGR image
# output: array of cell center locations and radius size
def ws_cells(image):
    cells = []
    mask2 = np.int32(image)
    distance = ndimage.distance_transform_edt(mask2)
    seeds = peak_local_max(distance, min_distance = 20, labels = mask2)
    markers = np.zeros(mask2.shape)
    markers[tuple(seeds.T)] = True
    markers,_ = ndimage.label(markers, structure = np.ones((3,3)))
    labels = watershed(-distance, markers, mask=mask2)

    #taken from Saloni
    for label in np.unique(labels):
        if label == 0:
            continue
        temp_mask = np.zeros(labels.shape, dtype = 'uint8')
        temp_mask[labels[:,:]==label] = 1
        contours,_ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center = max(contours, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(center)
        if (r > 3) :
            cells.append([x,y,r])
    return(np.array(cells))

# function to remove detected cell locations that are too close as set by d_threshold
def clean_overlapping_cells(cells, d_threshold = 25):
    num_cells = len(cells)
    overlaps = np.zeros(num_cells)

    for i in range(0, num_cells):
        if(overlaps[i] == 1):
            continue

        x = cells[i][0]
        y = cells[i][1]

        for j in range(i+1, num_cells):
            distance = math.sqrt((cells[j][0] - x)**2 + (cells[j][1] - y)**2)
            if (distance <= (d_threshold)):
                overlaps[j] = 1
    
    clean_cells = cells[overlaps==0]
    return(clean_cells)

# function to generate image showing detected cell locations in image
def make_wsImage(image, cells, clean_cells):
    temp = image.copy()

    for p in cells:
        x,y=p[0],p[1]
        cv2.circle(temp, (int(x), int(y)), 2, (255,0,0), 2)

    for p in clean_cells:
        x,y=p[0],p[1]
        cv2.circle(temp, (int(x), int(y)), 25, (0,0,255), 2)

    for p in clean_cells:
        x,y=p[0],p[1]
        cv2.circle(temp, (int(x), int(y)), 2, (0,0,255), 2)

    return(temp)

# function to genearte binary mask for detecting all cells
def get_all_mask(ihc_hed):
    #Taken from Saloni
    null = np.zeros_like(ihc_hed[:, :, 0])
    for x in range(ihc_hed.shape[0]):
        for y in range(ihc_hed.shape[1]):
            ihc_hed[x,y,0] = ihc_hed[x,y,0]+ihc_hed[x,y,1]
            if ihc_hed[x,y,2]>0.18:
                ihc_hed[x,y,0] = 0.11

    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_h = np.uint8(ihc_h*255)
    gray = rgb2gray(ihc_h)
    mask = np.int64(np.zeros(ihc_hed[:,:,2].shape))
    for x in range(ihc_hed.shape[0]):
        for y in range(ihc_hed.shape[1]):
            if gray[x,y]<0.65 or ihc_hed[x,y,2] >= 0.05:
                mask[x,y]=255

    return mask

# function to use ndi label to count clusters of viable panels
# clusters of less than 10 panels are removed and gaps < 6 are filled in
# note: ndi.label takes parameter "structure" which determines if panels on diagonal should be counted
def clean_map(boolean_map):
    markers,_ = ndimage.label(boolean_map, structure = np.ones((3,3)))

    for label in np.unique(markers):
         if len(markers[markers==label]) < 10:
            markers[markers==label] = 0

    inverse_boolean_map = (markers == 0)

    markers_inverse,_ = ndimage.label(inverse_boolean_map)

    for label in np.unique(markers_inverse):
        if len(markers_inverse[markers_inverse==label]) < 6:
            markers_inverse[markers_inverse==label] = 0

    final_map = (markers_inverse == 0)
    return(final_map)

#function to apply gauss filter to image
def filter_image(image, blur_radius):
    im = Image.fromarray(np.uint8(image.copy()))
    im = im.filter(ImageFilter.GaussianBlur(radius = blur_radius))
    image_fil = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
    return(np.uint8(image_fil))

#Conv Net for 400px and 256px iamges
class ConvNet400(nn.Module):
    def __init__(self):
        super(ConvNet400, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=3)
        self.conv2 = nn.Conv2d(6, 9, 5, stride=3)
        self.conv3 = nn.Conv2d(9, 12, 5, stride=3)
        self.fc1 = nn.Linear(12*13*13, 1200)
        self.fc2 = nn.Linear(1200, 240)
        self.fc3 = nn.Linear(240, 60)
        self.fc4 = nn.Linear(60, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 12*13*13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class ConvNet256(nn.Module):
    def __init__(self):
        super(ConvNet256, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=3)
        self.conv2 = nn.Conv2d(6, 12, 5, stride=3)
        self.conv3 = nn.Conv2d(12, 24, 5, stride=3)
        self.fc1 = nn.Linear(24*8*8, 480)
        self.fc2 = nn.Linear(480, 120)
        self.fc3 = nn.Linear(120, 30)
        self.fc4 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 24*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x