########################################################################################
# Tim Frawley
# July 2023
# Pipeline for locating regions of viable tissue and calculating cells counts for whole slide images in ZEISS CZI file format
# Set result and file directories to approriate locations
# For each image file in the input directory an excel file with all panels and their cells counts as well as a PNG map of the 
# viable and dense regions will be output to the results directory
# each image should take 20-30 minutes to complete depending on the size of the file and the amount of viable tiles to count cells
########################################################################################

import numpy as np
import pandas as pd
from aicspylibczi import CziFile
from PIL import Image,ImageFilter 
import torchvision.transforms as transforms
from helper_functions import *
import matplotlib.pyplot as plt
import os

results_dir = "M:\\Postbacs\\Tim\\Ki67_results\\Young Females 2"
IHC_directory = "M:\\Postdocs\\Former\\Saloni Agarwal\\working\\Data\\IHCimages\\Young Females"
image_list = os.listdir(IHC_directory)

#get model and make pytorch transform to normalize input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet256().to(device)
FILE = "W:\\python\\class256_6.pth"
model.load_state_dict(torch.load(FILE))
model.eval()

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

#assign weights for dense value calculation
dense_filter = np.array([1,2,1,2,4,2,1,2,1])/16
#panel size
step_size = 256
#Gauss filter settings
blur_radius = 3
dense_T = 5

#calculate cell counts and generate analysis maps for all files
for im in image_list:
    #Get CZI file and mouse information
    file_path = os.path.join(IHC_directory, im)
    mouse_ID = im.split('_')[0]
    czi = CziFile(file_path)
    num_scene = len(czi.get_dims_shape())
    
    for scene in range(num_scene):
        num_mosaic = czi.get_dims_shape()[scene]['M'][1]

        #make array for image map
        scene_x = czi.get_all_scene_bounding_boxes()[scene].x
        scene_y = czi.get_all_scene_bounding_boxes()[scene].y
        width = czi.get_all_scene_bounding_boxes()[scene].w / step_size
        height = czi.get_all_scene_bounding_boxes()[scene].h / step_size
        image_map = np.zeros((int(height), int(width), 3))
        boolean_map = np.zeros((int(height), int(width)))
        tile_bboxes = list(czi.get_all_mosaic_tile_bounding_boxes(S = scene).items())

        #create arrays to track output variables
        tile_ID = []
        viable = []
        dense = []
        dense_value = []
        posC_count = []
        allC_count = []
        x_cord = []
        y_cord = []

        #STEP 1 find viable panels using pytorch model
        for i in range(num_mosaic):
            mosaic_tile, mosaic_shape = czi.read_image(S = scene, M = i)
            tile_image = mosaic_tile[0,0,0,:,:,::-1]
            tile_x = (tile_bboxes[i][1].x - scene_x) / step_size
            tile_y = (tile_bboxes[i][1].y - scene_y) / step_size

            x_range = math.floor(len(tile_image[0])/step_size)
            y_range = math.floor(len(tile_image)/step_size)

            for x in range(0,x_range):
                for y in range(0,y_range):
                    x_start = x * (step_size)
                    y_start = y * (step_size)
                    x_end = x_start + step_size
                    y_end = y_start + step_size

                    boolean_x = int(tile_x + x)
                    boolean_y = int(tile_y + y)


                    #tile_name = "tile{}_{}{}_{}".format(i,x,y,step_size)
                    image = tile_image[y_start:y_end,x_start:x_end,:]
                    image_tensor = transform(image.copy())
                    image_tensor = image_tensor.to(device)

                    viability = model(image_tensor)
                    _, predicted = torch.max(viability, 1)

                    if (predicted == 1):
                        boolean_map[boolean_y, boolean_x] = True
            
        #use label to clean panel clusters <10 and fill gaps <6
        analysis_map = clean_map(boolean_map)
        count_map = np.zeros(analysis_map.shape)

        #STEP 2 use watershed method to count cells in vaible panels
        for i in range(num_mosaic):
            mosaic_tile, mosaic_shape = czi.read_image(S = scene, M = i)
            tile_image = mosaic_tile[0,0,0,:,:,::-1]
            tile_x = (tile_bboxes[i][1].x - scene_x) / step_size
            tile_y = (tile_bboxes[i][1].y - scene_y) / step_size


            x_range = math.floor(len(tile_image[0])/step_size)
            y_range = math.floor(len(tile_image)/step_size)

            for x in range(0,x_range):
                for y in range(0,y_range):
                    positive_cells = []
                    all_cells = []
                    x_start = x * (step_size)
                    y_start = y * (step_size)
                    x_end = x_start + step_size
                    y_end = y_start + step_size

                    boolean_x = int(tile_x + x)
                    boolean_y = int(tile_y + y)

                    tile_name = "tile{}_{}{}_{}".format(i,x,y,step_size)

                    if analysis_map[boolean_y, boolean_x] == True:
                        
                        image = tile_image[y_start:y_end,x_start:x_end,:]
                        image_fil = filter_image(image, blur_radius)
                        
                        positive_cells, all_cells = detect_cells(image_fil)
                        count_map[boolean_y, boolean_x] = len(positive_cells)

                    #update output lists
                    tile_ID.append(tile_name)
                    viable.append(analysis_map[boolean_y, boolean_x])
                    posC_count.append(len(positive_cells))
                    allC_count.append(len(all_cells))
                    x_cord.append(boolean_x + 1)
                    y_cord.append(boolean_y + 1)

        #STEP 3 use dense matrix to find dense regions of positive cells
        xlim = analysis_map.shape[1] - 1
        ylim = analysis_map.shape[0] - 1

        dense_map = np.zeros(count_map.shape)

        for x in range(1,xlim):
            for y in range(1,ylim):
                if(analysis_map[y,x]):
                    dense_map[y,x] = np.dot(count_map[y-1:y+2,x-1:x+2].flatten(), dense_filter)
        
        #Create analysis map for visualization of results
        for i in range(num_mosaic):
            mosaic_tile, mosaic_shape = czi.read_image(S = scene, M = i)
            tile_image = mosaic_tile[0,0,0,:,:,::-1]
            tile_x = (tile_bboxes[i][1].x - scene_x) / step_size
            tile_y = (tile_bboxes[i][1].y - scene_y) / step_size
            
            x_range = math.floor(len(tile_image[0])/step_size)
            y_range = math.floor(len(tile_image)/step_size)

            for x in range(0,x_range):
                for y in range(0,y_range):
                    is_dense = False

                    dense_x = int(tile_x + x)
                    dense_y = int(tile_y + y)

                    if (dense_map[dense_y, dense_x] > 0):
                        if (dense_map[dense_y, dense_x] > dense_T):
                            is_dense = True
                            image_map[dense_y,dense_x,0] = 255
                            image_map[dense_y,dense_x,1] = 255
                        else:
                            image_map[dense_y,dense_x,1] = 170
                            image_map[dense_y,dense_x,2] = 255
                    else:
                        image_map[dense_y,dense_x,0] = 160
                        image_map[dense_y,dense_x,2] = 50

                    dense.append(is_dense)
                    dense_value.append(dense_map[dense_y, dense_x])
                        


        #write out to results directory
        data = {'tile_name':tile_ID, 'viability':viable, 'dense_region':dense, 'dense_value':dense_value, 'positive_cell_count':posC_count, 'total_cell_count':allC_count, 'x_map_cord':x_cord, 'y_map_cord':y_cord}
        df = pd.DataFrame(data)
        df_path = os.path.join(results_dir, (mouse_ID + "_countResults.csv"))
        df.to_csv(df_path, index = False)
        image_path = os.path.join(results_dir, (mouse_ID + "_analysisMap.png"))
        cv2.imwrite(image_path, image_map[:,:,::-1])
        #plt.imshow(image_map.astype(int))
        #plt.show()
