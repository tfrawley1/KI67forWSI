import numpy as np
import pandas as pd
from aicspylibczi import CziFile
from PIL import Image,ImageFilter 
import torchvision.transforms as transforms
from helper_functions import *
import matplotlib.pyplot as plt
import os

results_dir = "M:\\Postbacs\\Tim\\Ki67_results"
IHC_directory = "M:\\Postdocs\\Former\\Saloni Agarwal\\working\\Data\\IHCimages\\Young Females"
image_list = os.listdir(IHC_directory)

#get the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet256().to(device)
FILE = "W:\\python\\net256_4.pth"
model.load_state_dict(torch.load(FILE))
model.eval()

for im in image_list[0:1]:
    file_path = os.path.join(IHC_directory, im)
    mouse_ID = im.split('_')[0]
    czi = CziFile(file_path)
    num_mosaic = czi.get_dims_shape()[0]['M'][1]

    #make array for image map
    scene_x = czi.get_all_scene_bounding_boxes()[0].x
    scene_y = czi.get_all_scene_bounding_boxes()[0].y
    width = czi.get_all_scene_bounding_boxes()[0].w * 0.1
    height = czi.get_all_scene_bounding_boxes()[0].h * 0.1
    image_map = np.zeros((int(height), int(width), 3))
    tile_bboxes = list(czi.get_all_mosaic_tile_bounding_boxes().items())

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    step_size = 256
    dense_T = 3
    tile_ID = []
    viable = []
    dense = []
    posC_count = []
    allC_count = []
    x_cord = []
    y_cord = []
    blur_radius = 3


    for i in range(num_mosaic):
        mosaic_tile, mosaic_shape = czi.read_image(M = i)
        tile_image = mosaic_tile[0,0,0,:,:,::-1]
        tile_x = 0.1 * (tile_bboxes[i][1].x - scene_x)
        tile_y = 0.1 * (tile_bboxes[i][1].y - scene_y)

        x_range = math.floor(len(tile_image[0])/step_size)
        y_range = math.floor(len(tile_image)/step_size)

        for x in range(0,x_range):
            for y in range(0,y_range):
                positive_cells = []
                all_cells = []
                is_dense = False
                x_start = x * (step_size)
                y_start = y * (step_size)
                x_end = x_start + step_size
                y_end = y_start + step_size

                image_x_start = int(tile_x + 0.1 * x_start)
                image_y_start = int(tile_y + 0.1 * y_start)
                image_x_end = int(tile_x + 0.1 * x_end)
                image_y_end = int(tile_y + 0.1 * y_end)


                tile_name = "tile{}_{}{}_{}".format(i,x,y,step_size)
                image = tile_image[y_start:y_end,x_start:x_end,:]
                image_tensor = transform(image.copy())

                viability = model(image_tensor)
                _, predicted = torch.max(viability, 1)

                if (predicted == 1):
                    im = Image.fromarray(np.uint8(image.copy()))
                    im = im.filter(ImageFilter.GaussianBlur(radius = blur_radius))
                    image_fil = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
                    image_fil = np.uint8(image_fil)
                    positive_cells, all_cells = detect_cells(image_fil)
                    if(len(positive_cells)>=dense_T):
                        is_dense = True
                        image_map[image_y_start:image_y_end,image_x_start:image_x_end,0] = 255
                        image_map[image_y_start:image_y_end,image_x_start:image_x_end,1] = 255
                    else:
                        image_map[image_y_start:image_y_end,image_x_start:image_x_end,1] = 170
                        image_map[image_y_start:image_y_end,image_x_start:image_x_end,2] = 255
                else:
                    image_map[image_y_start:image_y_end,image_x_start:image_x_end,0] = 160
                    image_map[image_y_start:image_y_end,image_x_start:image_x_end,2] = 50

                tile_ID.append(tile_name)
                viable.append(predicted)
                dense.append(is_dense)
                posC_count.append(len(positive_cells))
                allC_count.append(len(all_cells))
                x_cord.append(image_x_start)
                y_cord.append(image_y_start)

    data = {'tile_name':tile_ID, 'viability':viable, 'dense_region':dense, 'positive_cell_count':posC_count, 'total_cell_count':allC_count, 'x_map_cord':x_cord, 'y_map_cord':y_cord}
    df = pd.DataFrame(data)
    df_path = os.path.join(results_dir, (mouse_ID + "_countResults.csv"))
    df.to_csv(df_path, index = False)
    image_path = os.path.join(results_dir, (mouse_ID + "_analysisMap.png"))
    cv2.imwrite(image_path, image_map[:,:,::-1])
    #plt.imshow(image_map.astype(int))
    #plt.show()
