from aicspylibczi import CziFile
from helper_functions import *
import os

IHC_directory = "W:\\python\\IHC_images"
image_list = os.listdir(IHC_directory)

index = 0
czi1_path = os.path.join(IHC_directory,image_list[index])
czi1_ID = image_list[index].split('_')[0]
czi1 = CziFile(czi1_path)

czi_to_tiles(czi1, czi1_ID, "W:\\python")


