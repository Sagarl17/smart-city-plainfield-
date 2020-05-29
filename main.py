import os
import sys

from src.helpers import logger
from src.helpers.gdal_crop import gdal_crop
from src.helpers.trees import classify_trees,height_trees
from src.helpers.roads import classify_roads
from src.helpers.buildings import classify_buildings,heights_buildings

folder_name=sys.argv[1]

logger = logger.setup_custom_logger('myapp')    

logger.info('Sorting files started')

folder_path='data/'+folder_name
folder_contents= os.listdir(folder_path)

for file in folder_contents:
    if file.endswith('tif'):
        if 'dtm' or 'dsm' not in file:
            ortho=file
        elif 'dtm' in file:
            dtm=file
        elif 'dsm' in file:
            dsm=file
    elif file.endswith('.las'):
        point_cloud=file

logger.info('Sorting files ended')

logger.info('Cropping orthomosiac started')

width,height,cropped_image_path=gdal_crop(folder_path,ortho)

logger.info('Cropping orthomosiac finished')

if 'trees' in sys.argv:

    logger.info('Extracting trees image started')
    classify_trees(width,height,cropped_image_path)
    logger.info('Extracting trees image finished')


    logger.info('Extracting trees json started')
    height_trees(dtm,dsm,point_cloud)
    logger.info('Extracting trees json finished')


if 'buildings' in sys.argv:
    logger.info('Extracting buildings image started')
    classify_buildings(width,height,cropped_image_path,ortho)
    logger.info('Extracting buildings image finished')

    logger.info('Extracting builidngs json started')
    heights_buildings(point_cloud)
    logger.info('Extracting buildings json finished')


if 'roads' in sys.argv:
    logger.info('Extracting roads image and json started')
    classify_roads(width,height,cropped_image_path,ortho)
    logger.info('Extracting roads image and json finished')


