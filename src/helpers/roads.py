import os
import sys
import cv2
import gdal
import json
import math
import geojson
import numpy as np
from PIL import Image
from solaris import vector
from model import get_dilated_unet


folder_name=sys.argv[1]


def classify_roads(width,height,img_path,ortho):
    model = get_dilated_unet(input_shape=(2048,2048, 3), mode='cascade', filters=56,n_class=3)
    model.load_weights('models/roads_model_weights.hdf5')
    ds = gdal.Open('data/'+folder_name+'/'+ortho)
    geo_trans = ds.GetGeoTransform()
    
    classified_image=np.zeros(((height//2048+1)*2048,(width//2048+1)*2048,3))
    cv2.imwrite('data/'+folder_name+'/images/roads.jpg',classified_image)
    for img in os.listdir(img_path):
        classified_image=cv2.imread('data/'+folder_name+'/images/roads.jpg')
        if img.endswith('.tif'):
            im = Image.open(img_path+'/'+img)
            x_train = np.array(im,'f')
            x_train1 = model.predict(x_train)
            x=np.zeros((2048,2048,3))
            x[np.where(x_train1[0,:,:,0]>0.50)]=[255,0,0]
            x[np.where(x_train1[0,:,:,1]>0.50)]=[0,255,0]
            x[np.where(x_train1[0,:,:,2]>0.50)]=[0,0,0]
            stri=img[:-4].split('_')
            classified_image[int(stri[2]):int(stri[2])+2048,int(stri[1]):int(stri[1])+2048,:3]=x
            cv2.imwrite('data/'+folder_name+'/images/roads.jpg',classified_image)
            del classified_image
            os.remove(img_path+'/'+img)
    
    classified_image=cv2.imread('data/'+folder_name+'/images/roads.jpg')
    classified_image = classified_image[0:height,0:width]
    cv2.imwrite('data/'+folder_name+'/images/roads.jpg',classified_image)
    mask2poly = vector.mask.mask_to_poly_geojson(classified_image, channel_scaling=[1,-1,-1], bg_threshold=100, simplify=False)
    out = vector.polygon.georegister_px_df(mask2poly, affine_obj=geo_trans, crs='epsg:32618')
    out.to_file('data/'+folder_name+'/jsons/roads.json', driver='GeoJSON')

    mask2poly = vector.mask.mask_to_poly_geojson(classified_image, channel_scaling=[-1,1,-1], bg_threshold=100, simplify=False)
    out = vector.polygon.georegister_px_df(mask2poly, affine_obj=geo_trans, crs='epsg:32618')
    out.to_file('data/'+folder_name+'/jsons/footpaths.json', driver='GeoJSON')
    

