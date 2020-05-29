import os
import sys
import gdal




def gdal_crop(folder_path,ortho):
    input_filename = folder_path+'/'+ortho
    out_path = folder_path+'/cropped/'
    output_filename = 'tile_'
    
    tile_size_x = 2048
    tile_size_y = 2048
    
    ds = gdal.Open(input_filename)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    
    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
            os.system(com_string)
    
    return xsize,ysize,out_path