import os
import sys
import cv2
import gdal
import json
import math
import laspy
import geojson
import numpy as np
from PIL import Image
import multiprocessing
from solaris import vector
from model import get_dilated_unet
from shapely.geometry import Polygon,Point


Image.MAX_IMAGE_PIXELS = None

folder_name=sys.argv[1]

pi=3.14159265359
max_area=pi*2.5*2.5



class heightcalculation:
    def features(self,poly,point_3d):
        self.poly=poly
        self.point_3d=point_3d
        division=multiprocessing.cpu_count()
        index=list(range(0,len(poly)))
        print(len(poly))
        self.maximum_points=len(index)//division+1
        
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        result=pool.map(self.calc, range(division),chunksize=1)  # process data_inputs iterable with pool
        height=[]
        for divo in range(division):
            
            height=height+result[divo]
            print(divo,len(result[divo]))
        
        print(len(height),len(poly))
        
        final = {"type":"FeatureCollection", "features": []}

        for i in range(len(poly)):
                point=poly[i].centroid
                feature ={"type":"Feature","geometry":{"type":"Point","coordinates":[point.x,point.y]},"properties":{"id":i,"height":height[i],"radius":math.sqrt(poly[i].area/math.pi)}}
                final['features'].append(feature)
        
        
        with open('data/'+folder_name+'/jsons/trees.json', 'w') as outfile:
            json.dump(final, outfile)
        
        return
    
    def calc(self,div):
        

        if div==multiprocessing.cpu_count()-1:
            small_poly = self.poly[div*self.maximum_points:len(self.poly)]
            
        else:
            small_poly = self.poly[div*self.maximum_points:(div+1)*self.maximum_points]
        heights=[]
        c=1
        for i in small_poly:
            h=[]
            x,y=i.exterior.coords.xy
            maxx,minx,maxy,miny=max(x),min(x),max(y),min(y)
            point_3ds=self.point_3d[(self.point_3d[:,0]>= minx) & (self.point_3d[:,0]<=maxx)]                                                              #Filtering point cloud based on x coordinate
            point_3ds=point_3ds[(point_3ds[:,1]>= miny) & (point_3ds[:,1]<=maxy)]                                                           #Filtering point cloud based on y coordinate 
            for point in point_3ds:
                ppp_xy=Point(point[0],point[1])
                if ppp_xy.within(i):
                    h.append(point[2])
            try:
                heights.append(max(h)-min(h))
            except:
                heights.append(0)
            print(div,c,len(small_poly))
            c+=1
        
        return heights

def height_trees(dtm,dsm,point_cloud):
    
    dtm = gdal.Open('data/'+folder_name+'/'+dtm)
    band = dtm.GetRasterBand(1)
    dtm_elevation = band.ReadAsArray()

    dsm = gdal.Open('data/'+folder_name+'/'+dsm)
    band = dsm.GetRasterBand(1)
    dsm_elevation = band.ReadAsArray()
    geo_trans = dsm.GetGeoTransform()
    xsize = int(band.XSize)
    ysize = int(band.YSize)

    dtm_elevation=cv2.resize(dtm_elevation,(xsize,ysize))
    heights=np.subtract(dsm_elevation,dtm_elevation)

    img=cv2.imread('data/'+folder_name+'/images/trees.jpg')
    img[np.where(heights[:,:]==0)]=[0,0,0]
    cv2.imwrite('data/'+folder_name+'/images/trees.jpg',img)


    mask2poly = vector.mask.mask_to_poly_geojson(img, channel_scaling=[-1,1,1], bg_threshold=100, simplify=False)
    out = vector.polygon.georegister_px_df(mask2poly, affine_obj=geo_trans, crs='epsg:32644')
    out.to_file('data/'+folder_name+'/jsons/trees.json', driver='GeoJSON')

    trees=json.load(open('data/'+folder_name+'/jsons/trees.json'))
    poly=[]
    for i in range(len(trees['features'])):
        poly.append(trees['features'][i]['geometry']['coordinates'])


    new_poly=[]

    for i in len(range(poly)):
        polygon=Polygon(poly[i])
        if polygon.area > max_area:
            bounds=polygon.bounds
            minx, miny, maxx, maxy=bounds[0],bounds[1],bounds[2],bounds[3]
            loop_x=(maxx-minx)//5+1
            loop_y=(maxy-miny)//5+1

            bkup=miny
            dif_x=(maxx-minx)/loop_x
            dif_y=(maxy-miny)/loop_y

            while minx<maxx:
                while miny<maxy:
                    p=Polygon([(minx, miny), (minx+dif_x, miny), (minx+dif_x, miny+dif_y),(minx, miny+dif_y)])
                    p=p.intersection(polygon)
                    if p.area> pi/4:
                        new_poly.append(p)
                    miny=miny+dif_y
                minx=minx+dif_x
                miny=bkup
         
        else:
            new_poly.append(polygon)

    poly=new_poly
    infile=laspy.file.File('data/'+folder_name+'/'+point_cloud,mode="rw")
    point_3d=np.vstack((infile.x,infile.y,infile.z)).T
    fe=heightcalculation()
    fe.features(poly,point_3d)



def classify_trees(width,height,img_path):
    model = get_dilated_unet(input_shape=(2048,2048, 3), mode='cascade', filters=56,n_class=3)
    model.load_weights('models/trees_model_weights.hdf5')

    
    classified_image=np.zeros(((height//2048+1)*2048,(width//2048+1)*2048,3))
    cv2.imwrite('data/'+folder_name+'/images/trees.jpg',classified_image)
    for img in os.listdir(img_path):
        classified_image=cv2.imread('data/'+folder_name+'/images/trees.jpg')
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
            cv2.imwrite('data/'+folder_name+'/images/trees.jpg',classified_image)
            del classified_image
            os.remove(img_path+'/'+img)
    
    classified_image=cv2.imread('data/'+folder_name+'/images/trees.jpg')
    classified_image = classified_image[0:height,0:width]
    cv2.imwrite('data/'+folder_name+'/images/trees.jpg',classified_image)
    
    