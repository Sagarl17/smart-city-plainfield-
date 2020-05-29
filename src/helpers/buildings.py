import os
import sys
import cv2
import gdal
import time
import laspy
import geojson
import skimage
import shapely 
import numpy as np
from PIL import Image
import multiprocessing
from solaris import vector
from fastai.vision import *
from fastai.callbacks import *


Image.MAX_IMAGE_PIXELS = None

folder_name=sys.argv[1]

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True, convert_mode='RGB')
    
class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom



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
                geojson_out=geojson.Feature(geometry=poly[i])
                feature ={"type":"Feature","geometry":{"type":"Polygon","coordinates":[]},"properties":{"id":i,"height":height[i]}}
                feature['geometry']=geojson_out.geometry
                final['features'].append(feature)
        
        
        with open('data/'+folder_name+'/jsons/buildings.json', 'w') as outfile:
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



def dice_loss(input, target):
    smooth = 1.
    input = torch.sigmoid(input)
    iflat = input.contiguous().view(-1).float()
    tflat = target.contiguous().view(-1).float()
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / ((iflat + tflat).sum() +smooth))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean': return F_loss.mean()
        elif self.reduction == 'sum': return F_loss.sum()
        else: return F_loss

class DiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        loss = dice_loss(input, target)
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

class MultiChComboLoss(nn.Module):
    def __init__(self, reduction='mean', loss_funcs=[FocalLoss(),DiceLoss()], loss_wts = [1,1], ch_wts=[1,1,1]):
        super().__init__()
        self.reduction = reduction
        self.ch_wts = ch_wts
        self.loss_wts = loss_wts
        self.loss_funcs = loss_funcs 
        
    def forward(self, output, target):
        for loss_func in self.loss_funcs: loss_func.reduction = self.reduction # need to change reduction on fwd pass for loss calc in learn.get_preds(with_loss=True)
        loss = 0
        channels = output.shape[1]
        assert len(self.ch_wts) == channels
        assert len(self.loss_wts) == len(self.loss_funcs)
        for ch_wt,c in zip(self.ch_wts,range(channels)):
            ch_loss=0
            for loss_wt, loss_func in zip(self.loss_wts,self.loss_funcs): 
                ch_loss+=loss_wt*loss_func(output[:,c,None], target[:,c,None])
            loss+=ch_wt*(ch_loss)
        return loss/sum(self.ch_wts)

def acc_thresh_multich(input:Tensor, target:Tensor, thresh:float=0.5, sigmoid:bool=True, one_ch:int=None)->Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: input = input.sigmoid()
    n = input.shape[0]
    
    if one_ch is not None:
        input = input[:,one_ch,None]
        target = target[:,one_ch,None]
    
    input = input.view(n,-1)
    target = target.view(n,-1)
    return ((input>thresh)==target.byte()).float().mean()

def dice_multich(input:Tensor, targs:Tensor, iou:bool=False, one_ch:int=None)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
#     pdb.set_trace()
    n = targs.shape[0]
    input = input.sigmoid()
    
    if one_ch is not None:
        input = input[:,one_ch,None]
        targs = targs[:,one_ch,None]
    
    input = (input>0.5).view(n,-1).float()
    targs = targs.view(n,-1).float()

    intersect = (input * targs).sum().float()
    union = (input+targs).sum().float()
    if not iou: return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())
    else: return intersect / (union-intersect+1.0)



def heights_buildings(point_cloud):
    poly=[]

    input_file = open ('data/'+folder_name+'/jsons/buildings.json')
    poly1_geojson = json.load(input_file)
    for i in range(len(poly1_geojson['features'])):
        poly.append(shapely.geometry.asShape(poly1_geojson['features'][i]['geometry']))
    infile=laspy.file.File('data/'+folder_name+'/'+point_cloud,mode="rw")
    point_3d=np.vstack((infile.x,infile.y,infile.z)).T
    fe=heightcalculation()
    fe.features(poly,point_3d)



def get_pred(learner, tile):
#     pdb.set_trace()
    t_img = Image(pil2tensor(tile[:,:,:3],np.float32).div_(255))
    outputs = learner.predict(t_img)
    im = image2np(outputs[2].sigmoid())
    im = (im*255).astype('uint8')
    return im

inference_learner = load_learner(path='models/', file='znz001trn-focaldice.pkl')

def classify_buildings(width,height,img_path,ortho):
    classified_image=np.zeros(((height//2048+1)*2048,(width//2048+1)*2048,3))
    cv2.imwrite('data/'+folder_name+'/images/buildings.jpg',classified_image)
    ds = gdal.Open('data/'+folder_name+'/'+ortho)
    geo_trans = ds.GetGeoTransform()
    for img in os.listdir(img_path):
        classified_image=cv2.imread('data/'+folder_name+'/images/buildings.jpg')
        if img.endswith('.tif'):
            test_tile = skimage.io.imread(img_path+'/'+img)
            x = get_pred(inference_learner, test_tile)
            x=cv2.resize(x, (2048, 2048))
            stri=img[:-4].split('_')
            classified_image[int(stri[2]):int(stri[2])+2048,int(stri[1]):int(stri[1])+2048,:3]=x
            cv2.imwrite('data/'+folder_name+'/images/buildings.jpg',classified_image)
            del classified_image
            os.remove(img_path+'/'+img)
    
    classified_image=cv2.imread('data/'+folder_name+'/images/buildings.jpg')
    classified_image = classified_image[0:height,0:width]
    cv2.imwrite('data/'+folder_name+'/images/buildings.jpg',classified_image)
    classified_image=cv2.imread('data/'+folder_name+'/images/buildings.jpg')
    mask2poly = vector.mask.mask_to_poly_geojson(classified_image, channel_scaling=[1,-1,-1], bg_threshold=100, simplify=True,tolerance=3)
    out = vector.polygon.georegister_px_df(mask2poly, affine_obj=geo_trans, crs='epsg:32618')
    out.to_file('data/'+folder_name+'/jsons/buildings.json', driver='GeoJSON')
    del classified_image
    
