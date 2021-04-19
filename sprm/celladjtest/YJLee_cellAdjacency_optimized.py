# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:30:21 2021

@author: Young Je Lee
"""

import numpy as np
from skimage.morphology import binary_dilation
from skimage.morphology import dilation
from numpy import linalg as LA
import time
import numba as nb

@nb.jit
def CheckAdjacency(cellCoords_control,cellCoords_cur,thr):
    minValue=np.inf
    for k in range(len(cellCoords_cur[0])):
        subtracted=np.asarray([cellCoords_control[0]-cellCoords_cur[0,k],cellCoords_control[1]-cellCoords_cur[1,k]])
        distance=LA.norm(subtracted,axis=0)
        if np.any(distance<thr):
            return True 
    return False

def AdjacencyMatrix3(mask,cellEdgeList,thr=5,window=None):
    loc=mask.get_labels('cell_boundaries')
    #print('adjacency calculation begin')
    start=time.perf_counter()
    '''
    Initialization
    '''
    numCells=len(cellEdgeList)
    adjacencyMatrix=np.zeros((numCells,numCells))
    if window==None:
        delta=3
    else:
        delta=len(window)+3
    maskImg=mask.get_data()[0,0,loc,0,:,:]
    
    for i in range(1,numCells):
        maskImg=mask.get_data()[0,0,loc,0,:,:]
        xmin,xmax,ymin,ymax=np.min(cellEdgeList[i][0]),np.max(cellEdgeList[i][0]),np.min(cellEdgeList[i][1]),np.max(cellEdgeList[i][1])
        
        xmin=xmin-delta if xmin-delta>0 else 0
        xmax=xmax+delta+1 if xmax+delta+1<maskImg.shape[0] else maskImg.shape[0]
        ymin=ymin-delta if ymin-delta>0 else 0
        ymax = ymax+delta+1 if ymax+delta+1<maskImg.shape[1] else maskImg.shape[1]
        tempImg=np.zeros((xmax-xmin,ymax-ymin))
        
        tempImg[cellEdgeList[i][0]-xmin,cellEdgeList[i][1]-ymin]=1
        dilatedImg=binary_dilation(tempImg,selem=window)##
        mask_cropped=maskImg[xmin:xmax,ymin:ymax]
        #print('tempimg:',np.shape(tempImg),'cropped:',np.shape(mask_cropped))
        multipliedImg=mask_cropped*dilatedImg
        cellids=np.unique(multipliedImg)
        cellids=np.delete(cellids,cellids<=i)
        for j in cellids:
            check=CheckAdjacency(cellEdgeList[i],cellEdgeList[j],thr)
            if check==True:
                adjacencyMatrix[i,j]=1
                adjacencyMatrix[j,i]=1

    print('V3 start:',start)
    print('V3 end:',time.perf_counter())
    print('V3 total time:',time.perf_counter()-start)
    return adjacencyMatrix


