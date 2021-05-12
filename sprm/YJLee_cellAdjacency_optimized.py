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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
@nb.jit
def CheckAdjacency(cellCoords_control,cellCoords_cur,thr):
    minValue=np.inf
    for k in range(len(cellCoords_cur[0])):
        subtracted=np.asarray([cellCoords_control[0]-cellCoords_cur[0,k],cellCoords_control[1]-cellCoords_cur[1,k]])
        distance=LA.norm(subtracted,axis=0)
        if np.min(distance)<thr:
            return np.min(distance) 
    return 0

def AdjacencyMatrix(mask,cellEdgeList,baseoutputfilename,output_dir,thr=3,window=None):
    loc=mask.get_labels('cell_boundaries')
    #print('adjacency calculation begin')
    start=time.perf_counter()
    '''
    Initialization
    '''
    numCells=len(cellEdgeList)
    adjacencyMatrix=np.zeros((numCells,numCells))
    cellGraph=dict()
    
    cell_center = np.zeros((numCells, 2))

    for i in range(numCells):
        m = (np.sum(cellEdgeList[i], axis=1) / cellEdgeList[i].shape[1]).astype(int)
        cell_center[i, 0] = m[0]
        cell_center[i, 1] = m[1]
    
    
    
    
    for i in range(1,numCells):
        cellGraph[i]=set()
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
            minDist=CheckAdjacency(cellEdgeList[i],cellEdgeList[j],thr)
            if minDist!=0 and minDist<thr:
                adjacencyMatrix[i,j]=minDist
                adjacencyMatrix[j,i]=minDist
                cellGraph[i].add(j)
                cellGraph[j].add(i)

    AdjacencyMatrix2Graph(adjacencyMatrix,cell_center,cellGraph,output_dir/(baseoutputfilename+'_AdjacencyGraph.png'),thr)
    #Remove background
    adjacencyMatrix=np.delete(adjacencyMatrix,0,axis=0)
    adjacencyMatrix=np.delete(adjacencyMatrix,0,axis=1)
    adjacencyMatrix_df=pd.DataFrame(adjacencyMatrix,index=np.arange(1,len(adjacencyMatrix)+1),columns=np.arange(1,len(adjacencyMatrix)+1))
    adjacencyMatrix_df.to_csv(output_dir/(baseoutputfilename+'_AdjacencyMatrix.csv'))
    #return adjacencyMatrix
#plt.axline((x1,y1),(x2,y2))
def AdjacencyMatrix2Graph(adjacencyMatrix,cell_center,cellGraph,name,thr):
    fig, ax = plt.subplots(figsize=(17.0, 17.0))
    plt.plot(cell_center[:,0],cell_center[:,1],'o')
    plt.title('Cell Adjacency Graph, distance <'+str(thr))
    for i in range(1,len(cell_center)):
        line2draw=cell_center[list(cellGraph[i])]
        lines=[[cell_center[i],r] for r in line2draw]
        line=mc.LineCollection(lines,colors=[(1, 0, 0, 1)])
        ax.add_collection(line)
        for j in range(len(list(cellGraph[i]))):
            gap=(cell_center[i]-line2draw[j])/np.sqrt((cell_center[i]-line2draw[j])*(cell_center[i]-line2draw[j]))            
            ax.text(cell_center[i][0]+gap[0],cell_center[i][1]+gap[1],'%.1f' %adjacencyMatrix[i][list(cellGraph[i])[j]],ha='center',va='center')
    plt.savefig(name)
