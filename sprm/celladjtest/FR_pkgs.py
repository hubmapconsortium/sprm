# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:28:29 2021

@author: young
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import skimage.io as ioo
from sklearn.cluster import KMeans

def loadCSV(filename):
    header=pd.read_csv(filename,index_col=0,nrows=0).columns.tolist()
    matrix = np.genfromtxt(filename,delimiter=',',skip_header=True)[:,1:]
    return matrix,header


def ScatterPlot(xinput,yinput,titleName,savePlot=True):
    plt.figure()
    plt.scatter(xinput,yinput)
    plt.title(titleName)
    if savePlot==True:
        name=titleName.split(':')
        name=' '.join(name)
        plt.savefig('ScatterPlot/'+name)


 
def PrincipleComponentAnalysis_withPlot(matrix,figureName='PCA'):
    numComponents=1
    #Initialize
    cumulative_percentile=[]
    matrix=scale(matrix)
    while True:
        pca=PCA(n_components=numComponents)
        pca.fit(matrix)
        pcaResult=pca.explained_variance_ratio_
        cumulative_percentile=np.append(cumulative_percentile,np.sum(pcaResult))
        if cumulative_percentile[-1]>=1.0:
            break
        numComponents+=1
    
    print("Number of PC to explain 100%: ",numComponents)
    print('explained variance for ',figureName,' : ',str(pca.explained_variance_ratio_))
    #Plot
    plt.figure()
    plt.title(figureName+'_Cumulative_Percentile')
    plt.plot(cumulative_percentile)
    plt.ylim((0,1))
    for i, v in enumerate(cumulative_percentile):
        plt.text(i, v, "%.3f" %v, ha="center")
    
    return pca.transform(matrix)

def Texture_split_radii(matrix_texture,header_texture,numCells,numRadii=3,numChan=18,numTextureFeature=6):
    if numRadii==1:
        matrix1,header1=[],[]
        for i in range(0,len(header_texture)-1,numChan):
            for j in range(numTextureFeature):
                matrix1=np.append(matrix1,matrix_texture[:,i+j])
                header1.append(header_texture[i+j])
        matrix1=np.reshape(matrix1,(numCells,-1),order='F')
        return matrix1,header1
    
    matrix1,header1=[],[]
    matrix2,header2=[],[]
    matrix3,header3=[],[]
    for i in range(0,len(header_texture),numChan):
        for j in range(numTextureFeature):
            matrix1=np.append(matrix1,matrix_texture[:,i+j])
            header1.append(header_texture[i+j])
            matrix2=np.append(matrix2,matrix_texture[:,i+j+numTextureFeature])
            header2.append(header_texture[i+j+numTextureFeature])
            matrix3=np.append(matrix3,matrix_texture[:,i+j+numTextureFeature*2])
            header3.append(header_texture[i+j+numTextureFeature*2])
    matrix1=np.reshape(matrix1,(numCells,-1),order='F')
    matrix2=np.reshape(matrix2,(numCells,-1),order='F')
    matrix3=np.reshape(matrix3,(numCells,-1),order='F')

    return matrix1,header1,matrix2,header2,matrix3,header3

def KMeanWithVisual(matrix,numClusters,groundTruth,additional,tsne=False):
    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(matrix)
    cluster_labels=kmeans.labels_
    cluster_labels=cluster_labels+1
    curCellImg=np.zeros((len(groundTruth[0]),len(groundTruth[0][0])))
    for idx in range(len(groundTruth[1])):
        curROI = groundTruth[2][0][groundTruth[1][idx]]
        for roiIdx in range(len(curROI[0])):
            curCellImg[curROI[0][roiIdx],curROI[1][roiIdx]]=cluster_labels[idx]
    #ioo.imshow(curCellImg)
    plt.Figure()
    plt.imshow(curCellImg)
        
    if tsne==False:
        plt.title("KMeans for "+str(numClusters)+"Clusters")
        name=additional+str(numClusters)+"Clusters.png"
        plt.savefig('KMeans/'+name)
    elif tsne==True:
        plt.title(additional+"then KMeans for "+str(numClusters)+" Clusters")
        name=additional+str(numClusters)+"Clusters.png"
        plt.savefig('tsne/'+name)        
    return cluster_labels


'''
tsne video

from mpl_toolkits import mplot3d
from celluloid import Camera

numComp=2
perplexity=3
from sklearn.manifold import TSNE
matrix_all=np.asarray(matrix_all,dtype=float)


fig=plt.figure(figsize = (10,10))
axes=plt.gca()
axes.set_xlim([-1000,1000])
axes.set_ylim([-1000,1000])
camera = Camera(fig)

for i in range(5):
    tsne=TSNE(n_components=numComp,init='random',perplexity=perplexity+i*10)
    tsne_all=tsne.fit_transform(matrix_all)
    
    
    
    plt.title("tsne with"+str(i))
    plt.scatter(tsne_all[:,0],tsne_all[:,1],alpha=0.25)
    plt.xlabel('tsne1')
    plt.ylabel('tsne2')
    plt.pause(1)
    camera.snap()
    plt.clf()
    plt.show()

animation=camera.animate(interval=100,blit=True)
animation.save("tsne.gif")

'''























        
def glcm(im, mask, bestz, output_dir, cell_total, filename, options, angle, distances,ROI_coords):
    '''
    By: Young Je Lee
    '''
    texture_all=[]
    header = []
    colIndex = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    for i in range(int(len(mask.channel_labels) / 2)):  # latter ones are edge
        #For header
        for j in range(len(im.channel_labels)):
            for distance in distances:
                for ls in range(len(colIndex)):
                    header.append(str(im.channel_labels[j] + ":" + colIndex[ls] + ":" + str(distance) + ":" + mask.channel_labels[i]))
        for idx in range(1,cell_total[0] + 1):  # For each cell
            #print("current calculation:", im.channel_labels[j] + "_" + str(distance) + "_" + mask.channel_labels[i])
            curProps=[]
            curROI=ROI_coords[i][idx]
            xmax,xmin,ymax,ymin=np.max(curROI[0]),np.min(curROI[0]),np.max(curROI[1]),np.min(curROI[1])
            interiormask = mask.get_data()[0, 0, i, bestz[0], :, :]
            interiormask=interiormask[xmin:xmax+1,ymin:ymax+1]
            interiormask = (interiormask == idx)
            for j in range(len(im.channel_labels)):  # For each channel
                img = im.get_data()[0, 0, j, bestz[0], :, :]
                img=img[xmin:xmax+1,ymin:ymax+1]
                img = np.multiply(interiormask, img)
                img = uint16_2_uint8(img)
                for distance in distances:
                    result = greycomatrix(img.astype(np.uint8), [distance], [angle], levels=256)  # Calculate GLCM
                    result = result[1:, 1:]  # Remove background influence by delete first row & column
                    props = []
                    for ls in range(len(colIndex)):  # Get properties
                        props.append(greycoprops(result, colIndex[ls]).flatten()[0])
                    curProps.append(props)
            texture_all=np.append(texture_all,np.asarray(curProps))
            #print("cur:",idx)
    texture_all=np.asarray(texture_all)
    texture_all = np.reshape(texture_all, (1, int(len(mask.channel_labels) / 2), cell_total[0], len(im.channel_labels), len(colIndex) * len(distances)))
    #For csv writing
    dataFrame=pd.DataFrame(np.reshape(texture_all,(cell_total[0],-1)))
    dataFrame.index=range(1,len(dataFrame)+1)
    dataFrame.index.name = 'ID'
    dataFrame.to_csv('sprm_outputs/'+filename + '_' +"texture.csv",header=header)
    return texture_all, header

def uint16_2_uint8(uint16matrix):
    maxvalue = np.max(uint16matrix)
    if maxvalue == 0:
        return uint16matrix
    return uint16matrix * (255 / maxvalue)

def glcmProcedure(im, mask, bestz, output_dir, cell_total, filename,ROI_coords, options):
    print("GLCM calculation initiated")
    angle = options.get('glcm_angles')
    distances = options.get('glcm_distances')
    angle = ''.join(angle)[1:-1].split(',')
    distances = ''.join(distances)[1:-1].split(',')
    angle = [int(i) for i in angle][0]  # Only supports 0 for now
    distances = [int(i) for i in distances]
    texture, texture_featureNames = glcm(im, mask, bestz, output_dir, cell_total, filename, options, angle, distances,ROI_coords)
    print("GLCM calculations completed")    
    return [texture, texture_featureNames]





