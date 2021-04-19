# -*- coding: utf-8 -*-

#Created on Fri Feb 19 15:28:18 2021
#
#@author: Young Je Lee



import FR_pkgs as pkg
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import SPRM.SPRM_pkg as sprm
from pathlib import Path

'''
Load csv files

X: cov, meanAll, mean, total, texture, shape
y(for reference): clusterIdx
'''


f_cov='R001_X001_Y001.ome.tiff-cells_channel_covar.csv'
f_meanAll='R001_X001_Y001.ome.tiff-cell_channel_meanAll.csv'
f_mean='R001_X001_Y001.ome.tiff-cells_channel_mean.csv'
f_total='R001_X001_Y001.ome.tiff-cells_channel_total.csv'
f_texture='R001_X001_Y001.ome.tiff_texture123.csv'
f_shape='R001_X001_Y001.ome.tiff-cell_shape.csv'
f_clusterIdx='R001_X001_Y001.ome.tiff-cells_cluster.csv'


matrix_cov,header_cov=pkg.loadCSV(f_cov)
matrix_meanAll,header_meanAll=pkg.loadCSV(f_meanAll)
matrix_mean,header_mean=pkg.loadCSV(f_mean)
matrix_total,header_total=pkg.loadCSV(f_total)
matrix_texture,header_texture=pkg.loadCSV(f_texture)
matrix_shape,header_shape=pkg.loadCSV(f_shape)
matrix_clusterIdx,header_clusterIdx=pkg.loadCSV(f_clusterIdx)


'''
Load mask for visualization
codes from SPRM V1.0, author: Ted


'''
mask_path='SPRM/mask_hubmap/R001_X001_Y001.ome.tiff'
options_path='SPRM/options.txt'
mask_files = sprm.get_paths(Path(mask_path))
options = sprm.read_options(Path(options_path))
mask= sprm.MaskStruct(mask_files[0], options)
if mask.get_data().shape[0] > 1 and len(mask.get_channel_labels()) > 1:
    data = mask.get_data()
    s, t, c, z, y, x = data.shape
    data = data.reshape(c, t, s, z, y, x)
    mask.set_data(data)
ROI_coords = sprm.get_coordinates(mask, options)
img_dir='SPRM/img_hubmap/R001_X001_Y001.ome.tiff'
img_files = sprm.get_paths(Path(img_dir))
img_file = img_files[0]
im = sprm.IMGstruct(img_file, options)

sprm.quality_control(mask, im, ROI_coords, options)
inCells = mask.get_interior_cells()


'''
Preprocessing
1. Texture can be splited into cell only and cell+nuclei
2. Provide merged every features
3. Texture matricies with splited based on radii: Influenced by 1

'''
#Global variable: numCells
numCells=len(matrix_cov)

# 1. Cell only
cellTexture=matrix_texture[:,:int(len(matrix_texture[0])/2)]
cellTexture_header=header_texture[:int(len(matrix_texture[0])/2)]

# 2. Merged Array
matrix_all=np.concatenate((matrix_mean,matrix_cov,matrix_total,matrix_texture),axis=1)
matrix_all_OnlyCell=np.concatenate((matrix_mean,matrix_cov,matrix_total,cellTexture),axis=1)



###Split based on radii
##Cell+Nuclei
#matrix1,header1,matrix2,header2,matrix3,header3=pkg.Texture_split_radii(matrix_texture,header_texture,numCells,numRadii=3,numChan=18,numTextureFeature=6)
##Cell only
matrix1,header1,matrix2,header2,matrix3,header3=pkg.Texture_split_radii(cellTexture,cellTexture_header,numCells,numRadii=3,numChan=18,numTextureFeature=6)

'''
First task: Observation between covariance matrix and texture
Since texture has more features, we can use it as a input

input: texture features matrix of one channel
Y : input channel - its relation between other.

'''    
#Path('ScatterPlot').mkdir(parents=False,exist_ok=True)
# for i in range(len(matrix_cov[0])):
#     plt.scatter(matrix_texture1[:,0],matrix_cov[:,i])

'''
PCA

PCA texture based on radii
'''

# comp1=pkg.PrincipleComponentAnalysis_withPlot(matrix1,'radii 1')
# comp2=pkg.PrincipleComponentAnalysis_withPlot(matrix2,'radii 2')
# comp3=pkg.PrincipleComponentAnalysis_withPlot(matrix3,'radii 3')
# comp4=pkg.PrincipleComponentAnalysis_withPlot(matrix4,'all')



'''
Factor Analysis - ongoing
'''

# # df1=pd.DataFrame(matrix1)
# # from factor_analyzer import FactorAnalyzer
# # from factor_analyzer.factor_analyzer import calculate_kmo
# # kmo_all,kmo_model=calculate_kmo(df1)

# # fa = FactorAnalyzer()
# # fa.fit(df1)
# # fa.analyze(df1, 30, rotation=None)
# # ev, v = fa.get_eigenvalues()
'''
Clustering Observation


use inCells. Outer cells are considered as background
'''
cellImg=mask.get_data()[0,0,0,0,:,:]
groundTruth=[cellImg,inCells,ROI_coords]

# numClusters=20

# for i in range(1,numClusters):
#     pkg.KMeanWithVisual(matrix_all_OnlyCell,i,groundTruth)

# plt.show()


    

# #for i in range(0,len(header_texture),6):
# for i in range(2):
# #for i in range(6,7):
#     for j in range(18):
#         for k in range(6):
#             inputMatrix = matrix_texture[:,i*6+k]
#             target = matrix_cov[:,i*18+j]
#             pkg.ScatterPlot(inputMatrix,target,header_texture1[i*6+k]+'   &&   '+header_cov[i*18+j])

#         #Run PCA
#         pca=PCA(n_components=2)
#         pcaInput=matrix_texture[:,i:i+6]
#         pca.fit(pcaInput)
#         inputMatrix = np.dot(matrix_texture1[:,i:i+6],pca.components_[0])
#         target = matrix_cov[:,i*18+j]
#         pkg.ScatterPlot(inputMatrix,target,header_texture1[i]+'   &&   '+header_cov[i*18+j]+'pca')


'''
t-SNE
'''


numComp=2
from sklearn.manifold import TSNE
matrix_all=np.asarray(matrix_all,dtype=float)
perplexityValue=5
tsne=TSNE(n_components=numComp,init='random',perplexity=perplexityValue)
tsne_all=tsne.fit_transform(matrix_all)
for i in range(1,10):
    pkg.KMeanWithVisual(tsne_all,i,groundTruth,'tsne_'+str(perplexityValue))


# for i in range(5):
#     perplexity=5+i*10
#     tsne=TSNE(n_components=numComp,init='random',perplexity=perplexity)
#     tsne_all=tsne.fit_transform(matrix_all)
#     pkg.KMeanWithVisual(tsne_all,3,groundTruth)
    # fig=plt.figure(figsize = (10,10))
    # axes=plt.gca()
    # axes.set_xlim([-200,200])
    # axes.set_ylim([-200,200])
    
    # plt.title("tsne with"+str(i))
    # plt.scatter(tsne_all[:,0],tsne_all[:,1],alpha=0.25)
    # plt.xlabel('tsne1')
    # plt.ylabel('tsne2')
    # plt.savefig('tsne_'+'all'+'_'+str(perplexity)+'.jpg')
    # plt.clf()



# for i in range(1,numClusters):
#     pkg.KMeanWithVisual(matrix_all_OnlyCell,i,groundTruth)


