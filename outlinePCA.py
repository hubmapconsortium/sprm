import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

"""

Companion to SPRM.py
Package functions that are integral to running main script
Author:    Ted Zhang & Robert F. Murphy
01/21/2020 - 04/07/2020
Version: 0.50


"""

def shape_cluster(cell_matrix,options):

    cellbycluster = KMeans(n_clusters=options.get("num_shapeclusters"),\
                           random_state=0).fit(cell_matrix)
    
    #returns a vector of len cells and the vals are the cluster numbers
    cellbycluster_labels = cellbycluster.labels_
    #print(cellbycluster_labels.shape)
    #print(cellbycluster_labels)
    #print(len(np.unique(cellbycluster_labels)))
    clustercenters = cellbycluster.cluster_centers_
    #print(clustercenters.shape)

    return cellbycluster_labels, clustercenters

def getcellshapefeatures(outls,options):

    numpoints = options.get("num_outlinepoints")
    pca_shapes = PCA(n_components=numpoints,svd_solver='full')
    #print(pca_shapes)

#    outlinesall = outls.reshape(outls.shape[0]*outls.shape[1],outls.shape[2])
#    print(outlinesall.shape)
    features = pca_shapes.fit_transform(outls)
    #print(features.shape)
    if features.shape[1] != numpoints:
        print('error: dimensions do not match.')
        exit
#    shape_features = features.reshape(outls.shape[0],outls.shape[1],check)
    
    return features

def getparametricoutline(mask,nseg,npoints):

    cellmask = mask.get_data()[0,0,nseg,0,:,:]
    #print(cellmask.shape)
    #print('Largest value in cell mask=')
    #print(np.amax(cellmask))
#    plt.imshow(cellmask)
#    plt.show()
    # the second dimension accounts for x & y coordinates for each point
    pts = np.zeros((np.amax(cellmask),npoints*2))
    for i in range(1,np.amax(cellmask)):

        coor = np.where(cellmask == i)
        #print(len(coor))
        #print(len(coor[0]))
        if edgetest(coor):
            #leave the coordinates for anything on the edge as zeros
            break
#        cmask[coor[0],coor[1]]=1

    #simple method from https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
        ptsx = coor[0]-round(np.mean(coor[0]))
        ptsy = coor[1]-round(np.mean(coor[1]))
        ptscentered = np.stack([ptsx,ptsy])
        #print(ptscentered.shape)
        xmin = min(ptscentered[0,:])
        #print(xmin)
        ymin = min(ptscentered[1,:])
        xxx = ptscentered[0,:]-xmin
        yyy = ptscentered[1,:]-ymin
        xxx = xxx.astype(int)
        yyy = yyy.astype(int)
        cmask = np.zeros(cellmask.shape)
        cmask[xxx,yyy]=1
        #plt.imshow(cmask)
        #plt.show()

        ptscov = np.cov(ptscentered)
        #print(ptscov)
        eigenvals, eigenvecs = np.linalg.eig(ptscov)
        #print(eigenvals,eigenvecs)
        sindices = np.argsort(eigenvals)[::-1]
        #print(sindices)
        x_v1, y_v1 = eigenvecs[:, sindices[0]] #eigenvector with largest eigenvalue
        x_v2, y_v2 = eigenvecs[:, sindices[1]]
        theta = np.arctan((x_v1)/(y_v1))
        #print(x_v1,y_v1,theta)
        rotationmatrix = np.matrix([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        tmat = rotationmatrix * ptscentered
        xrotated,yrotated = tmat.A
        #plt.plot(xrotated,yrotated,'b+')
        #plt.show()
        #need to flip over minor axis if necessary

        tminx = min(xrotated)
        #print(tminx)
        tminy = min(yrotated)
        #print(tminy)
        xrotated = xrotated - tminx
#        print(xrotated)
        tmatx = xrotated.round().astype(int)
#        print(tmatx)
        yrotated = yrotated - tminy
        tmaty = yrotated.round().astype(int)
        #make the object mask have a border of zeroes
        cmask = np.zeros((max(tmatx)+3,max(tmaty)+3))
        cmask[tmatx+1,tmaty+1]=1
        #fill the image to handle artifacts from rotation
        #find an interior point
        cmask = fillimage(cmask)
#        cmask = flood_fill(cmask,seed,1)
        
        #plt.imshow(cmask)
        #plt.show()
    
        pts[i,:]=paramshape(cmask,npoints)
        
    #print(pts.shape)
    return pts

def edgetest(coor):

    #check if any of the coordinates are on the edge of the image
    return False

def fillimage(cmask):

    changedsome = True
    while changedsome:
        changedsome = False
        for j in range(2,cmask.shape[0]-1):
            for k in range(2,cmask.shape[1]-1):
                if cmask[j,k] == 0:
                    m = cmask[j-1,k-1] + cmask[j-1,k+1] + \
                        cmask[j+1,k-1] + cmask[j+1,k+1]
                    if m==4:
                        cmask[j,k]=1
                        changedsome = True
    return cmask

def paramshape(cellmask,npoints):

    polyg = measure.find_contours(cellmask,0.5,fully_connected='low', positive_orientation='low')
    #print(len(polyg))
    #print(polyg)
    
#    if len(polyg) > 1:
#        print('Warning: too many polygons found')

    #print('in paramshape')
    for i in range(0,len(polyg)):
        poly = np.asarray(polyg[i])
        #print(i,poly.shape)
        #print(poly[0,0],poly[0,1],poly[-1,0],poly[-1,1])
        if i==0:
            polyall = poly
            #plt.plot(poly[:,0],poly[:,1],'bo')
            #plt.plot(poly[0,0],poly[0,1],'gd')
#        else:
#            polyall = np.append(polyall,poly,axis=0)
#            plt.plot(poly[:,0],poly[:,1],'rx')
#            plt.plot(poly[0,0],poly[0,1],'y+')
    #plt.show()
            
    #plt.plot(polyall[:,0],polyall[:,1],'rx')
    #plt.show()

    pts = interpalong(polyall,npoints)
    #print(pts.shape)
    #plt.plot(pts[:,0],pts[:,1],'go')
    #plt.show()

# return a linearized vector of x and y coordinates 
    xandy = pts.reshape(pts.shape[0]*pts.shape[1])
#    print(xandy)
    
    return xandy
    
def interpalong(poly,npoints):

    polylen = 0
    for i in range(0,len(poly)):
        j = i+1
        if i == len(poly)-1:
            j = 0
        polylen = polylen + np.sqrt((poly[j,0]-poly[i,0])**2 + \
                                    (poly[j,1]-poly[i,1])**2)
    #print(polylen)
    #polylen = poly.geometry().length()
#    minlen = minneidist(poly)
#    npmin = polylen/minlen
#        if npmin > npoints:
#            print('Warning: not enough interpolation points.')

    interval = polylen/npoints

    pts = np.zeros((npoints,2))
    pts[0,:] = poly[0,:]
    j = 1
    curpos = pts[0,:]
    for i in range(1,npoints):
        sofar = 0
        while sofar < interval:
            #check whether we wrapped around
            if j >= len(poly):
#                print('wrapped around')
#                print(j,len(poly))
                j = 0
            #print('i,j=')
            #print(i,j)
            xdist = poly[j,0] - curpos[0]
            ydist = poly[j,1] - curpos[1]
            tdist = np.sqrt(xdist**2 + ydist**2)
            need = interval - sofar
            #print(xdist,ydist,tdist,need)
            if tdist >= need:
                #save next sampled position
                #print('need to interpolate')
                ocurpos = curpos.copy()
                curpos[0] = curpos[0]+(need/tdist)*xdist
                curpos[1] = curpos[1]+(need/tdist)*ydist
                #print(ocurpos,curpos)
                pts[i,:] = curpos
                sofar = interval
#                if (curpos == poly[j,:]).all:
                if (curpos[0]==poly[j,0]) and (curpos[1]==poly[j,1]):
                    #print(curpos,poly[j,:])
                    #print('exact match of new point to a vertex')
                    j = j + 1
                    #print(j)
            else:
                #advance to the next vertex
                #print('advanced')
                #print(j,curpos)
                curpos = poly[j,:]
                j = j + 1
                sofar = sofar + tdist
        #print('found point')
        #print(i,j)
    #print(pts)
    return pts
                    

#def minneidist(poly):
#    for i in range(0,poly.size[0]-1):
#        ndist = dist(poly[i,:]-poly[i+1,:])
#    return min(ndist)
