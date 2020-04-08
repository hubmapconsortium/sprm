from itertools import product
from PIL import Image
import random

"""
Companion to SPRM.py
Used for finding connected components for image without mask 
Function: Simple boogle algo to determine set membership of all elements useful in determining connected components
Author:     @Ted Zhang
1/17/2020

"""

class UFarray:
    def __init__(self):
        # Array which holds label -> set equivalences
        self.P = []

        # Name of the next label, when one is created
        self.label = 0

    def makeLabel(self):
        r = self.label
        self.label += 1
        self.P.append(r)
        return r
    
    # Makes all nodes "in the path of node i" point to root
    def setRoot(self, i, root):
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root

    # Finds the root node of the tree containing node i
    def findRoot(self, i):
        while self.P[i] < i:
            i = self.P[i]
        return i
    
    # Finds the root of the tree containing node i
    # Simultaneously compresses the tree
    def find(self, i):
        root = self.findRoot(i)
        self.setRoot(i, root)
        return root
    
    # Joins the two trees containing nodes i and j
    # Modified to be less agressive about compressing paths
    # because performance was suffering some from over-compression
    def union(self, i, j):
        if i != j:
            root = self.findRoot(i)
            rootj = self.findRoot(j)
            if root > rootj: root = rootj
            self.setRoot(j, root)
            self.setRoot(i, root)
    
    def flatten(self):
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]
    
    def flattenL(self):
        k = 1
        for i in range(1, len(self.P)):
            if self.P[i] < i:
                self.P[i] = self.P[self.P[i]]
            else:
                self.P[i] = k
                k += 1
                
class Watcher:
    '''' simple class to watch variable'''
    
    def __init__(self):
        self.v = 0
        self.count = 0
    def compare(self,new_value):
        if self.v != new_value:
            self.pre_change()
            self.v = new_value
    def pre_change(self):
        self.count += 1
    def get_counter(self):
        return self.count

def get_cc(img):
    data = img
    width = img.shape[0]
    height = img.shape[1]
    # watches for any changes to components which represent a different region 
    watch = Watcher()
    # Pixel indexed regions
    pixelidx = {}
    # Union find data structure
    uf = UFarray()
 
    #
    # First pass
    #
 
    # Dictionary of point:label pairs
    labels = {}
 
    for y, x in product(range(height), range(width)):
 
        #
        # Pixel names were chosen as shown:
        #
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e |   |
        #   -------------
        #   |   |   |   |
        #   -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 1 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #
 
        # If the current pixel is black, it's obviously not a component...
        if data[x, y] == 0:
            pass
 
        # If pixel b is in the image and white:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and data[x, y-1] == 1:
            labels[x, y] = labels[(x, y-1)]
 
        # If pixel c is in the image and white:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x+1 < width and y > 0 and data[x+1, y-1] == 1:
 
            c = labels[(x+1, y-1)]
            labels[x, y] = c
 
            # If pixel a is in the image and white:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and data[x-1, y-1] == 1:
                a = labels[(x-1, y-1)]
                uf.union(c, a)
 
            # If pixel d is in the image and white:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and data[x-1, y] == 1:
                d = labels[(x-1, y)]
                uf.union(c, d)
 
        # If pixel a is in the image and white:
        #    We already know b and c are black
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and data[x-1, y-1] == 1:
            labels[x, y] = labels[(x-1, y-1)]
 
        # If pixel d is in the image and white
        #    We already know a, b, and c are black
        #    so simpy assign d's label to e
        elif x > 0 and data[x-1, y] == 1:
            labels[x, y] = labels[(x-1, y)]
 
        # All the neighboring pixels are black,
        # Therefore the current pixel is a new component
        else: 
            labels[x, y] = uf.makeLabel()
 
    #
    # Second pass
    #
 
    uf.flatten()
 
    colors = {}

    # Image to display the components in a nice, colorful way
    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()
  
    for (x, y) in labels:
        
        
        
        # Name of the component the current point belongs to
        component = uf.find(labels[(x, y)])
    
        # Update the labels with correct information
        labels[(x, y)] = component
        
        
        # #watcher for component
        # watch.compare(component)
        # counter = watch.get_counter()
        
        # # Get pixelidx dict
        # if counter != 0:
        #     if counter in pixelidx:
        #         pixelidx[counter].append((x,y))
        #     else:
        #         pixelidx[counter] = [(x,y)]


        # Associate a random color with this component 
        if component not in colors: 
            colors[component] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
            
        # Colorize the image
        outdata[y, x] = colors[component]
        
    # labels = pixelidx
    
    return (labels, output_img)