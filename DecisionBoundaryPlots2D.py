#!/usr/bin/env python
# coding: utf-8

# In[1]:

## The python style guide (https://pep8.org/) recommends putting all imports
## at the top of a module
## See PEP 257 (https://www.python.org/dev/peps/pep-0257/)
## about documentation strings for functions and modules in python

import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

def decBoundary(features,target, h, clf):
    """ 
    Function to draw the decision boundary of a classifier along with given classified data.
    The rectangular region where the boundary is drawn is determined by the range of values in the features data set
    features-> 2D dataset of features (shape = (m, 2) where m is number of data points) 
    target-> A target class vector, of shape = (m, ), that lists the class of each features data point 
    h -> coarseness of the grid to put in the data space 
    clf -> a FITTED classifier
    """
    xmin=np.min(features[:,0])
    xmax=np.max(features[:,0])
    ymin=np.min(features[:,1])
    ymax=np.max(features[:,1])
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.scatter(features[:,0], features[:,1],c=target,s=20,cmap='autumn')
    plt.show()
