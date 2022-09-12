from time import time
import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os



#------------------Data cleaning------------------------------------------------------------
def sample_even_from_group_idx(y):
    y2treat = pd.Series(y).map(wells_to_genetype_dict)
    y2treat = pd.DataFrame(y2treat)
    n_min = y2treat.groupby(0).size().min()
    y_treat_sample = y2treat.groupby(0).sample(n_min)
    return y_treat_sample.index.values

def normalize_axis(X_train):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    return x_norm , scaler

def idx_no_outliers_after_norm(x_norm):
    x_norm_re = x_norm.copy()
    x_norm_re = x_norm.reshape(x_norm.shape[0], x_norm.shape[1]*x_norm.shape[2])
    idx = np.where(np.all(np.abs(x_norm_re)<8, axis =1))[0]
    return idx



# -------------------Feature Transformation----------------------------------------
def spheroid(long_sq,short_sq):
    return np.sqrt((long_sq-short_sq)/long_sq) if long_sq!=short_sq else 0
def elip(majorminor):
    maj_sq = majorminor[0]**2
    min_sq = majorminor[1]**2
    return spheroid(maj_sq,min_sq) if maj_sq>min_sq else spheroid(min_sq,maj_sq)

def getAngleRad(diffs):
    return np.arctan2(diffs[1], diffs[0])
def getAngleRadZero(xy):
    return np.arctan2(xy[1]-0, xy[0]-0)

def add_transformations(X_train):
    #ellipticity
    ellipticity = np.apply_along_axis(elip, 2, X_train[:,:,3:5])
    X_train = np.dstack((X_train,ellipticity))
    #step size
    d = np.diff(X_train[:,:,:2],axis=1,prepend=0)
    step_size = np.sqrt(np.power(d,2).sum(axis=2))
    X_train = np.dstack((X_train,step_size))
    #displacement
    displacement = np.sum(X_train[:,:,:2]**2,axis=2)
    X_train = np.dstack((X_train,displacement))
    #acceleration
    acceleration = np.diff(step_size,axis=1,prepend=0)/30.0
    X_train = np.dstack((X_train,acceleration))
    #angle of step
    d = np.diff(X_train[:,:,:2],axis=1,prepend=0)
    angles_of_step = np.apply_along_axis(getAngleRad, 2, d)
    X_train = np.dstack((X_train,angles_of_step))
    #angle from center
    angles_from_center = np.apply_along_axis(getAngleRad, 2, X_train[:,:,:2])
    X_train = np.dstack((X_train,angles_from_center))

    return X_train
