import numpy as np
# from.aolp import StokestoAoLP as AoLP
from scipy.ndimage.filters import gaussian_filter as gf
import scipy.ndimage as nd
from .dop import StokestoDoLP as StD
from scipy.signal import convolve2d as conv
np.seterr(invalid='ignore')

# functions to calculate Delta metric and mask from stokes
# Last edited 10:44 09/02/2018 by AWK


def delta(S,element=3):
    # inputs:
    #   S :  n x m x 3 or 4) Stokes
    #   element :
    #       as int or float: side length of square window (element x element) for averaging
    #       as array or list: 2-D array describing (weighted) averaging window
    #           eg. 0 1 0
    #               1 1 1
    #               0 1 0

    # outputs:
    #   n x m delta metric for array
    element_error = 'element must be an int or a 2D array'
    if type(element) in [int, float]:
        neighborhood = np.ones((element, element))
    elif not type(element) is np.ndarray: # if it is not an int or float it must be array
        raise ValueError(element_error)
    elif not len(element.shape) == 2: # if it's an array must be 2D
        raise ValueError(element_error)
    else:
        neighborhood = element

    # gives the number of cells that will be averaged for each cell. besides the edges, this will be element**2
    N = conv(np.ones((S.shape[0], S.shape[1])), neighborhood, 'same')
    P = StD(S) #DoP
    ca = S[:,:, 1]/ P # cos 2 AoLP
    sa = S[:,:, 2]/ P # sin 2 AoLP
    cam = conv(ca, neighborhood, 'same') / N # averaged cosine
    sam = conv(sa, neighborhood, 'same') / N # averaged sine

    return np.sqrt(1 - (cam ** 2 + sam ** 2)) # delta metric, see Tyo et al 2016

def delta_aop(aop, element=3):
    # inputs:
    #   aop: n x m array of angle of polarization in radians with a range of pi
    #   element :
    #       as int or float: side length of square window (element x element) for averaging
    #       as array or list: 2-D array describing (weighted) averaging window
    #           eg. 0 1 0
    #               1 1 1
    #               0 1 0

    # outputs:
    #   n x m delta metric for array

    # gives the number of cells that will be averaged for each cell. besides the edges, this will be element*
    if type(element) in [int, float]:
        neighborhood = np.ones((element, element))
    elif not type(element) in [np.ndarray, list]:
        neighborhood = np.ones((3,3))
    else:
        neighborhood = element

    N = conv(np.ones((aop.shape[0], aop.shape[1])), neighborhood, 'same')
    ca = np.cos(2 * aop)
    sa = np.sin(2 * aop)  # sin AoLP
    cam = conv(ca, neighborhood, 'same') / N  # averaged cosine
    sam = conv(sa, neighborhood, 'same') / N  # averaged sine
    
    return np.sqrt(1 - (cam ** 2 + sam ** 2)) # delta metric, see Tyo et al 2016

def dmask(delta, thresh=0.5, morph=True, struct=3, smooth=False, sigma=1):
    mask = np.where(delta > thresh, 0, 1) # set values where delta > thresh to 0 and otherwise to 1
    mask[np.isnan(delta)] = 0 # make sure NaNs are masked
    if morph: # removes holes and island pixels
        mask = nd.binary_closing(
               nd.binary_opening(mask, structure=np.ones((struct,struct))),
               structure=np.ones((struct,struct)))
    if smooth:
        mask = gf(mask.astype(float), sigma) # reduces pixelation. Not applicable with mixed mapping from Tyo et al
                                             # only usable if P is multiplied by D prior to visualization
    return mask