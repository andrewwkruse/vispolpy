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
    #   S :  n x m x (2,3 or 4) Stokes
    #   element : side length of square window (element x element) for averaging
    # outputs:
    #   n x m delta metric for array

    # gives the number of cells that will be averaged for each cell. besides the edges, this will be element**2
    N = conv(np.ones((S.shape[0], S.shape[1])), np.ones((element, element)), 'same')
    P = StD(S) #DoP
    ca = S[:,:, 1]/ P # cos AoLP
    sa = S[:,:, 2]/ P # sin AoLP
    cam = conv(ca, np.ones((element, element)), 'same') / N # averaged cosine
    sam = conv(sa, np.ones((element, element)), 'same') / N # averaged sine

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