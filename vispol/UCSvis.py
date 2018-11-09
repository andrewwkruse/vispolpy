import numpy as np
import colorspacious as cs
from .aolp import StokestoAoLP as StA
from .dop import StokestoDoLP as StD
from .deltametric import delta, dmask

def clipatperc(X, perc):
    return np.clip(X / np.percentile(X, perc), 0, 1)
def clipatmax(X, Max):
    return np.clip(X/Max, 0, 1)

def Jbounds(M, xspline, yspline):
    # determine bounds J'0 and J'1 given a set of spline points approximating gamut curve c
    # inputs:
    #   M: colorfulness array representing DoLP
    #   xspline : set of J' points approximating curve c
    #   yspline : set of M' points approximating curve c
    slopes = []
    Mmax = max(yspline)
    bounds = np.zeros((M.shape[0], M.shape[1], 2)) # create empty array to place bounds into
    # find slopes between spline points
    for i in range(len(yspline) - 1):
        slopes.extend([(yspline[i + 1] - yspline[i]) / (xspline[i + 1] - xspline[i])])

    # solve piecewise function for c(J') = M
    for idx, p in np.ndenumerate(M):
        lower_done = False
        for i, y in enumerate(yspline):
            if not lower_done:
                if p <= y or (abs(p - y) < .00001):
                    if slopes[i-1]:
                        bounds[idx][0] = (p - y) / slopes[i - 1] + xspline[i]
                    else:
                        bounds[idx][0] = xspline[i]
                    lower_done = True
            else:
                if (p >= y or (abs(p - y) < .00001)) and i > yspline.index(Mmax):
                    if slopes[i - 1]:
                        bounds[idx][1] = (p - y) / slopes[i - 1] + xspline[i]
                    else:
                        bounds[idx][1] = xspline[i]
    return bounds

def PtoM(P,yspline):
    M = P * max(yspline)
    M[np.isnan(P)] = 0
    return M

def ItoJ(I,bounds):
    J = I * (bounds[:,:,1] - bounds[:,:,0]) + bounds[:,:,0]
    J[np.isnan(I)] = 0
    return J

def Atoh(A, red_offset = False):
    h = 2 * A
    h[np.isnan(A)] = 0
    # OPTIONAL: offset so that hue 0 -> sRGB Red to make hues match previous methods closer
    if red_offset:
        Red = cs.cspace_convert([1, 0, 0], "sRGB1", "CAM02-UCS")
        Rhue = np.arctan(Red[1] / Red[2])
        h += Rhue
    return h

def JMhtoJab(J,M,h):
    a = M * np.cos(h)
    b = M * np.sin(h)
    return J, a, b

def JabtoRGB(J,a,b, space='sRGB1'):
    s = list(J.shape)
    s.extend([3])
    J = np.reshape(J, -1)
    a = np.reshape(a, -1)
    b = np.reshape(b, -1)
    RGB = np.clip(cs.cspace_convert(np.stack([J,a,b],-1), "CAM02-UCS", space), 0, 1)
    return np.reshape(RGB, s)

def StokestoRGB(S,
                Ibar_params = None,
                Pbar_params = None,
                Abar_params = None,
                delta_params=None,
                xspline=[0, 5, 20, 40, 73, 77, 100],
                yspline=[0.0, 6.6, 13.7, 19.4, 26.4, 24.1, 0.0],
                returnUCS = False,
                space='sRGB1'):
    # inputs:
    #   S: n x m x 2,3, or 4 Stokes array
    #   Ibar_params, Pbar_params, Abar_params: list or dict of preparing I,P,A arrays
    #       if list, only include the parameter values
    #       if dict, include parameter name as the key with a colon, followed by the paramater value
    #       'function' : transformation function of the form fun(X,*params) where X is the polarization array
    #       'params' : if list, this is nested list, if dict this is a sub-dict
    #                  Other parameters that input into the function after the polarization array
    #   delta_params: either list or dictionary specifying Delta Mask parameters
    #       ##################
    #       Delta metric (Tyo et al 2016) is used to assess variability in angle of polarization. Depending on
    #       instrument and other situational variables, there can be a significant amount of noise in the degree of
    #       polarization. By setting a threshold of variability, the noise can be visibly reduced where the areas of
    #       high variability have their degree of polarization set to zero.
    #       ##################
    #       If list, only include the parameter values
    #       if dict, include parameter name as the key with a colon, followed by the paramater value
    #       'mask_on' : boolean, set to True for mask to be on. If the only param specified, rest will be default vals
    #       'element' : element, ##### see function delta from deltametric.py
    #       'mask_params' : { # if list, this is nested list, if dict this is a sub-dict,
    #                           see function dmask from deltametric.py
    #                       'thresh' : thresh,
    #                       'struct' : struct,
    #                       'morph' : morph,
    #                       'smooth' : smooth,
    #                       'sigma' : sigma
    #                       }
    #   xspline: set of xpoints approximating gamut curve, default is fit for sRGB
    #   yspline: set of xpoints approximating gamut curve, default is fit for sRGB
    #   space: RGB colorspace for output display. Spaces are defined in colorspacious docs, default is sRGB1

    I = S[:,:,0]
    if Ibar_params == None:
        Ibar = I
    elif type(Ibar_params) is list:
        Ibar = Ibar_params[0](I,*Ibar_params[1:])
    elif type(Ibar_params) is dict:
        Ibar = Ibar_params['function'](I,**Ibar_params['params'])
    else:
        Ibar = I
    nonnan = np.where(np.isnan(I))
    if np.any(I[nonnan]<0) or np.any(I[nonnan]>1):
        print('Reformatting Intensity (S0) to the range 0 to 1')
        Ibar = np.clip(Ibar/np.amax(Ibar[nonnan]), 0, 1)

    P = StD(S)

    if Pbar_params == None:
        Pbar = P
    elif type(Pbar_params) is list:
        Pbar = Pbar_params[0](P, *Pbar_params[1:])
    elif type(Pbar_params) is dict:
        Pbar = Pbar_params['function'](P,**Pbar_params['params'])
    else:
        Pbar = P
    A = StA(S)
    if Abar_params == None:
        Abar = A
    elif type(Abar_params) is list:
        Abar = Abar_params[0](A, *Abar_params[1:])
    elif type(Abar_params) is dict:
        Abar = Abar_params['function'](A, **Abar_params['params'])
    else:
        Abar = A

    if not delta_params == None:
        if type(delta_params) is bool:
            if delta_params:
                DM = delta(S)
                Mask = dmask(DM)
                Pbar *= Mask
        elif type(delta_params) is list:
            if delta_params[0]:
                DM = delta(S,delta_params[1])
                Mask = dmask(DM,*delta_params[2:])
                Pbar *= Mask
        elif type(delta_params) is dict:
            if delta_params['mask_on']:
                DM = delta(S, delta_params['element']) if 'element' in delta_params.keys() else delta(S)
                Dmaskattr = delta_params['mask_params'] if 'mask_params' in delta_params.keys() else {}
                Mask = dmask(DM,**Dmaskattr)
                Pbar *= Mask


    M = PtoM(Pbar,yspline)
    Jb = Jbounds(M, xspline, yspline)
    J = ItoJ(Ibar, Jb)
    h = Atoh(Abar)
    J, a, b = JMhtoJab(J, M, h)
    RGB = JabtoRGB(J, a, b, space)
    if returnUCS:
        Jab = np.stack([J, a, b], -1)
        return RGB, Jab
    else:
        return RGB

def IPAtoStokes(I, P = None, A = None, **kwargs):
    # Intensity (I), degree of linear polarization (P), and angle of polarization (A), can be entered as:
    #   3 individual NxM arrays
    #   1 combined NxMx3 array entered in the place of I
    # kwargs are the same as in StokestoRGB. They are directly passed to that function

    if len(I.shape) > 2:
        P = I[:,:,1]
        A = I[:,:,2]
        I = I[:,:,0]
    nonnan = np.where(np.isnan(A))
    if not (np.any(A[nonnan] < 0) or np.any(A[nonnan] > 1)):
        print('Assuming input Angle of Polarization is in the range 0 to 1...\n')
        print('Reformatting to the range 0 to pi')
        # range is 0 to 1, set to 0 to pi
        A *= np.pi

    S1 = P * np.cos(2*A)
    S2 = P * np.sin(2*A)
    Stokes = np.dstack((I,S1,S2))
    if 'returnUCS' in kwargs.keys():
        if kwargs['returnUCS']:
            RGB, Jab = StokestoRGB(Stokes, **kwargs)
            return RGB, Jab
        else:
            RGB = StokestoRGB(Stokes, **kwargs)
            return RGB
    else:
        RGB = StokestoRGB(Stokes, **kwargs)
        return RGB