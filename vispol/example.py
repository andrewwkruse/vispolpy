import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from .UCSvis import clipatmax, clipatperc, StokestoRGB

# This provides a quick implementation of StokestoRGB with a provided simulated Stokes array

def generate(out_type = 'Stokes'):
    ################### Sample Stokes ###################
    # out_type should be either 'Stokes' or 'IPA'

    S0 = np.tile(np.linspace(0, 1, 100), (100, 1))
    S0 *= np.random.uniform(.7, 1, (100, 100))  # add noise
    # Example of false polarization signals:
    # S1 and S2 are moderately high, but because the resulting AoLP has high variance, a delta mask would suppress it
    S1 = np.random.uniform(-.4, .4, (100, 100))
    S2 = np.random.uniform(-.4, .4, (100, 100))
    # Actual polarized square
    S1[25:75, 25:75] = np.random.uniform(0.3, 0.6, (50, 50))  # S1 is large while S1 is small so AoLP has low variance
    S2[25:75, 25:75] = np.random.uniform(-0.1, 0.1, (50, 50))
    S = np.stack((S0, S1, S2), -1)
    ###############################
    if out_type == 'Stokes':
        return S
    elif out_type == 'IPA':
        P = np.sqrt(S1**2 + S2**2)
        A = 0.5 * np.atan2(S2, S1)
        IPA = np.dstack((S0, P, A))
        return IPA
    else:
        raise ValueError('Please enter out_type as \'Stokes\' or \'IPA\'')
def example(S,
                Ibar_params = None,
                Pbar_params = None,
                Abar_params = None,
                delta_params=None,
                xspline=[0, 5, 20, 40, 73, 77, 100],
                yspline=[0.0, 6.6, 13.7, 19.4, 26.4, 24.1, 0.0],
                space='sRGB1'):
    print('Here are your parameters')
    print('Ibar_params = ', Ibar_params)
    print('Pbar_params = ', Pbar_params)
    print('Abar_params = ', Abar_params)
    print('delta_params = ', delta_params)
    print('xspline = ', xspline)
    print('yspline = ', yspline)
    print('space = ', space)
    print('\n')
    if not np.any([Ibar_params, Pbar_params, delta_params]):
        print('Here are some other options to consider')
        print('Ibar_params = ')
        print('\t{\'function\':clipatperc, \'params\' : {\'perc\':99}} (dict version)')
        print('\t[clipatperc, 99] (list version)')
        print('\t[clipatperc, [99]] (nested list version)')
        print('Pbar_params = ')
        print('\t{\'function\':clipatmax, \'params\' : {\'Max\':0.5}} (dict version)')
        print('\t[clipatmax, 0.5] (list version)')
        print('\t[clipatmax, [0.5]] (nested list version)')
        print('delta_params = ')
        print('\t{\'mask_on\':True,\'element\':3,\'mask_prams\':')
        print('\t\t{\'thresh\':0.4,\'struct\':3,\'morph\':True,\n\t\t\'smooth\':True,\'sigma\':2}} (dict version)')
        print('\tTrue (all defaults)')
        print('\t[True, 3, 0.4, 3, True, True, 2] (list version)')
        print('\t[True, 3, [0.4, 3, True, True, 2]] (nested list version)')



    out = StokestoRGB(S,
                      Ibar_params,
                      Pbar_params,
                      Abar_params,
                      delta_params,
                      xspline,
                      yspline,
                      space)
    return out

