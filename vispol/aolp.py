import numpy as np
from .deltametric import delta, delta_aop, dmask
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.sparse import diags

def circular_mean(angles, weights=None, interval='auto'):
    weight_err = 'Weights must be an array or list with the same size as the input array of angles or None'
    angles_err = 'Input angles must be an array or list'
    interr = 'interval must be a list, tuple, or array of length 2 or \'auto\''

    if type(angles) in [list, tuple]:
        a = np.array(angles)
    elif type(angles) is np.ndarray:
        a = angles
    else:
        raise ValueError(angles_err)

    if weights is None:
        w = np.ones_like(a)
    elif type(weights) in [list, tuple]:
        w = np.array(weights)
        if not w.shape == a.shape:
            raise ValueError(weight_err)
    elif type(weights) is np.ndarray:
        w = weights
    else:
        raise ValueError(weight_err)

    if interval is 'auto':
        [bottom, top] = detect_range(a, readout=False)
        if bottom is None:
            raise ValueError('Cannot detect interval for input. Rerun with interval input specified')
    else:
        try:
            if len(interval) == 2:
                [bottom, top] = interval
            else:
                raise ValueError(interr)

        except:
            raise ValueError(interr)

    a_pi = np.pi * (a - bottom) / (top - bottom)
    cosa = np.cos(2.0 * a_pi)
    sina = np.sin(2.0 * a_pi)
    # average_a = np.arctan2(np.sum(sina * w, axis=-1), np.sum(cosa * w, axis=-1)) / 2.0
    average_a = np.arctan2(np.sum(sina * w), np.sum(cosa * w)) / 2.0
    if type(average_a) in [float, np.float64, np.float32]:
        if average_a < 0:
            average_a += np.pi
    else:
        average_a[np.where(average_a < 0)] += np.pi
    scaled = (average_a / np.pi) * (top - bottom) + bottom
    return scaled

def StokestoAoLP(S):
# function to calculate AoLP of 2D array of Stokes vectors (as in an image)
# S = input stokes vector
# S can be nxmx2, nxmx3, nxmx4 array
# nxmx2:
#   S[0[ = S1, S[1] = S2
# nxmx3:
#   S[0] = S0, S[1] = S1, S[2] = S2
# nxmx4:
#   S[0] = S0, S[1] = S1, S[2] = S2, S[3] = S3
#
# outputs:
#   a: nxm vector of angles
#

# Last edited 15:41 08/02/2017 by AWK

    # set indices
    if S.shape[2] == 2:
        S1 = S[:, :, 0]
        S2 = S[:, :, 1]
    else:
        S1 = S[:, :, 1]
        S2 = S[:, :, 2]

    a = 0.5 * np.arctan2(S2, S1)
    a[np.isnan(S1)] = np.nan
    a[np.isnan(S2)] = np.nan
    a[np.where(np.isin(S1,[0]) * np.isin(S2,[0]))] = np.nan # set 0/0 to nan
    return a

def StokestoAoLP_masked(S, **delta_params):
# same as StokestoAoLP except returns a masked AoLP array where delta_params are the inputs to function deltametric
    aolp_array = StokestoAoLP(S)
    delta_array = delta(S, element=delta_params['element']) if 'element' in delta_params else delta(S)
    mask_params = {}
    for p in ['thresh', 'morph', 'struct']:
        if p in delta_params.keys():
            mask_params[p] = delta_params[p]
    delta_mask = dmask(delta_array, **mask_params)
    aolp_array = np.where(delta_mask, aolp_array, np.nan) # where mask is false replace with nan
    return aolp_array

def detect_range(aop, badvals=None, readout=True):
    if badvals is not None:

        try:
            _ = len(badvals)
        except:
            badvals = [badvals]

        goodvals = np.array([a for a in aop.flatten() if a not in badvals and not np.isnan(a)])
    else:
        goodvals = np.array([a for a in aop.flatten() if not np.isnan(a)])

    maxval = max(goodvals)
    minval = min(goodvals)
    ninetynine_percentile = np.percentile(goodvals, 99.9)
    first_percentile = np.percentile(goodvals, 0.1)

    possible_tops = np.array([1.0, np.pi/2.0, np.pi, 90.0, 180.0])
    possible_bottoms = np.array([-90.0, -np.pi/2.0, 0.0])
    possible_tops_ranges = np.array([1.0, np.pi, np.pi, 180.0, 180.0])

    max_diff = np.abs(possible_tops - maxval)
    min_diff = np.abs(possible_bottoms - minval)
    ninetynine_diff = np.abs(possible_tops - ninetynine_percentile)
    first_diff = np.abs(possible_bottoms - first_percentile)

    close = 0.05

    max_index = np.argmin(max_diff)
    ninetynine_index = np.argmin(ninetynine_diff)

    if max_diff[max_index] <= ninetynine_diff[ninetynine_index]:
        top = possible_tops[max_index]
        rangeof = possible_tops_ranges[max_index]
        checktop = np.abs(top - maxval) / rangeof
    else:
        top = possible_tops[ninetynine_index]
        rangeof = possible_tops_ranges[ninetynine_index]
        checktop = np.abs(top - ninetynine_percentile) / rangeof

    if checktop > close:
        return [None, None]

    min_index = np.argmin(min_diff)
    first_index = np.argmin(first_diff)

    if min_diff[min_index] <= first_diff[first_index]:
        bottom = possible_bottoms[min_index]
        checkbottom = np.abs(bottom - minval) / rangeof
    else:
        bottom = possible_bottoms[min_index]
        checkbottom = np.abs(bottom - minval) / rangeof

    if checkbottom > close:
        return [None, None]

    if top is None or bottom is None:
        return [None, None]

    if readout:
        print('Max Value = {} \n'
              'Min Value = {} \n'
              '99.9 Percentile = {} \n'
              '0.1 Percentile = {} \n'.format(maxval,
                                              minval,
                                              ninetynine_percentile,
                                              first_percentile))
        if top is not None:
            print('Estimated range is {}, {}'.format(bottom, top))
        else:
            print('Cannot estimate range')
    return [bottom, top]

def colormap(isoluminant=False):
    # this creates a periodic colormap appropriate for AoP
    # colormap is formed in UCS color space by parametrizing a tilted circle in 3D
    # the center of the circle is located at J' = 70, a' = 0, b' = 0
    # the circle has a zenith angle of 45 degrees into the lightness axis
    # the circle has an azimuthal angle oriented with the highest portion at -60 degrees
    # colors have been copy-pasted for efficiency from calculations made by author
    # see "Polarization-color mapping strategies: catching up with color theory" by Kruse et al

    from matplotlib.colors import LinearSegmentedColormap as lsc
    from matplotlib.cm import register_cmap as rc

    colors = aop_colors(isoluminant)

    name = "isoluminantAoP" if isoluminant else "AoP"
    cmap = lsc.from_list(name, colors)

    rc(name, cmap) # cmap can be referenced by the name "AoP" instead of saving cmap as a variable
    return cmap

def LUT_matching(aop, lut):
    # locates closest values in aop array to the first row of the lut and replaces with
    # corresponding values in second row of lut

    # If tick locations are put in for aop, you will get back the transformed tick locations

    aop_matched = np.array([lut[1,np.argmin(np.abs(a - lut[0]))] if not np.isnan(a) else np.nan
                            for idx, a in np.ndenumerate(aop)]
                           ).reshape(aop.shape)

    return aop_matched

def histogram_eq(aop,
                 bins='fd',
                 box = None,
                 interval='auto',
                 weighted=False,
                 min_change = 0.0,
                 suppress_noise=False,
                 deltas=None,
                 histogram=None,
                 **delta_params):
    boxerr = 'box must be 2x2 array like [[row_start, row_end], [column_start, column_end]]'
    if type(box) in [list, tuple]:
        try:
            box = np.array(box)
        except:
            raise ValueError(boxerr)

    if box is None:
        hist_region = np.ones_like(aop, dtype=bool)

    elif type(box) is np.ndarray:
        if box.shape == aop.shape:
            hist_region = box
        elif box.shape == (2, 2):
            rowstart, rowend = box[0]
            colstart, colend = box[1]
            hist_region = np.zeros_like(aop, dtype=bool)
            hist_region[rowstart:rowend, colstart:colend] = True
        else:
            raise ValueError(boxerr)
    else:
        raise ValueError(boxerr)

    interr = 'interval must be a list, tuple, or array of length 2 or \'auto\''
    if interval is 'auto':
        [bottom, top] = detect_range(aop, readout=False)
        if bottom is None:
            raise ValueError('Cannot detect interval for input. Rerun with interval input specified')
    else:
        try:
            if len(interval) == 2:
                [bottom, top] = interval
            else:
                raise ValueError(interr)

        except:
            raise ValueError(interr)

    deltaerr = 'Input deltas must be array with the same shape as aop'
    if deltas is None:
        aop_pi = np.pi * (aop - bottom) / (top - bottom)
        delta_array = delta_aop(aop_pi, element=delta_params['element']) \
            if 'element' in delta_params else delta_aop(aop_pi)
    else:
        try:
            if deltas.shape == aop.shape:
                delta_array = deltas
            else:
                raise ValueError(deltaerr)
        except:
            raise ValueError(deltaerr)

    if histogram is None:
        aop_good = aop[hist_region] # select values indicated by box
        aop_not_nans = ~np.isnan(aop_good)
        aop_good = aop_good[aop_not_nans] # remove nan from array
        if weighted:
            delta_array = delta_array[hist_region]
            delta_array = delta_array[aop_not_nans]
            weights = 1 - delta_array
            weights[np.isnan(weights)] = 0
            # for weighted data you can't automate bin edges so they have to be initialized first
            edges = np.histogram_bin_edges(aop_good, bins=bins)
            counts, edges = np.histogram(aop_good, bins=edges, weights=weights)

        else:
            mask_params = {}
            for p in ['thresh', 'morph', 'struct']:
                if p in delta_params.keys():
                    mask_params[p] = delta_params[p]
            delta_mask = dmask(delta_array, **mask_params)
            delta_mask = delta_mask[hist_region]
            delta_mask = delta_mask[aop_not_nans]
            aop_masked = np.where(delta_mask, aop_good, np.nan)  # where mask is false replace with nan
            counts, edges = np.histogram(aop_masked, bins=bins)
    else:
        counts, edges = histogram

    pdf = counts / np.sum(counts)
    old_levels = edges[:-1]  # leftmost bin edges used for original pixel values
    num_bins = len(old_levels)
    if min_change < 1.0:
        # d is the scalar amount added to the pdf s.t. the derivative of the lut is always greater than min_change
        # min_change = 0 returns lut as it would normally
        # as min_change -> 1, lut becomes more linear
        # e.g. if min_change = 0.5, the returned LUT can only "shrink" parts to half of the normal rate
        d = min_change / (num_bins * (1 - min_change))
        pdf += d
        pdf /= np.sum(pdf)
    else:
        # if min_change is 1, calculation of d has div0, so instead return a uniform pdf
        pdf = 1.0 / num_bins * np.ones_like(pdf)

    cdf = np.cumsum(pdf)

    new_levels = cdf * (top - bottom) + bottom
    lut = np.stack((old_levels, new_levels))

    aop_equalised = LUT_matching(aop, lut)

    if suppress_noise:
        aop_eq_pi = np.pi * (aop_equalised - bottom) / (top - bottom)
        if 'element' in delta_params:
            delta_eq = delta_aop(aop_eq_pi, element=delta_params['element'])
            if type(delta_params['element']) in [int, float]:
                neighborhood = np.ones((delta_params['element'], delta_params['element']))
            else:
                neighborhood = delta_params['element']
        else:
            neighborhood = np.ones((3,3))
            delta_eq = delta_aop(aop_eq_pi)

        cos_average = convolve2d(np.cos(2 * aop_eq_pi), neighborhood, 'same')
        sin_average = convolve2d(np.sin(2 * aop_eq_pi), neighborhood, 'same')
        aop_average = np.arctan2(sin_average, cos_average) / 2
        aop_average[aop_average < 0] += np.pi

        aop_both = np.dstack((aop_average, aop_eq_pi))
        weights = np.dstack((delta_eq, 1.0 - delta_eq))
        aop_equalised = circular_mean(aop_both, weights=weights, interval=[0, np.pi])
        aop_equalised = (aop_equalised + bottom) * (top - bottom) / np.pi
    return aop_equalised, lut

def histogram_eq_Stokes(S, bins='fd', interval='auto', weighted=False, min_change = 0.0, **delta_params):
    aop = StokestoAoLP(S)
    aop_equalised, lut = histogram_eq(aop, bins=bins, interval=interval, weighted=weighted, min_change=min_change,
                                      **delta_params)
    return aop_equalised, lut

def colormap_delta(aop,
                   background=[0.0,0.0,0.0],
                   deltas = None,
                   method='opacity',
                   isoluminant=False,
                   interval='auto',
                   **kwargs):
    if method in ['opacity', 'Opacity']:
        mask = False
    elif method in ['mask', 'Mask']:
        mask = True
    else:
        raise ValueError('Invalid method, must be \'opacity\' or \'mask\'')

    # check if background is between 0 and 1 and if it is either a number or a list of length 3
    bg_error = 'background color should be entered as a gray value between 0 and 1 or an rgb ' \
               'triplet of length 3'
    if not isinstance(background, list):
        try:
            between01 = 0 <= background and 1 >= background
        except:
            raise ValueError(bg_error)
        if not between01:
            raise ValueError(bg_error)
        else:
            background = [background] * 3
    elif not len(background) == 3:
        raise ValueError(bg_error)
    else:
        for b in background:
            if b < 0 or b > 1:
                raise ValueError(bg_error)

    aoperr = 'Input aop must be NxM array'
    try:
        dims = aop.shape
        if not len(dims) == 2:
            raise ValueError(aoperr)
    except:
        raise ValueError(aoperr)

    interr = 'interval must be a list, tuple, or array of length 2 or \'auto\''
    if interval is 'auto':
        [bottom, top] = detect_range(aop, readout=False)
        if bottom is None:
            raise ValueError('Cannot detect interval for input. Rerun with interval input specified')
    else:
        try:
            if len(interval) == 2:
                [bottom, top] = interval
            else:
                raise ValueError(interr)

        except:
            raise ValueError(interr)

    aop_pi = np.pi * (aop - bottom) / (top - bottom)
    cmap = colormap(isoluminant)
    colors = cmap(aop_pi / np.pi)
    if deltas is None:
        delta_array = delta_aop(aop_pi, element=kwargs['element']) if 'element' in kwargs else delta_aop(aop_pi)
    else:
        delta_array = deltas

    if mask:
        params = {}
        for p in ['thresh', 'morph', 'struct', 'smooth', 'sigma']:
            if p in kwargs.keys():
                params[p] = kwargs[p]
        alpha = dmask(delta_array, **params)
    else:
        alpha = 1 - delta_array

    alpha = np.where(~np.isnan(alpha), alpha, 0)
    colors[:,:,3] = alpha
    rgb = rgba2rgb(colors, background)

    return rgb

def colormap_delta_Stokes(S,
                          background=[0.0,0.0,0.0],
                          method='opacity',
                          isoluminant=False,
                          **kwargs):
    aop = StokestoAoLP(S)
    rgb = colormap_delta(aop, background, method, isoluminant, **kwargs)
    return rgb

def rgba2rgb(rgba, background):
    bg_array = np.ones_like(rgba[:, :, :3]) * background
    alpha = rgba[:, :, 3]
    # reformat arrays so they can be broadcasted
    rgba_formatted = np.moveaxis(rgba[:, :, :3], 2, 0)
    bg_formatted = np.moveaxis(bg_array, 2, 0)
    rgb = (1 - alpha) * bg_formatted + alpha * rgba_formatted
    # reformat back to original shape
    rgb = np.moveaxis(rgb, 0, 2)
    return rgb

def sobel_aop_single(aop, type='x', sigma=None, mode='same', center=True):
    if type in ['x', 'X']:
        # kernel = np.array([[-1, -2, -1],
        #                    [ 0,  0,  0],
        #                    [ 1,  2,  1]])
        if center:
            kernel1 = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [0, 0, 0]])

            kernel2= np.array([[0, 0, 0],
                                [0, 0, 0],
                                [1, 2, 1]])
        else:
            kernel1 = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [0, 0, 0]])

            kernel2 = np.array([[0, 0, 0],
                                [1, 2, 1],
                                [0, 0, 0]])
    elif type in ['y', 'Y']:
        # kernel = np.array([[-1, 0, 1],
        #                    [-2, 0, 2],
        #                    [-1, 0, 1]])
        if center:
            kernel1 = np.array([[1, 0, 0],
                                [2, 0, 0],
                                [1, 0, 0]])

            kernel2 = np.array([[0, 0, 1],
                                [0, 0, 2],
                                [0, 0, 1]])
        else:
            kernel1 = np.array([[1, 0, 0],
                                [2, 0, 0],
                                [1, 0, 0]])

            kernel2 = np.array([[0, 1, 0],
                                [0, 2, 0],
                                [0, 1, 0]])

    else:
        raise ValueError('type must be \'x\' or \'y\'')
    cosA = np.cos(2 * aop)
    sinA = np.sin(2 * aop)
    if not sigma is None:
        cosA = gaussian_filter(cosA, sigma=sigma)
        sinA = gaussian_filter(sinA, sigma=sigma)

    # conv_cos = convolve2d(cosA, np.abs(kernel), mode=mode, boundary='symm')
    # conv_sin = convolve2d(sinA, kernel, mode=mode, boundary='symm')
    # aop_sobel = np.arctan(conv_sin/conv_cos) / 2.0
    # aop_sobel[aop_sobel > np.pi / 2] = np.pi - aop_sobel[aop_sobel > np.pi / 2]
    # aop_sobel[aop_sobel < -np.pi / 2] = -np.pi - aop_sobel[aop_sobel < -np.pi / 2]

    ave_cos1 = convolve2d(cosA, kernel1, mode=mode, boundary='fill')
    ave_cos2 = convolve2d(cosA, kernel2, mode=mode, boundary='fill')
    ave_sin1 = convolve2d(sinA, kernel1, mode=mode, boundary='fill')
    ave_sin2 = convolve2d(sinA, kernel2, mode=mode, boundary='fill')
    aop_1 = np.arctan2(ave_sin1, ave_cos1) / 2.0
    aop_2 = np.arctan2(ave_sin2, ave_cos2) / 2.0
    aop_sobel = aop_1 - aop_2
    aop_sobel[aop_sobel > np.pi/2] = np.pi - aop_sobel[aop_sobel > np.pi/2]
    aop_sobel[aop_sobel < -np.pi/2] = -np.pi - aop_sobel[aop_sobel < -np.pi/2]
    return aop_sobel

def sobel_aop(aop, sigma=None, mode='same', center=True):
    return [sobel_aop_single(aop, type='x', sigma=sigma, mode=mode, center=center),
            sobel_aop_single(aop, type='y', sigma=sigma, mode=mode, center=center)]

def gradmag_aop(aop, sigma=None, mode='same'):
    gradients = np.array(sobel_aop(aop, sigma=sigma, mode=mode))
    print(gradients.shape)
    return np.sum(np.abs(gradients), axis=0)

def laplacian_aop(aop, sigma=None, mode='same',diagonal=False):
    if diagonal:
        kernel = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]])
    else:
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
    cosA = np.cos(2 * aop)
    sinA = np.sin(2 * aop)
    if not sigma is None:
        cosA = gaussian_filter(cosA, sigma=sigma)
        sinA = gaussian_filter(sinA, sigma=sigma)
    aop_smooth = np.arctan2(sinA, cosA) / 2.0
    ave_cos = convolve2d(cosA, kernel, mode=mode, boundary='symm')
    ave_sin = convolve2d(sinA, kernel, mode=mode, boundary='symm')
    ave_aop = np.arctan2(ave_sin, ave_cos) / 2.0
    aop_lap = ave_aop - aop_smooth
    aop_lap[aop_lap > np.pi/2] = np.pi - aop_lap[aop_lap > np.pi/2]
    aop_lap[aop_lap < -np.pi/2] = -np.pi - aop_lap[aop_lap < -np.pi/2]
    return aop_lap

def construct_matrix(shape, type="laplacian"):
    ny, nx = shape
    ntot = nx * ny
    if type is "laplacian_diag":

        M = diags([1, 1, 1, 1, -8, 1, 1, 1, 1], offsets=[-1 * (nx - 1),
                                                         -1 * nx,
                                                         -1 * (nx + 1),
                                                         -1,
                                                         0,
                                                         1,
                                                         nx - 1,
                                                         nx,
                                                         nx + 1],
                  shape=(ntot, ntot),
                  format='csc')
    elif type is "laplacian":
        # M = diags([1, 1, -4, 1, 1], offsets=[-1 * nx,
        #                                      -1,
        #                                      0,
        #                                      1,
        #                                      nx],
        #           shape=(ntot, ntot),
        #           format='csc')
        A0_pattern = -4 * np.ones(nx, dtype='int')
        A0_pattern[0] = -3
        A0_pattern[-1] = -3
        A0 = np.tile(A0_pattern, ny)
        A0[:nx] += 1
        A0[-nx:] += 1

        A1_pattern = np.ones(nx, dtype='int')
        A1_pattern[-1] = 0
        A1 = np.tile(A1_pattern, ny)
        A1 = A1[:-1]

        Anx = np.ones(ntot-nx, dtype='int')

        M = diags([A0, A1, A1, Anx, Anx], offsets=[0, 1, -1, nx, -1 * nx], format='csc')

    elif type is 'gradient_x':
        M = diags([1, 2, 1, -1, -2, -1], offsets=[-1,
                                                  0,
                                                  1,
                                                  nx + 1,
                                                  nx,
                                                  nx - 1],
                  shape=(ntot, ntot),
                  format='csc')


        # A0 = np.ones(ntot, dtype='int')
        # # A0[:nx] = 0
        # Anx = np.ones(ntot-nx, dtype='int')
        # M = diags([A0, Anx], offsets=[0, nx], format='csc')

    elif type is 'gradient_y':
        M = diags([1, 2, 1, -1, -2, -1], offsets=[nx,
                                                  0,
                                                  -1 * nx,
                                                  nx + 1,
                                                  1,
                                                  -1 * (nx - 1)],
                  shape=(ntot, ntot),
                  format='csc')
    elif type is 'lap_xy':
        A0_pattern = -2 * np.ones(nx, dtype='int')
        A0_pattern[-1] = -1
        A1_pattern = (A0_pattern + 1) * (-1)
        A0 = np.tile(A0_pattern, ny)
        A0[-nx:] += 1
        A1 = np.tile(A1_pattern, ny)
        A1 = A1[:-1]
        Anx = np.ones(ntot-nx,dtype='int')
        M = diags([A0, A1, Anx], offsets=[0, -1, -1 * nx],
                  shape=(ntot, ntot),
                  format='csc')

    return M

def aop_colors(isoluminant=False):
    if isoluminant:
        c =   [ [  0.7498592323907544, 0.6244911504415356, 0.19579328266660095, 1.0  ],
                [  0.7570286531448105, 0.6209669407230255, 0.19489636610526795, 1.0  ],
                [  0.7640663332364643, 0.6174274223692727, 0.1944113755850764, 1.0  ],
                [  0.7709707936522707, 0.6138752823142762, 0.1943411476057526, 1.0  ],
                [  0.7777405847833383, 0.6103132628434608, 0.19468631993177757, 1.0  ],
                [  0.7843742892407457, 0.6067441589283072, 0.19544533625053018, 1.0  ],
                [  0.7908705246287135, 0.6031708152995389, 0.19661451430428672, 1.0  ],
                [  0.7972279462690779, 0.5995961232543897, 0.1981881731550919, 1.0  ],
                [  0.8034452498702738, 0.5960230171948517, 0.2001588115174065, 1.0  ],
                [  0.8095211741336462, 0.592454470895377, 0.20251732624751057, 1.0  ],
                [  0.8154545032896214, 0.5888934935002493, 0.20525325836404312, 1.0  ],
                [  0.8212440695559989, 0.5853431252527564, 0.20835505346738797, 1.0  ],
                [  0.8268887555104342, 0.5818064329603833, 0.21181032403487343, 1.0  ],
                [  0.8323874963690441, 0.578286505202478, 0.21560610257233742, 1.0  ],
                [  0.8377392821629674, 0.5747864472892329, 0.2197290767055225, 1.0  ],
                [  0.8429431598047259, 0.5713093759833112, 0.22416579968692668, 1.0  ],
                [  0.8479982350362862, 0.5678584139980517, 0.22890287219756417, 1.0  ],
                [  0.8529036742508538, 0.5644366842888564, 0.23392709352377927, 1.0  ],
                [  0.8576587061806591, 0.5610473041570536, 0.23922558204720878, 1.0  ],
                [  0.8622626234432507, 0.5576933791882622, 0.24478586643336941, 1.0  ],
                [  0.8667147839392342, 0.5543779970499292, 0.25059594993287515, 1.0  ],
                [  0.8710146120947766, 0.5511042211753174, 0.25664435085270754, 1.0  ],
                [  0.8751615999427638, 0.5478750843636929, 0.262920122572199, 1.0  ],
                [  0.879155308037057, 0.5446935823287555, 0.2694128565388955, 1.0  ],
                [  0.8829953661949733, 0.5415626672294417, 0.27611267155249775, 1.0  ],
                [  0.8866814740638109, 0.5384852412190528, 0.2830101923929491, 1.0  ],
                [  0.8902134015080375, 0.5354641500501747, 0.29009652052330565, 1.0  ],
                [  0.8935909888145803, 0.5325021767739891, 0.2973631992395541, 1.0  ],
                [  0.8968141467145168, 0.5296020355733643, 0.30480217527748554, 1.0  ],
                [  0.8998828562203629, 0.5267663657694162, 0.31240575854094477, 1.0  ],
                [  0.90279716827912, 0.5239977260410834, 0.32016658129854936, 1.0  ],
                [  0.9055572032421312, 0.5212985888966478, 0.3280775579137081, 1.0  ],
                [  0.908163150153801, 0.5186713354349658, 0.33613184592771284, 1.0  ],
                [  0.9106152658621508, 0.5161182504325307, 0.34432280910724644, 1.0  ],
                [  0.9129138739551323, 0.5136415177903106, 0.3526439828934496, 1.0  ],
                [  0.9150593635275486, 0.5112432163716227, 0.3610890425464449, 1.0  ],
                [  0.9170521877843079, 0.508925316259141, 0.36965177416313194, 1.0  ],
                [  0.918892862486586, 0.5066896754555509, 0.37832604865337915, 1.0  ],
                [  0.9205819642483014, 0.5045380370483127, 0.3871057986868544, 1.0  ],
                [  0.9221201286910256, 0.5024720268546662, 0.395984998566347, 1.0  ],
                [  0.9235080484661594, 0.5004931515583477, 0.40495764694069875, 1.0  ],
                [  0.9247464711538455, 0.49860279734460594, 0.41401775223879855, 1.0  ],
                [  0.9258361970485892, 0.4968022290351493, 0.4231593206834608, 1.0  ],
                [  0.9267780768421002, 0.495092589719536, 0.43237634672854425, 1.0  ],
                [  0.9275730092142082, 0.4934749008745544, 0.441662805752967, 1.0  ],
                [  0.9282219383430441, 0.4919500629582026, 0.4510126488400689, 1.0  ],
                [  0.9287258513459065, 0.4905188564601914, 0.4604197994691199, 1.0  ],
                [  0.9290857756623627, 0.4891819433864862, 0.46987815194686416, 1.0  ],
                [  0.9293027763912073, 0.4879398691513643, 0.4793815714101631, 1.0  ],
                [  0.929377953592864, 0.4867930648468511, 0.48892389523558927, 1.0  ],
                [  0.9293124395687187, 0.48574184985630325, 0.4984989356977541, 1.0  ],
                [  0.9291073961286734, 0.4847864347763245, 0.5081004837249762, 1.0  ],
                [  0.928764011857963, 0.48392692460923764, 0.5177223136082753, 1.0  ],
                [  0.928283499393951, 0.48316332218692576, 0.5273581885274684, 1.0  ],
                [  0.9276670927231875, 0.4824955317861474, 0.5370018667661602, 1.0  ],
                [  0.9269160445085952, 0.48192336289522003, 0.5466471084955188, 1.0  ],
                [  0.9260316234561239, 0.4814465340924656, 0.5562876830148501, 1.0  ],
                [  0.9250151117296442, 0.4810646769977915, 0.5659173763450085, 1.0  ],
                [  0.9238678024222369, 0.4807773402603763, 0.5755299990785734, 1.0  ],
                [  0.9225909970914559, 0.4805839935473731, 0.5851193943984278, 1.0  ],
                [  0.9211860033653898, 0.4804840315010929, 0.5946794461838506, 1.0  ],
                [  0.9196541326257351, 0.4804767776348509, 0.6042040871304685, 1.0  ],
                [  0.9179966977733609, 0.4805614881407901, 0.6136873068173614, 1.0  ],
                [  0.9162150110811165, 0.4807373555863243, 0.6231231596612942, 1.0  ],
                [  0.9143103821379763, 0.48100351247924894, 0.6325057727044077, 1.0  ],
                [  0.9122841158878593, 0.4813590346851519, 0.6418293531877762, 1.0  ],
                [  0.9101375107657809, 0.4818029446842604, 0.6510881958689992, 1.0  ],
                [  0.9078718569333344, 0.48233421465834864, 0.6602766900474286, 1.0  ],
                [  0.9054884346148117, 0.4829517694016829, 0.6693893262657941, 1.0  ],
                [  0.9029885125346536, 0.4836544890531828, 0.6784207026618019, 1.0  ],
                [  0.9003733464563215, 0.4844412116499232, 0.687365530947826, 1.0  ],
                [  0.89764417782208, 0.4853107355048521, 0.6962186420010316, 1.0  ],
                [  0.8948022324926893, 0.48626182141399404, 0.70497499105022, 1.0  ],
                [  0.8918487195854597, 0.48729319470056637, 0.7136296624493308, 1.0  ],
                [  0.8887848304086944, 0.48840354710521877, 0.722177874030913, 1.0  ],
                [  0.8856117374901107, 0.48959153853309606, 0.730614981035985, 1.0  ],
                [  0.8823305936964492, 0.49085579866957335, 0.7389364796195477, 1.0  ],
                [  0.8789425314411664, 0.49219492847734553, 0.747138009933602, 1.0  ],
                [  0.8754486619767842, 0.4936075015880755, 0.7552153587918732, 1.0  ],
                [  0.8718500747682392, 0.4950920656020486, 0.7631644619225588, 1.0  ],
                [  0.8681478369433496, 0.49664714330925036, 0.7709814058173095, 1.0  ],
                [  0.8643429928163562, 0.498271233845007, 0.7786624291863239, 1.0  ],
                [  0.8604365634803437, 0.49996281379284896, 0.7862039240309222, 1.0  ],
                [  0.8564295464642733, 0.5017203382465807, 0.7936024363462338, 1.0  ],
                [  0.8523229154502532, 0.5035422418427206, 0.8008546664677494, 1.0  ],
                [  0.8481176200467055, 0.505426939773498, 0.807957469076397, 1.0  ],
                [  0.8438145856129933, 0.5073728287895908, 0.8149078528775845, 1.0  ],
                [  0.8394147131311873, 0.5093782882006059, 0.8217029799702565, 1.0  ],
                [  0.8349188791206409, 0.5114416808802021, 0.8283401649224925, 1.0  ],
                [  0.8303279355911178, 0.5135613542815511, 0.8348168735705165, 1.0  ],
                [  0.8256427100303302, 0.5157356414676671, 0.8411307215582109, 1.0  ],
                [  0.820864005421822, 0.5179628621599943, 0.8472794726343406, 1.0  ],
                [  0.8159926002892787, 0.5202413238075401, 0.853261036724704, 1.0  ],
                [  0.8110292487634376, 0.5225693226778217, 0.8590734677963396, 1.0  ],
                [  0.8059746806679561, 0.524945144969887, 0.8647149615307677, 1.0  ],
                [  0.8008296016207078, 0.527367067948828, 0.8701838528229795, 1.0  ],
                [  0.7955946931471439, 0.5298333611003828, 0.8754786131226046, 1.0  ],
                [  0.7902706128025195, 0.532342287303505, 0.8805978476333035, 1.0  ],
                [  0.7848579942999026, 0.5348921040182204, 0.8855402923860154, 1.0  ],
                [  0.7793574476410902, 0.5374810644855122, 0.8903048112012351, 1.0  ],
                [  0.7737695592476568, 0.5401074189356194, 0.89489039255498, 1.0  ],
                [  0.7680948920895356, 0.5427694158007719, 0.8992961463625836, 1.0  ],
                [  0.7623339858086807, 0.5454653029281575, 0.9035213006938838, 1.0  ],
                [  0.7564873568354312, 0.5481933287887961, 0.9075651984327995, 1.0  ],
                [  0.7505554984954228, 0.55095174367787, 0.9114272938936917, 1.0  ],
                [  0.7445388811049036, 0.5537388009021121, 0.9151071494062892, 1.0  ],
                [  0.7384379520524739, 0.5565527579498898, 0.9186044318803648, 1.0  ],
                [  0.7322531358653633, 0.5593918776397413, 0.9219189093607099, 1.0  ],
                [  0.7259848342583946, 0.562254429243317, 0.9250504475823513, 1.0  ],
                [  0.7196334261638904, 0.5651386895788701, 0.9279990065353442, 1.0  ],
                [  0.7131992677408022, 0.5680429440717096, 0.9307646370478632, 1.0  ],
                [  0.7066826923614049, 0.570965487778292, 0.9333474773957248, 1.0  ],
                [  0.7000840105738754, 0.5739046263709525, 0.935747749945889, 1.0  ],
                [  0.6934035100391257, 0.5768586770805726, 0.9379657578409188, 1.0  ],
                [  0.6866414554402045, 0.5798259695948317, 0.9400018817308208, 1.0  ],
                [  0.679798088362579, 0.5828048469100093, 0.9418565765581622, 1.0  ],
                [  0.6728736271435144, 0.5857936661346557, 0.9435303684018231, 1.0  ],
                [  0.6658682666887529, 0.588790799243764, 0.9450238513842634, 1.0  ],
                [  0.6587821782545257, 0.591794633782428, 0.9463376846466919, 1.0  ],
                [  0.6516155091928713, 0.594803573518261, 0.9474725893960668, 1.0  ],
                [  0.6443683826580705, 0.5978160390421652, 0.9484293460274259, 1.0  ],
                [  0.6370408972717966, 0.6008304683173463, 0.9492087913246171, 1.0  ],
                [  0.629633126744443, 0.60384531717671, 0.9498118157421122, 1.0  ],
                [  0.6221451194498211, 0.606859059769057, 0.9502393607702103, 1.0  ],
                [  0.614576897950145, 0.6098701889547189, 0.9504924163855847, 1.0  ],
                [  0.6069284584679434, 0.6128772166515039, 0.9505720185887817, 1.0  ],
                [  0.5991997703011737, 0.6158786741320035, 0.950479247029996, 1.0  ],
                [  0.5913907751774268, 0.6188731122735118, 0.9502152227241204, 1.0  ],
                [  0.5835013865426864, 0.6218591017619443, 0.9497811058558229, 1.0  ],
                [  0.5755314887795705, 0.6248352332513107, 0.9491780936751336, 1.0  ],
                [  0.5674809363494633, 0.6278001174803942, 0.9484074184837968, 1.0  ],
                [  0.5593495528523041, 0.630752385348409, 0.9474703457124204, 1.0  ],
                [  0.551137129997051, 0.6336906879514969, 0.9463681720882634, 1.0  ],
                [  0.5428434264751265, 0.6366136965819843, 0.9451022238933015, 1.0  ],
                [  0.5344681667281421, 0.6395201026923913, 0.943673855312065, 1.0  ],
                [  0.5260110396002784, 0.6424086178262264, 0.9420844468685659, 1.0  ],
                [  0.5174716968644791, 0.6452779735176251, 0.9403354039515072, 1.0  ],
                [  0.5088497516104091, 0.6481269211619131, 0.9384281554268339, 1.0  ],
                [  0.5001447764805359, 0.6509542318591902, 0.9363641523365808, 1.0  ],
                [  0.49135630173918726, 0.6537586962330234, 0.9341448666828512, 1.0  ],
                [  0.48248381315743477, 0.6565391242263215, 0.9317717902956898, 1.0  ],
                [  0.4735267496945719, 0.659294344876474, 0.929246433783525, 1.0  ],
                [  0.4644845009545499, 0.6620232060717678, 0.9265703255647825, 1.0  ],
                [  0.45535640439291764, 0.6647245742911161, 0.9237450109792059, 1.0  ],
                [  0.44614174224670017, 0.6673973343290613, 0.920772051477392, 1.0  ],
                [  0.4368397381560678, 0.6700403890079899, 0.9176530238869608, 1.0  ],
                [  0.42744955344253266, 0.6726526588794548, 0.9143895197537815, 1.0  ],
                [  0.4179702830037521, 0.6752330819164539, 0.9109831447566188, 1.0  ],
                [  0.40840095077965904, 0.6777806131984613, 0.9074355181935467, 1.0  ],
                [  0.39874050473857564, 0.6802942245909694, 0.9037482725384539, 1.0  ],
                [  0.38898781132505944, 0.6827729044212347, 0.8999230530659508, 1.0  ],
                [  0.3791416493032463, 0.685215657151881, 0.895961517542968, 1.0  ],
                [  0.3692007029206848, 0.6876215030539605, 0.8918653359853408, 1.0  ],
                [  0.35916355430752, 0.689989477881009, 0.8876361904776573, 1.0  ],
                [  0.3490286750146748, 0.6923186325456057, 0.8832757750546527, 1.0  ],
                [  0.3387944165824283, 0.6946080327998686, 0.8787857956424312, 1.0  ],
                [  0.3284590000174053, 0.6968567589212913, 0.8741679700577984, 1.0  ],
                [  0.31802050404209437, 0.699063905405262, 0.8694240280639967, 1.0  ],
                [  0.3074768519672481, 0.7012285806655723, 0.8645557114811383, 1.0  ],
                [  0.2968257970251618, 0.7033499067441631, 0.8595647743496404, 1.0  ],
                [  0.28606490599345025, 0.7054270190313252, 0.854452983144983, 1.0  ],
                [  0.2751915409381847, 0.7074590659975201, 0.849222117042107, 1.0  ],
                [  0.2642028389194089, 0.7094452089379513, 0.8438739682278041, 1.0  ],
                [  0.2530956895420039, 0.7113846217309736, 0.8384103422594378, 1.0  ],
                [  0.24186671031994678, 0.713276490611396, 0.8328330584683782, 1.0  ],
                [  0.23051221998436272, 0.7151200139596879, 0.8271439504065246, 1.0  ],
                [  0.21902821015986224, 0.7169144021080827, 0.8213448663343281, 1.0  ],
                [  0.2074103163535253, 0.7186588771645127, 0.8154376697487288, 1.0  ],
                [  0.1956537901094369, 0.7203526728553188, 0.8094242399494607, 1.0  ],
                [  0.18375347576897208, 0.7219950343875979, 0.8033064726421918, 1.0  ],
                [  0.1717037980639315, 0.7235852183320733, 0.797086280576995, 1.0  ],
                [  0.15949877172136964, 0.7251224925273116, 0.7907655942206817, 1.0  ],
                [  0.14713205320022016, 0.7266061360061016, 0.7843463624615612, 1.0  ],
                [  0.1345970711716276, 0.7280354389447777, 0.7778305533452297, 1.0  ],
                [  0.1218873035726025, 0.7294097026362567, 0.771220154840043, 1.0  ],
                [  0.10899683001954244, 0.7307282394875149, 0.764517175630973, 1.0  ],
                [  0.09592141182212541, 0.7319903730422384, 0.757723645940621, 1.0  ],
                [  0.08266061272374078, 0.733195438029328, 0.7508416183762202, 1.0  ],
                [  0.06922205241627152, 0.7343427804379418, 0.7438731688015521, 1.0  ],
                [  0.05563024234800595, 0.7354317576197188, 0.7368203972327939, 1.0  ],
                [  0.04194583131315057, 0.7364617384188239, 0.7296854287574244, 1.0  ],
                [  0.02918296972466305, 0.737432103330413, 0.7224704144754511, 1.0  ],
                [  0.0192519622788623, 0.7383422446881096, 0.715177532462358, 1.0  ],
                [  0.012217624138274692, 0.7391915668810543, 0.7078089887533602, 1.0  ],
                [  0.008101473474541012, 0.7399794866010684, 0.7003670183487392, 1.0  ],
                [  0.0069239937458705205, 0.7407054331204451, 0.6928538862402704, 1.0  ],
                [  0.008704595012929408, 0.7413688486008522, 0.6852718884590062, 1.0  ],
                [  0.013461574075962766, 0.7419691884338149, 0.6776233531449894, 1.0  ],
                [  0.02121207345293915, 0.742505921613211, 0.6699106416398145, 1.0  ],
                [  0.03197203922309695, 0.742978531140174, 0.6621361496033372, 1.0  ],
                [  0.045480195183905094, 0.7433865144607753, 0.6543023081563087, 1.0  ],
                [  0.05948146780353868, 0.74372938393683, 0.6464115850511868, 1.0  ],
                [  0.0734214956682514, 0.7440066673501079, 0.6384664858739955, 1.0  ],
                [  0.08724739392266684, 0.7442179084402247, 0.6304695552807323, 1.0  ],
                [  0.10093752473681436, 0.7443626674764168, 0.6224233782725909, 1.0  ],
                [  0.11448497236741367, 0.7444405218633838, 0.6143305815151041, 1.0  ],
                [  0.12788986903108807, 0.7444510667813117, 0.6061938347072604, 1.0  ],
                [  0.14115566466911786, 0.7443939158601605, 0.5980158520077258, 1.0  ],
                [  0.15428725239100827, 0.7442687018882292, 0.5897993935265111, 1.0  ],
                [  0.1672900079612796, 0.7440750775549649, 0.5815472668917623, 1.0  ],
                [  0.18016929639171195, 0.7438127162279146, 0.5732623289028856, 1.0  ],
                [  0.19293022398797333, 0.7434813127636573, 0.564947487282895, 1.0  ],
                [  0.20557752180820654, 0.7430805843524895, 0.5566057025447693, 1.0  ],
                [  0.21811550006456137, 0.7426102713965556, 0.5482399899886929, 1.0  ],
                [  0.23054804063636242, 0.7420701384210415, 0.5398534218493961, 1.0  ],
                [  0.24287860955890356, 0.7414599750179873, 0.531449129615382, 1.0  ],
                [  0.2551102793659818, 0.7407795968221503, 0.5230303065446865, 1.0  ],
                [  0.26724575562499814, 0.740028846518318, 0.5146002104049393, 1.0  ],
                [  0.2792874045268829, 0.7392075948793155, 0.5061621664689521, 1.0  ],
                [  0.291237279835625, 0.7383157418339097, 0.4977195708008016, 1.0  ],
                [  0.30309714833189655, 0.7373532175636786, 0.48927589387147447, 1.0  ],
                [  0.3148685133615493, 0.7363199836278163, 0.480834684547569, 1.0  ],
                [  0.3265526363702425, 0.7352160341147439, 0.47239957450130093, 1.0  ],
                [  0.3381505564554831, 0.7340413968192654, 0.46397428309515226, 1.0  ],
                [  0.34966310804612527, 0.732796134443908, 0.45556262279988363, 1.0  ],
                [  0.3610909368569204, 0.7314803458229416, 0.4471685052102677, 1.0  ],
                [  0.3724345142788335, 0.7300941671674682, 0.43879594772872527, 1.0  ],
                [  0.38369415036562016, 0.7286377733298155, 0.43044908099295914, 1.0  ],
                [  0.39487000556943497, 0.7271113790853518, 0.4221321571295366, 1.0  ],
                [  0.40596210136693517, 0.7255152404296802, 0.413849558920984, 1.0  ],
                [  0.4169703299046073, 0.7238496558890337, 0.40560580997906814, 1.0  ],
                [  0.4278944627791229, 0.722114967841541, 0.3974055860211868, 1.0  ],
                [  0.4387341590561512, 0.7203115638468635, 0.3892537273497976, 1.0  ],
                [  0.4494889726195785, 0.7184398779815501, 0.38115525263588285, 1.0  ],
                [  0.46015835893249873, 0.7165003921772892, 0.373115374106015, 1.0  ],
                [  0.4707416812820931, 0.7144936375590446, 0.3651395142275779, 1.0  ],
                [  0.48123821657193505, 0.7124201957799057, 0.35723332397709734, 1.0  ],
                [  0.4916471607179577, 0.7102807003492668, 0.3494027027609444, 1.0  ],
                [  0.5019676336976988, 0.7080758379507683, 0.34165382003425726, 1.0  ],
                [  0.5121986842966226, 0.7058063497462285, 0.33399313863066177, 1.0  ],
                [  0.5223392945902883, 0.7034730326615682, 0.3264274397698838, 1.0  ],
                [  0.5323883841965565, 0.7010767406505294, 0.3189638496497775, 1.0  ],
                [  0.5423448143281642, 0.6986183859317292, 0.3116098674504624, 1.0  ],
                [  0.5522073916723983, 0.6960989401943825, 0.3043733944776538, 1.0  ],
                [  0.5619748721216135, 0.6935194357677433, 0.29726276404614793, 1.0  ],
                [  0.5716459643755083, 0.69088096674908, 0.29028677154914306, 1.0  ],
                [  0.5812193334337143, 0.6881846900847075, 0.28345470397129613, 1.0  ],
                [  0.59069360399505, 0.6854318265983241, 0.2767763678807944, 1.0  ],
                [  0.600067363777833, 0.6826236619606196, 0.2702621146775312, 1.0  ],
                [  0.6093391667739643, 0.6797615475937934, 0.26392286158275025, 1.0  ],
                [  0.6185075364478619, 0.6768469015043368, 0.257770106536228, 1.0  ],
                [  0.6275709688899546, 0.6738812090370998, 0.2518159348318334, 1.0  ],
                [  0.6365279359331191, 0.6708660235433419, 0.2460730149899979, 1.0  ],
                [  0.6453768882392943, 0.667802966955148, 0.24055458106428057, 1.0  ],
                [  0.6541162583623863, 0.6646937302582531, 0.23527439834738773, 1.0  ],
                [  0.6627444637926152, 0.6615400738550072, 0.23024670932921898, 1.0  ],
                [  0.6712599099864974, 0.6583438278088877, 0.22548615682569528, 1.0  ],
                [  0.6796609933857799, 0.6551068919616801, 0.22100768150882705, 1.0  ],
                [  0.6879461044278744, 0.651831235914148, 0.21682639169284934, 1.0  ],
                [  0.6961136305494715, 0.6485188988607871, 0.21295740422546333, 1.0  ],
                [  0.7041619591843893, 0.6451719892690102, 0.2094156567307106, 1.0  ],
                [  0.71208948075592, 0.6417926843929554, 0.20621569324368932, 1.0  ],
                [  0.7198945916633023, 0.6383832296119835, 0.203371427402883, 1.0  ],
                [  0.7275756972612778, 0.6349459375838766, 0.2008958896891307, 1.0  ],
                [  0.7351312148310842, 0.6314831872027568, 0.1988009675133819, 1.0  ],
                [  0.7425595765405989, 0.6279974223518645, 0.19709714898819336, 1.0  ] ]

    else:
        c =   [ [ 0.950139716668437, 0.7988733088110443, 0.48237219386703917, 1.0 ] ,
                [ 0.9448875491188792, 0.8050650707407802, 0.48804974326621525, 1.0 ] ,
                [ 0.9393934113480029, 0.8111167985886586, 0.493876842553537, 1.0 ] ,
                [ 0.933663045687964, 0.8170221665984069, 0.4998441881699473, 1.0 ] ,
                [ 0.9277026782525013, 0.8227750129127813, 0.5059418414950936, 1.0 ] ,
                [ 0.9215189879665214, 0.8283693669681744, 0.5121592426183297, 1.0 ] ,
                [ 0.9151190656616458, 0.8337994786966255, 0.5184852326694709, 1.0 ] ,
                [ 0.9085103628217575, 0.839059849151053, 0.524908085914121, 1.0 ] ,
                [ 0.901700629882155, 0.8441452620443818, 0.5314155526582272, 1.0 ] ,
                [ 0.8946978443707027, 0.8490508155702773, 0.5379949137576794, 1.0 ] ,
                [ 0.8875101296176124, 0.853771953760682, 0.5446330471908828, 1.0 ] ,
                [ 0.8801456652337694, 0.8583044965422681, 0.5513165067319495, 1.0 ] ,
                [ 0.8726125910407019, 0.8626446675896454, 0.5580316122736925, 1.0 ] ,
                [ 0.8649189065977811, 0.8667891190462672, 0.5647645508151067, 1.0 ] ,
                [ 0.8570723688801313, 0.8707349522014484, 0.5715014865765766, 1.0 ] ,
                [ 0.8490803909791392, 0.8744797332783164, 0.5782286781723013, 1.0 ] ,
                [ 0.8409499448943548, 0.8780215036037674, 0.5849326002908835, 1.0 ] ,
                [ 0.8326874715352314, 0.8813587835949632, 0.5916000669487862, 1.0 ] ,
                [ 0.8242988009374191, 0.8844905702004637, 0.5982183531205978, 1.0 ] ,
                [ 0.8157890854173421, 0.8874163276671959, 0.6047753114402691, 1.0 ] ,
                [ 0.8071627479505898, 0.8901359717532465, 0.6112594807236709, 1.0 ] ,
                [ 0.7984234474876016, 0.8926498477553796, 0.6176601832872111, 1.0 ] ,
                [ 0.7895740622496601, 0.8949587029531834, 0.6239676084187684, 1.0 ] ,
                [ 0.780616691323439, 0.8970636542738591, 0.6301728798722639, 1.0 ] ,
                [ 0.7715526741417428, 0.8989661521404468, 0.6362681058717365, 1.0 ] ,
                [ 0.7623826267494108, 0.9006679415731241, 0.6422464107839106, 1.0 ] ,
                [ 0.7531064931503307, 0.9021710216634532, 0.6481019483059594, 1.0 ] ,
                [ 0.7437236095473532, 0.9034776045354684, 0.6538298966747415, 1.0 ] ,
                [ 0.734232778944999, 0.9045900748494846, 0.6594264369975218, 1.0 ] ,
                [ 0.7246323533945486, 0.9055109508023103, 0.6648887163022013, 1.0 ] ,
                [ 0.7149203211200136, 0.9062428474414473, 0.6702147972874641, 1.0 ] ,
                [ 0.7050943958583127, 0.9067884429523888, 0.6754035970103054, 1.0 ] ,
                [ 0.6951521059560795, 0.9071504484088762, 0.6804548168804784, 1.0 ] ,
                [ 0.6850908810624041, 0.9073315813068072, 0.6853688663472655, 1.0 ] ,
                [ 0.6749081346122469, 0.90733454304256, 0.6901467825789179, 1.0 ] ,
                [ 0.664601340681604, 0.9071620003528429, 0.6947901482686001, 1.0 ] ,
                [ 0.6541681041869382, 0.9068165706107503, 0.6993010094741359, 1.0 ] ,
                [ 0.6436062237773086, 0.9063008107740042, 0.7036817951336762, 1.0 ] ,
                [ 0.6329137471120523, 0.9056172097071068, 0.7079352396153203, 1.0 ] ,
                [ 0.6220890185191964, 0.9047681835483997, 0.712064309372786, 1.0 ] ,
                [ 0.6111307192845611, 0.903756073763628, 0.7160721345050882, 1.0 ] ,
                [ 0.6000379010274225, 0.9025831475166843, 0.719961945766039, 1.0 ] ,
                [ 0.5888100127777173, 0.9012515999924282, 0.7237370173459031, 1.0 ] ,
                [ 0.577446922486872, 0.8997635583225075, 0.727400615556491, 1.0 ] ,
                [ 0.5659489337858802, 0.8981210867896238, 0.730955953393435, 1.0 ] ,
                [ 0.5543167988576285, 0.8963261930158888, 0.7344061508245764, 1.0 ] ,
                [ 0.5425517283234872, 0.8943808348741954, 0.7377542005590504, 1.0 ] ,
                [ 0.5306553990649511, 0.8922869278959515, 0.7410029389845495, 1.0 ] ,
                [ 0.5186299609172738, 0.8900463529823676, 0.7441550219167605, 1.0 ] ,
                [ 0.5064780431912413, 0.8876609642587174, 0.7472129047811882, 1.0 ] ,
                [ 0.494202762008874, 0.8851325969406759, 0.750178826839761, 1.0 ] ,
                [ 0.48180772948661116, 0.8824630751085847, 0.7530547990792857, 1.0 ] ,
                [ 0.46929706587296355, 0.8796542193090001, 0.7558425953927552, 1.0 ] ,
                [ 0.45667541585580645, 0.8767078539231276, 0.7585437467051129, 1.0 ] ,
                [ 0.4439479704071821, 0.8736258142588161, 0.7611595377199643, 1.0 ] ,
                [ 0.43112049574267347, 0.8704099533368914, 0.7636910059910879, 1.0 ] ,
                [ 0.41819937125315954, 0.8670621483539648, 0.7661389430509519, 1.0 ] ,
                [ 0.4051916386373621, 0.8635843068128013, 0.7685038973566605, 1.0 ] ,
                [ 0.3921050649479822, 0.8599783723181221, 0.7707861788409835, 1.0 ] ,
                [ 0.378948222892442, 0.8562463300406665, 0.7729858648817978, 1.0 ] ,
                [ 0.3657305925396398, 0.8523902118557461, 0.775102807526988, 1.0 ] ,
                [ 0.35246268962547733, 0.8484121011645934, 0.7771366418333885, 1.0 ] ,
                [ 0.3391562269820293, 0.8443141374078511, 0.7790867951975934, 1.0 ] ,
                [ 0.32582431731227773, 0.8400985202807035, 0.780952497573402, 1.0 ] ,
                [ 0.3124817276792861, 0.8357675136586433, 0.7827327924853567, 1.0 ] ,
                [ 0.29914519876974666, 0.8313234492418435, 0.784426548760377, 1.0 ] ,
                [ 0.2858338453140956, 0.8267687299246921, 0.7860324729099992, 1.0 ] ,
                [ 0.27256965805122574, 0.8221058328953771, 0.7875491221043894, 1.0 ] ,
                [ 0.2593781322699165, 0.8173373124685616, 0.7889749176862233, 1.0 ] ,
                [ 0.24628905298004278, 0.8124658026522663, 0.7903081591778959, 1.0 ] ,
                [ 0.2333374714682983, 0.8074940194481102, 0.7915470387395136, 1.0 ] ,
                [ 0.2205649108530648, 0.8024247628821696, 0.7926896560378278, 1.0 ] ,
                [ 0.20802083626673845, 0.7972609187618641, 0.7937340334879126, 1.0 ] ,
                [ 0.19576441292528407, 0.7920054601525911, 0.7946781318300405, 1.0 ] ,
                [ 0.18386654304048755, 0.7866614485662771, 0.795519866004024, 1.0 ] ,
                [ 0.17241210500725337, 0.7812320348526475, 0.7962571212824028, 1.0 ] ,
                [ 0.16150219369716295, 0.7757204597828636, 0.7968877696223385, 1.0 ] ,
                [ 0.1512559536098376, 0.7701300543142398, 0.7974096861940954, 1.0 ] ,
                [ 0.141811291505832, 0.7644642395240614, 0.7978207660415628, 1.0 ] ,
                [ 0.1333233798548732, 0.7587265262000851, 0.7981189408275832, 1.0 ] ,
                [ 0.12595954815712335, 0.7529205140751212, 0.7983021956139331, 1.0 ] ,
                [ 0.11988920324794972, 0.7470498906931676, 0.798368585622743, 1.0 ] ,
                [ 0.11526825763671833, 0.741118429894948, 0.7983162529230774, 1.0 ] ,
                [ 0.11221947256881021, 0.7351299899113027, 0.7981434429833217, 1.0 ] ,
                [ 0.11081273848899598, 0.7290885110537818, 0.7978485210270917, 1.0 ] ,
                [ 0.1110511571622983, 0.7229980129929465, 0.7974299881276267, 1.0 ] ,
                [ 0.11286804031105466, 0.7168625916162845, 0.7968864969731072, 1.0 ] ,
                [ 0.11613621518771589, 0.7106864154592922, 0.7962168672331766, 1.0 ] ,
                [ 0.1206863613053164, 0.7044737217051578, 0.7954201004551085, 1.0 ] ,
                [ 0.12632840743953333, 0.6982288117505459, 0.794495394416713, 1.0 ] ,
                [ 0.13287049398539094, 0.6919560463372773, 0.7934421568621488, 1.0 ] ,
                [ 0.14013249746669051, 0.6856598402521037, 0.7922600185464709, 1.0 ] ,
                [ 0.1479536959788616, 0.6793446565993574, 0.7909488455148833, 1.0 ] ,
                [ 0.1561957083510281, 0.673015000653927, 0.789508750543465, 1.0 ] ,
                [ 0.16474230438713558, 0.6666754133047404, 0.7879401036694719, 1.0 ] ,
                [ 0.1734974952503692, 0.660330464101723, 0.7862435417412847, 1.0 ] ,
                [ 0.18238290061414678, 0.6539847439219618, 0.7844199769206103, 1.0 ] ,
                [ 0.19133499603606471, 0.647642857273573, 0.7824706040726856, 1.0 ] ,
                [ 0.20030255522488033, 0.6413094142583748, 0.7803969069838717, 1.0 ] ,
                [ 0.20924441938169652, 0.6349890222170917, 0.7782006633502183, 1.0 ] ,
                [ 0.21812762317035234, 0.628686277083156, 0.7758839484851533, 1.0 ] ,
                [ 0.22692585580829705, 0.6224057544734203, 0.7734491376994516, 1.0 ] ,
                [ 0.23561821475312866, 0.6161520005461091, 0.7708989073118939, 1.0 ] ,
                [ 0.24418820445367823, 0.6099295226581112, 0.7682362342544936, 1.0 ] ,
                [ 0.2526229353050245, 0.603742779855244, 0.7654643942417401, 1.0 ] ,
                [ 0.2609124836976765, 0.597596173230404, 0.7625869584788558, 1.0 ] ,
                [ 0.2690493805257731, 0.5914940361855132, 0.7596077888894924, 1.0 ] ,
                [ 0.27702820161905944, 0.585440624633936, 0.756531031848479, 1.0 ] ,
                [ 0.28484523886080665, 0.5794401071805771, 0.7533611104100597, 1.0 ] ,
                [ 0.2924982351536873, 0.5734965553171769, 0.7501027150264002, 1.0 ] ,
                [ 0.29998616995548505, 0.5676139336705027, 0.7467607927549476, 1.0 ] ,
                [ 0.30730908494055953, 0.5617960903411987, 0.7433405349563101, 1.0 ] ,
                [ 0.31446794157761987, 0.5560467473711053, 0.7398473634867544, 1.0 ] ,
                [ 0.32146450416572026, 0.5503694913769492, 0.7362869153910163, 1.0 ] ,
                [ 0.32830124323889937, 0.5447677643885693, 0.7326650261020055, 1.0 ] ,
                [ 0.33498125531735645, 0.5392448549303673, 0.7289877111541211, 1.0 ] ,
                [ 0.3415081958154447, 0.5338038893855641, 0.7252611464164267, 1.0 ] ,
                [ 0.34788622256557733, 0.5284478236842697, 0.7214916468509904, 1.0 ] ,
                [ 0.354119947922973, 0.5231794353584189, 0.7176856438005578, 1.0 ] ,
                [ 0.3602143978102901, 0.5180013160094342, 0.7138496608086529, 1.0 ] ,
                [ 0.366174976367848, 0.5129158642381562, 0.7099902879746862, 1.0 ] ,
                [ 0.3720074351131593, 0.5079252790912256, 0.7061141548471029, 1.0 ] ,
                [ 0.37771784569746986, 0.5030315540837691, 0.7022279018597521, 1.0 ] ,
                [ 0.38331257548830405, 0.49823647186495107, 0.6983381503211049, 1.0 ] ,
                [ 0.3887982653147766, 0.4935415996006591, 0.6944514709735894, 1.0 ] ,
                [ 0.39418180879357684, 0.4889482851562001, 0.6905743511519493, 1.0 ] ,
                [ 0.3994703327144069, 0.48445765417114456, 0.68671316058615, 1.0 ] ,
                [ 0.4046711780086397, 0.4800706081281207, 0.6828741159168481, 1.0 ] ,
                [ 0.40979188085892065, 0.47578782352693566, 0.6790632440206876, 1.0 ] ,
                [ 0.4148401535337223, 0.47160975228433827, 0.6752863442794446, 1.0 ] ,
                [ 0.41982386455324366, 0.4675366234873864, 0.6715489499718205, 1.0 ] ,
                [ 0.4247510178149388, 0.46356844663375635, 0.66785628901972, 1.0 ] ,
                [ 0.4296297303311563, 0.4597050164947007, 0.6642132443818414, 1.0 ] ,
                [ 0.43446820826130356, 0.45594591973442394, 0.6606243144556238, 1.0 ] ,
                [ 0.43927472095908765, 0.4522905434125147, 0.6570935739224643, 1.0 ] ,
                [ 0.44405757280416214, 0.4487380854825898, 0.6536246355484432, 1.0 ] ,
                [ 0.4488250726489476, 0.4452875673795094, 0.6502206135303917, 1.0 ] ,
                [ 0.45358550078655896, 0.4419378487587382, 0.6468840890511256, 1.0 ] ,
                [ 0.4583470734350495, 0.43868764441416985, 0.6436170787732505, 1.0 ] ,
                [ 0.4631179048357856, 0.4355355433550526, 0.6404210070528376, 1.0 ] ,
                [ 0.46790596717753324, 0.43248002996915874, 0.6372966826867668, 1.0 ] ,
                [ 0.4727190486795651, 0.42951950713913634, 0.634244281014993, 1.0 ] ,
                [ 0.4775647102917961, 0.42665232111405327, 0.6312633321762222, 1.0 ] ,
                [ 0.4824502415917748, 0.42387678787093575, 0.6283527162583327, 1.0 ] ,
                [ 0.48738261657038023, 0.42119122063477643, 0.6255106659906691, 1.0 ] ,
                [ 0.49236845009278746, 0.41859395816359535, 0.6227347774934877, 1.0 ] ,
                [ 0.4974139558913527, 0.4160833933514623, 0.6200220294320943, 1.0 ] ,
                [ 0.5025249069858995, 0.41365800166060235, 0.617368810724049, 1.0 ] ,
                [ 0.5077065994290039, 0.41131636886728623, 0.6147709567242127, 1.0 ] ,
                [ 0.5129638202362544, 0.40905721759765784, 0.612223793573792, 1.0 ] ,
                [ 0.5183008202830484, 0.40687943214084055, 0.6097221901570612, 1.0 ] ,
                [ 0.5237212928324508, 0.40478208105804503, 0.6072606168754104, 1.0 ] ,
                [ 0.5292283582077348, 0.40276443715719534, 0.6048332102351219, 1.0 ] ,
                [ 0.5348245549457477, 0.4008259944707994, 0.6024338420644584, 1.0 ] ,
                [ 0.5405118375726267, 0.39896648195708395, 0.6000561920369358, 1.0 ] ,
                [ 0.5462915809421742, 0.3971858737366958, 0.5976938220883378, 1.0 ] ,
                [ 0.552164590880442, 0.39548439577471123, 0.5953402512791347, 1.0 ] ,
                [ 0.5581311206982165, 0.39386252901529206, 0.5929890296721644, 1.0 ] ,
                [ 0.5641908929753768, 0.3923210090691949, 0.5906338098651173, 1.0 ] ,
                [ 0.5703431258944366, 0.39086082263823363, 0.5882684149328958, 1.0 ] ,
                [ 0.5765865633099642, 0.38948320093203215, 0.585886901688302, 1.0 ] ,
                [ 0.5829195076877101, 0.38818961038863925, 0.5834836183511896, 1.0 ] ,
                [ 0.5893398550320008, 0.38698174105022043, 0.5810532559157813, 1.0 ] ,
                [ 0.5958451309392963, 0.3858614929678743, 0.5785908927129421, 1.0 ] ,
                [ 0.6024325269654214, 0.3848309610162593, 0.576092031869185, 1.0 ] ,
                [ 0.609098936568009, 0.38389241849066447, 0.5735526315586564, 1.0 ] ,
                [ 0.6158409899778253, 0.38304829983853356, 0.5709691281215967, 1.0 ] ,
                [ 0.6226550874563096, 0.3823011828468093, 0.5683384522778226, 1.0 ] ,
                [ 0.6295374305056685, 0.38165377056852123, 0.5656580387936072, 1.0 ] ,
                [ 0.6364840507068148, 0.38110887322948533, 0.5629258300635053, 1.0 ] ,
                [ 0.6434908359646159, 0.3806693903114255, 0.5601402741453472, 1.0 ] ,
                [ 0.6505535540359267, 0.38033829296351407, 0.5573003178381059, 1.0 ] ,
                [ 0.6576678733011254, 0.38011860685217386, 0.5544053954208565, 1.0 ] ,
                [ 0.6648293808128891, 0.3800133955205425, 0.5514554136794173, 1.0 ] ,
                [ 0.6720335977161417, 0.3800257442953476, 0.5484507338386633, 1.0 ] ,
                [ 0.6792759921804752, 0.3801587447508383, 0.5453921509961183, 1.0 ] ,
                [ 0.6865519900216446, 0.38041547971718576, 0.5422808716194908, 1.0 ] ,
                [ 0.6938569832129403, 0.38079900880452416, 0.5391184896301716, 1.0 ] ,
                [ 0.7011863365016757, 0.3813123544032522, 0.5359069615490322, 1.0 ] ,
                [ 0.708535392352146, 0.38195848811605465, 0.5326485811323627, 1.0 ] ,
                [ 0.7158994744356495, 0.38274031757663424, 0.5293459538763663, 1.0 ] ,
                [ 0.7232738898819071, 0.3836606736138007, 0.526001971719722, 1.0 ] ,
                [ 0.7306539304959, 0.3847222977264709, 0.5226197882266034, 1.0 ] ,
                [ 0.7380348731307385, 0.3859278298446106, 0.5192027944878743, 1.0 ] ,
                [ 0.745411979391886, 0.38727979636233795, 0.5157545959367134, 1.0 ] ,
                [ 0.7527804948316458, 0.38878059844153023, 0.5122789902368193, 1.0 ] ,
                [ 0.7601356477759803, 0.3904325005967065, 0.5087799463669161, 1.0 ] ,
                [ 0.7674726479090487, 0.3922376195839133, 0.5052615849944048, 1.0 ] ,
                [ 0.7747866847247106, 0.39419791362738094, 0.5017281602036554, 1.0 ] ,
                [ 0.7820729259390183, 0.3963151720272468, 0.4981840426203897, 1.0 ] ,
                [ 0.7893265159434558, 0.39859100519940655, 0.49463370395260847, 1.0 ] ,
                [ 0.7965425743657687, 0.40102683520413035, 0.4910817029503182, 1.0 ] ,
                [ 0.8037161947934128, 0.40362388682347805, 0.4875326727705735, 1.0 ] ,
                [ 0.8108424437042262, 0.40638317924850326, 0.48399130972080456, 1.0 ] ,
                [ 0.8179163596396358, 0.40930551843598917, 0.4804623633416823, 1.0 ] ,
                [ 0.8249329526477086, 0.412391490190918, 0.4769506277806716, 1.0 ] ,
                [ 0.8318872040163252, 0.41564145402544583, 0.47346093439858933, 1.0 ] ,
                [ 0.8387740663108825, 0.4190555378379101, 0.4699981455437034, 1.0 ] ,
                [ 0.8455884637258604, 0.4226336334468042, 0.46656714942096994, 1.0 ] ,
                [ 0.8523252927554259, 0.42637539300494015, 0.4631728559776506, 1.0 ] ,
                [ 0.858979423184789, 0.43028022630864715, 0.459820193720688, 1.0 ] ,
                [ 0.8655456994011996, 0.434347299006096, 0.4565141073756394, 1.0 ] ,
                [ 0.8720189420212763, 0.43857553169810515, 0.4532595562916191, 1.0 ] ,
                [ 0.8783939498295337, 0.4429635999144347, 0.4500615134914681, 1.0 ] ,
                [ 0.8846655020217126, 0.4475099349387505, 0.4469249652612601, 1.0 ] ,
                [ 0.8908283607454808, 0.4522127254466089, 0.4438549111682198, 1.0 ] ,
                [ 0.8968772739304364, 0.45706991991295404, 0.4408563643912384, 1.0 ] ,
                [ 0.9028069783989131, 0.4620792297390309, 0.4379343522434889, 1.0 ] ,
                [ 0.9086122032489296, 0.4672381330432965, 0.4350939167622526, 1.0 ] ,
                [ 0.9142876735005897, 0.47254387905696066, 0.43234011523715665, 1.0 ] ,
                [ 0.9198281139974602, 0.4779934930621275, 0.4296780205447205, 1.0 ] ,
                [ 0.9252282535547666, 0.4835837818091571, 0.4271127211546663, 1.0 ] ,
                [ 0.9304828293467029, 0.489311339349687, 0.4246493206720741, 1.0 ] ,
                [ 0.9355865915257897, 0.49517255322267045, 0.4222929367794384, 1.0 ] ,
                [ 0.9405343080679401, 0.5011636109325912, 0.4200486994442694, 1.0 ] ,
                [ 0.9453207698377563, 0.5072805066616954, 0.41792174826135803, 1.0 ] ,
                [ 0.9499407958696312, 0.5135190481612905, 0.4159172288044856, 1.0 ] ,
                [ 0.9543892388613723, 0.5198748637709469, 0.41404028787041863, 1.0 ] ,
                [ 0.9586609908784319, 0.5263434095184686, 0.41229606750873476, 1.0 ] ,
                [ 0.9627509892683234, 0.532919976257784, 0.41068969774452563, 1.0 ] ,
                [ 0.9666542227865541, 0.5395996968062036, 0.40922628791737875, 1.0 ] ,
                [ 0.9703657379373068, 0.5463775530467623, 0.4079109165792831, 1.0 ] ,
                [ 0.973880645534314, 0.5532483829654934, 0.4067486199160638, 1.0 ] ,
                [ 0.977194127489757, 0.5602068875973671, 0.405744378681395, 1.0 ] ,
                [ 0.9803014438417145, 0.5672476378582919, 0.40490310365897786, 1.0 ] ,
                [ 0.9831979400336716, 0.5743650812438782, 0.40422961969651244, 1.0 ] ,
                [ 0.985879054462781, 0.5815535483787208, 0.4037286483839791, 1.0 ] ,
                [ 0.9883403263171228, 0.5888072594026373, 0.40340478947760444, 1.0 ] ,
                [ 0.9905774037259248, 0.5961203301827467, 0.4032625011987279, 1.0 ] ,
                [ 0.9925860522506996, 0.6034867783424592, 0.4033060795626133, 1.0 ] ,
                [ 0.9943621637493517, 0.6109005291004573, 0.4035396369148108, 1.0 ] ,
                [ 0.9959017656495542, 0.618355420914672, 0.40396707987099933, 1.0 ] ,
                [ 0.9972010306718403, 0.6258452109281537, 0.40459208686909526, 1.0 ] ,
                [ 0.9982562870468691, 0.63336358021578, 0.4054180855489745, 1.0 ] ,
                [ 0.9990640292749314, 0.6409041388329714, 0.4064482301745586, 1.0 ] ,
                [ 0.9996209294787796, 0.6484604306702377, 0.4076853793048238, 1.0 ] ,
                [ 0.9999238494029076, 0.656025938120524, 0.4091320739042683, 1.0 ] ,
                [ 0.9999698531131594, 0.6635940865701828, 0.4107905160597245, 1.0 ] ,
                [ 0.999756220449508, 0.671158248729119, 0.41266254843970646, 1.0 ] ,
                [ 0.9992804612815002, 0.6787117488214193, 0.41474963459578795, 1.0 ] ,
                [ 0.9985403306095283, 0.6862478666647118, 0.4170528401642377, 1.0 ] ,
                [ 0.9975338445450975, 0.693759841674881, 0.419572814982209, 1.0 ] ,
                [ 0.9962592971887386, 0.7012408768425495, 0.4223097760884071, 1.0 ] ,
                [ 0.9947152784043095, 0.708684142739182, 0.4252634915359733, 1.0 ] ,
                [ 0.9929006924622185, 0.7160827816236869, 0.4284332649080771, 1.0 ] ,
                [ 0.9908147774905699, 0.7234299117350261, 0.431817920397442, 1.0 ] ,
                [ 0.9884571256315656, 0.730718631872364, 0.43541578829275446, 1.0 ] ,
                [ 0.9858277037498352, 0.7379420263814455, 0.43922469071053305, 1.0 ] ,
                [ 0.9829268744791043, 0.7450931706837594, 0.4432419274233957, 1.0 ] ,
                [ 0.9797554173237216, 0.7521651375028138, 0.44746426166707093, 1.0 ] ,
                [ 0.9763145494519169, 0.7591510039588698, 0.45188790586087874, 1.0 ] ,
                [ 0.9726059457295625, 0.7660438597183402, 0.4565085072508524, 1.0 ] ,
                [ 0.9686317574481381, 0.7728368163955676, 0.46132113358145127, 1.0 ] ,
                [ 0.964394629101435, 0.7795230184110294, 0.46632025901993496, 1.0 ] ,
                [ 0.9598977124661602, 0.7860956555094074, 0.4714997506946271, 1.0 ] ,
                [ 0.9551446771475113, 0.7925479771312397, 0.4768528563605053, 1.0 ] ]
    return c
