import vispol
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/python scripts/ciecam02 plot')
import Read_Meredith as rm
import scipy.io as sio
import colorspacious as clr
from scipy.stats import entropy
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.ndimage.filters import gaussian_filter
import os
import imageio
from re import search

element = 15

def angular_difference(ang1, ang2):
    difference = np.abs(ang1 - ang2)
    difference[difference > np.pi/2] = np.pi - difference[difference > np.pi/2]
    return difference

def time_gradient(aop_time):
    # sigma = 21
    # length = 101
    # side = np.arange(-length//2, length//2 + 1)
    # xx, yy = np.meshgrid(side, side)
    # gauss = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    # gauss /= np.sum(gauss)
    # gauss = np.dstack((np.zeros_like(gauss), gauss, np.zeros_like(gauss)))
    # cosa = np.cos(2.0 * aop_time * np.pi / 180.0)
    # sina = np.sin(2.0 * aop_time * np.pi / 180.0)
    # cos_smooth = convolve(cosa, gauss, mode='same')
    # sin_smooth = convolve(sina, gauss, mode='same')
    # a_smooth = np.arctan2(sin_smooth, cos_smooth) / 2.0
    # conv = a_smooth
    kernel = np.array([[[1, 0, -1]]])

    conv = np.abs(convolve(aop_time, kernel, mode='same'))
    conv = np.clip(conv / (np.pi / 2), 0, 1)
    conv[conv > 0.5] = 1.0 - conv[conv > 0.5]
    conv *= 2
    return conv

data_folder = 'c:/users/z5052714/documents/unsw/unsw/data_sets/forscottfrommeredith'

time_slices = 0
fnames = []
for fname in os.listdir(data_folder):
    if 'hdf5' in fname:
        time_slices += 1
        fnames.append(fname)

fnames.sort()

hist = np.load(os.path.join(data_folder, 'Timelapse', 'histogram_time_weighted.npz'))
bins = hist['bins']
histogram = hist['histogram']
# hist2 = np.load(os.path.join(data_folder, 'Timelapse', 'histogram.npz'))
# bins2 = hist2['bins']
# histogram2 = hist2['counts']
# rgb_normal_frames = None
#
# normal_rgb_names = None
# for name in os.listdir(os.path.join(data_folder, 'Timelapse')):
#     # if 'suppressed.png' in name:
#     # if 'normal.png' in name:
#     if 'weighted.png' in name:
#         # normal_rgb_names.append(name)
#         number = int(search('\d+', name).group(0))
#         rgb = plt.imread(os.path.join(data_folder, 'Timelapse', name))
#         rgb = np.round(255.0 * rgb).astype('uint8')
#         rgb_shape = np.array(rgb.shape)
#         rgb_shape[0] -= rgb_shape[0] % 2
#         rgb_shape[1] -= rgb_shape[1] % 2
#         rgb = rgb[:rgb_shape[0], :rgb_shape[1], :]
#         if rgb_normal_frames is None:
#             rgb_normal_frames = np.zeros((80, rgb.shape[0], rgb.shape[1], rgb.shape[2]), dtype='uint8')
#         rgb_normal_frames[number] = rgb
#
# np.savez_compressed(os.path.join(data_folder, 'Timelapse', 'rgb_suppressed_time_weighted.npz'), rgb_normal_frames)
# np.savez_compressed(os.path.join(data_folder, 'Timelapse', 'rgb_suppressed_frames.npz'), rgb_normal_frames)
# rgb_normal_frames = np.load(os.path.join(data_folder, 'Timelapse', 'rgb_normal_frames.npz'))['arr_0']
# rgb_suppressed_frames = np.load(os.path.join(data_folder, 'Timelapse', 'rgb_suppressed_frames.npz'))['arr_0']
# rgb_normal_frames = np.concatenate((rgb_normal_frames, rgb_suppressed_frames), axis=2)
# bins = 'fd'
# histogram = None

# A_time = None
# # delta_time = np.load(os.path.join(data_folder, 'Timelapse', 'delta_time.npz'))['delta_time']
# # rgb_suppressed_frames = None
# #
sigma = 2
# for fnumber, fname in enumerate(fnames):
#     filename = os.path.join(data_folder, fname)
#     _, _, A = rm.getIPA(filename, start=[2, 32], end=[3931, 1403])
#     if A_time is None:
#         A_time = np.zeros((len(fnames), A.shape[0], A.shape[1]))
#     cosa = np.cos(2.0 * A * np.pi / 180.0)
#     sina = np.sin(2.0 * A * np.pi / 180.0)
#     cos_smooth = gaussian_filter(cosa, sigma=sigma)
#     sin_smooth = gaussian_filter(sina, sigma=sigma)
#     A_smooth = np.arctan2(sin_smooth, cos_smooth) / 2.0
#     A_time[fnumber] = A_smooth * 180.0 / np.pi
# np.savez_compressed(os.path.join(data_folder, 'Timelapse', 'A_time_smooth.npz'), A_time=A_time)
# np.savez_compressed(os.path.join(data_folder, 'Timelapse', 'convolution.npz'), conv)
A_time = np.load(os.path.join(data_folder, 'Timelapse', 'A_time_smooth.npz'))['A_time']
conv = time_gradient(A_time)

# conv = np.load(os.path.join(data_folder, 'Timelapse', 'convolution.npz'))['arr_0']
for idx in range(10):
    f, [ax0, ax1] = plt.subplots(1, 2)
    delta = vispol.delta_aop(A_time[idx] * np.pi / 180.0, element=element)
    ax0.imshow(conv[idx])
    ax1.imshow(1 - delta)

plt.show()

# #
# # avecosA = np.mean(np.cos(2.0 * np.pi * A_time / 180.0), axis=0)
# # avesinA = np.mean(np.sin(2.0 * np.pi * A_time / 180.0), axis=0)
# #
# # delta_time = np.sqrt(1.0 - (avecosA**2 + avesinA**2))
# # np.savez_compressed(os.path.join(data_folder, 'Timelapse', 'delta_time.npz'), delta_time = delta_time)
#
# # for A in A_time:
#     hist_region = np.zeros_like(A, dtype=bool)
#     hist_region[1100:] = True
#     deltas = vispol.delta_aop(A * np.pi / 180, element=element)
#     if bins is 'fd':
#         bins = np.histogram_bin_edges(A[hist_region], bins=bins)
#
    # counts, bins = np.histogram(A[hist_region], weights=deltas[hist_region] * delta_time[hist_region], bins=bins)
#     # ax.plot(bins[:-1], counts/np.sum(counts), label='frame {}'.format(fnumber))
#     if histogram is None:
#         histogram = counts
#     else:
#         histogram = histogram + counts
#
# np.savez(os.path.join(data_folder, 'Timelapse', 'histogram_time_weighted.npy'), histogram=histogram, bins=bins)


# filename = data_folder + 'forscottfrommeredith/162041.L1B2.v006.hdf5'
# filename = data_folder + 'Julia Craven/ProcessedData.mat'
# Stokes_sets = sio.loadmat(filename)['StokesData']
# WL0 = Stokes_sets[:, :, :, 1]
# A = vispol.StokestoAoLP(WL0)
# rgb_normal_frames = None
# for fnumber, fname in enumerate(fnames):
#     filename = os.path.join(data_folder, fname)
#     _, _, A = rm.getIPA(filename, start=[2, 32], end=[3931, 1403])
#     # f, ax0 = plt.subplots()
#     if rgb_normal_frames is None:
#         rgb_normal_frames = np.zeros((len(fnames), A.shape[0], A.shape[1], 3), dtype='uint8')
#     if rgb_suppressed_frames is None:
#         rgb_suppressed_frames = np.zeros((len(fnames), A.shape[0], A.shape[1], 3), dtype='uint8')


    # deltas = vispol.delta_aop(A * np.pi / 180, element=15)
# deltas = vispol.delta(WL0, element=3)
#     Aeq, lut = vispol.histogram_eq(A,
#                                    weighted=True,
#                                    min_change=0.25,
#                                    element=element,
#                                    suppress_noise=True,
#                                    histogram=[histogram, bins],
#                                    deltas=deltas,
#                                    interval=[0.0,180.0],
#                                    box=[[1100, A.shape[0]], [0, A.shape[1]]])


# Aeq_noisy, lut_noisy = vispol.histogram_eq(A,
#                                weighted=True,
#                                min_change=0.25,
#                                element=element,
#                                histogram=[histogram, bins],
#                                suppress_noise=False,
#                                interval=[0.0,180.0],
#                                box=[[1100, A.shape[0]], [0, A.shape[1]]])
# _, lut = vispol.histogram_eq(A[1100:], weighted=True, min_change=0.0, element=15)
# _, lut = vispol.histogram_eq_Stokes(WL0, weighted=True, min_change=0.25, element=3)
# print(lut)
# A_eq_pi = np.pi * Aeq / 180.0
# A_pi_noise = np.pi * Aeq_noisy / 180.0
# delta_eq = vispol.delta_aop(A_eq_pi, element=15)
# delta_noise = vispol.delta_aop(A_pi_noise, element=15)
#     vispol.register_cmaps()

# neighborhood = np.ones((15,15))
# cos_average = convolve2d(np.cos(2 * A_eq_pi), neighborhood, 'same')
# sin_average = convolve2d(np.sin(2 * A_eq_pi), neighborhood, 'same')
# aop_average = np.arctan2(sin_average, cos_average) / 2
# aop_average[aop_average < 0] += np.pi
# aop_average_180 = aop_average / np.pi * 180.0
# difference = angular_difference(aop_average, A_eq_pi) * 180 / np.pi * (1 - deltas)

    # rgb_normal = vispol.colormap_delta(A, deltas=deltas, interval=[0,180])
    # rgb_normal *= 255
    # rgb_normal = np.round(rgb_normal).astype('uint8')
# # rgb1 = vispol.colormap_delta(Aeq_noisy, deltas=deltas, interval=[0,180])
#
#     rgb_suppressed = vispol.colormap_delta(Aeq, deltas=deltas, interval=[0,180])
#     rgb_suppressed *= 255
#     rgb_suppressed = np.round(rgb_suppressed).astype('uint8')
#     # if rgb_normal_frames is None:
#     if rgb_suppressed_frames is None:
#         # rgb_normal_frames = np.zeros((80, rgb_suppressed.shape[0], rgb_suppressed.shape[1], rgb_suppressed.shape[2]), dtype='uint8')
#         rgb_suppressed_frames = np.zeros((80, rgb_suppressed.shape[0], rgb_suppressed.shape[1], rgb_suppressed.shape[2]), dtype='uint8')
# # #     plt.imsave(os.path.join(data_folder, 'Timelapse', 'frame{:d}_normal'.format(fnumber)), rgb_normal.astype('uint8'))
#     plt.imsave(os.path.join(data_folder, 'Timelapse', 'frame{:d}_suppressed_time_weighted'.format(fnumber)), rgb_suppressed.astype('uint8'))
# #     # rgb_normal_frames[fnumber, :, :, :] = rgb_normal
#     rgb_suppressed_frames[fnumber, :, :, :] = rgb_suppressed
#
# imageio.mimwrite(os.path.join(data_folder, 'Timelapse', 'normal_full.mp4'), rgb_normal_frames, fps=10, macro_block_size=None)
# imageio.mimwrite(os.path.join(data_folder, 'Timelapse', 'both_full.mp4'), rgb_normal_frames, fps=10, macro_block_size=None)
# imageio.mimwrite(os.path.join(data_folder, 'Timelapse', 'equalised_both.mp4'), rgb_normal_frames, fps=10, macro_block_size=None)
# # rgb3 = vispol.colormap_delta(aop_average_180, deltas=deltas, interval=[0,180])
# np.save(os.path.join(data_folder, 'Timelapse', 'lut.npy'), lut)
# ax0.imshow(deltas)
# ax1.imshow(delta_noise)
# plt.show()
# Jab0 = clr.cspace_convert(rgb0[1100:].reshape((-1, 3)), "sRGB1", "CAM02-UCS")
# Jab1 = clr.cspace_convert(rgb1[1100:].reshape((-1, 3)), "sRGB1", "CAM02-UCS")
# Jab2 = clr.cspace_convert(rgb2[1100:].reshape((-1, 3)), "sRGB1", "CAM02-UCS")

# rgb0 = vispol.colormap_delta(A, deltas=deltas)
# rgb1 = vispol.colormap_delta(Aeq, deltas=deltas)
#     im0 = ax0.imshow(rgb2, cmap='AoP')
# im0 = ax0.imshow(rgb0, cmap='AoP')
# im0 = ax0.imshow(delta)
# im1 = ax1.imshow(rgb3, cmap='AoP')
# im2 = ax2.imshow(rgb2, cmap="AoP")
# im2 = ax2.imshow(rgb2, cmap="AoP")
# im0 = ax0.imshow(A, cmap='AoP')
# im1 = ax1.imshow(Aeq, cmap='AoP')
# im2 = ax2.imshow(difference)

    # old_ticks_minor = np.linspace(0, 180, 37)
    # # old_ticks_minor = np.linspace(-np.pi/2, np.pi/2, 37)
    # new_ticks_minor = vispol.LUT_matching(old_ticks_minor, lut=lut)
    # old_ticks_major = ['{:d}'.format(int(t)) if idx%3 == 0 else '' for idx, t in enumerate(old_ticks_minor)]
    # # old_ticks_major = ['{:.3f}'.format(t) if idx%3 == 0 else '' for idx, t in enumerate(old_ticks_minor)]
    # # new_ticks_major = ['{:d}'.format(int(t)) if idx%5 == 0 else '' for idx, t in enumerate(new_ticks_minor)]
    #
    # # for a in [ax0, ax1, ax2]:
    # #     a.set_axis_off()
    # ax0.set_axis_off()
    # from matplotlib.colors import Normalize
    # norm = Normalize(vmin=0, vmax=180)
    # # norm = Normalize(vmin=-np.pi/2, vmax=np.pi/2)
    # cb0 = vispol.cbar(ax0, norm=norm, cmap="AoP", ticks=old_ticks_minor)
    # # cb1 = vispol.cbar(ax1, norm=norm, cmap="AoP", ticks=new_ticks_minor)
    # # cb2 = vispol.cbar(ax2, norm=norm, cmap="AoP", ticks=new_ticks_minor)
    # # cb2 = plt.colorbar(im2, ax=ax2)
    # cb0.set_ticklabels(old_ticks_major)
    # cb1.set_ticklabels(old_ticks_major)
    # cb2.set_ticklabels(old_ticks_major)
    # cb0.set_clim(0, 180)
    # cb1.set_clim(0, 180)
    # ax3.plot(lut[0], lut[1])
    # ax3.set_xlabel('Old level')
    # ax3.set_ylabel('New level')

    # plt.show()
