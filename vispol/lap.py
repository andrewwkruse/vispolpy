import sys
sys.path.append('C:/python scripts/ciecam02 plot')
import Read_Meredith as rm
# from scipy.ndimage import binary_dilation
# from scipy.stats import circstd
# import scipy.fftpack as fftpack
# from scipy.linalg import solve_banded
import vispol
import numpy as np
from scipy.sparse.linalg import spsolve
# from scipy.sparse.linalg import spilu
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
# from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_erosion
from scipy.signal import medfilt2d
from scipy.signal import wiener

def ang_diff(D):
    P = np.pi
    return 1 - 2/P * np.abs((D + P/2) % (2 * P) - P)

vispol.register_cmaps()
data_folder = 'c:/users/z5052714/documents/unsw/unsw/data_sets/'
# filename = data_folder + 'forscottfrommeredith/162041.L1B2.v006.hdf5'
# filename = data_folder + 'forscottfrommeredith/162651.L1B2.v006.hdf5'
filename = data_folder + 'forscottfrommeredith/094821.L1B2.v006.hdf5'
# I, P, A = rm.getIPA(filename, start=[2, 32], end=[3900, 1403])
_, P, A = rm.getIPA(filename, start=[2, 32], end=[3000, 1403])
# I, P, A = rm.getIPA(filename, start=[1000, 800], end=[1600, 1300])
# I, _, A = rm.getIPA(filename, start=[2000, 150], end=[2200, 450]) # looks great
# _, _, A = rm.getIPA(filename, start=[1500, 150], end=[2200, 450]) # looks good with scaling
# _, _, A = rm.getIPA(filename, start=[2, 150], end=[2600, 1000])
# I = np.clip(I/np.percentile(I,99), 0, 1)
A *= np.pi/180.0
A[A > np.pi] -= np.pi

delta = vispol.delta_aop(A)
A45 = A + np.pi/8
A45[A45 > np.pi] -= np.pi
Aneg45 = A - np.pi/8
Aneg45[Aneg45 < 0 ] += np.pi
# plt.plot(np.linspace(-np.pi, np.pi, 256), ang_diff(np.linspace(-np.pi, np.pi, 256)))

# delta_patch = delta[2520:2620, 1200:1300].reshape((-1, 1))
# A_patch = A[2520:2620, 1200:1300].reshape((-1, 1))
# P_patch = P[2520:2620, 1200:1300].reshape((-1, 1))
# hist1, hist2, edges = np.histogram2d(A_patch, delta_patch, bins='fd')

# f, ax = plt.subplots(3)
# ax[0].scatter(A_patch, delta_patch)
# ax[1].scatter(A_patch, P_patch)
# ax[2].scatter(P_patch, delta_patch)
# print(np.mean(P_patch))
# print(np.std(P_patch))
# print(vispol.circular_mean(A_patch))
# print(np.sqrt(-2 * np.log(np.hypot(np.mean(np.sin(2 * A_patch)), np.mean(np.cos(2 * A_patch))))))
# print(np.mean(delta_patch))
# plt.show()

cap = 95
sigma = 2
# delta = grey_erosion(delta, size=(5, 5))
# delta = medfilt2d(delta, 7)
# delta = wiener(delta, 5)
# A, _ = vispol.histogram_eq(A,
#                                weighted=True,
#                                min_change=0.25,
#                                element=5,
#                                deltas = delta,
#                                suppress_noise=True,
#                                interval=[0.0,np.pi])#,
#                                # box=[[1100, A.shape[0]], [0, A.shape[1]]])
# plt.imsave("C:/users/z5052714/documents/weekly_meetings/28-06-2019/AoP_rot.png", A, cmap="AoP", vmin=0, vmax=np.pi)

f, ax = plt.subplots(1, 4)
# ax[0].imshow(delta, vmin=0, vmax=1)
# ax[1].imshow(A, vmin=0, vmax=np.pi, cmap="AoP")
# ax[2].imshow(P, vmin=0, vmax=1)
ax[0].imshow(np.cos(2 * A), cmap="gray")
ax[1].imshow(np.cos(2 * A45), cmap="gray")
ax[2].imshow(np.sin(2 * A), cmap="gray")
ax[3].imshow(np.cos(2 * Aneg45), cmap="gray")
plt.show()
# kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Ld = convolve2d(delta, kernel, mode='same', boundary='symm')
# Ld /= np.percentile(np.abs(Ld), cap)
# Ld = np.clip(Ld, -1, 1)

x_neg = np.array([[0, 0, 0],
                  [0, -1, 1],
                  [0, 0, 0]])
x_pos = np.array([[0, 0, 0],
                  [1, -1, 0],
                  [0, 0, 0]])
y_neg = np.array([[0, 0, 0],
                  [0, -1, 0],
                  [0, 1, 0]])
y_pos = np.array([[0, 1, 0],
                  [0, -1, 0],
                  [0, 0, 0]])

L = np.zeros_like(A)

# plt.imshow(medfilt2d(delta, 7))
# plt.show()

# f, ax = plt.subplots(1,4)
# cosA = np.cos(2 * A)
# sinA = np.sin(2 * A)
# ax[0].imshow(cosA, vmin=-1, vmax=1, cmap="BlueWhiteRed")
# ax[2].imshow(sinA, vmin=-1, vmax=1, cmap="BlueWhiteRed")
# filt_size = 5
# cosA = wiener(cosA, filt_size)
# sinA = wiener(sinA, filt_size)
# cosA = medfilt2d(cosA, filt_size)
# sinA = medfilt2d(sinA, filt_size)
# ax[1].imshow(cosA, vmin=-1, vmax=1, cmap="BlueWhiteRed")
# ax[3].imshow(sinA, vmin=-1, vmax=1, cmap="BlueWhiteRed")
# plt.show()
close_to_zero = np.abs(np.cos(2 * A) - 1) < 0.000005

for kernel in [x_neg, x_pos, y_neg, y_pos]:
# for kernel in [x_neg, y_neg]:
    # Lsub = np.sin(convolve2d(A, kernel, mode='same', boundary='symm'))
    Lsub0 = ang_diff(convolve2d(A, kernel, mode='same', boundary='symm'))
    Lsub45 = ang_diff(convolve2d(A45, kernel, mode='same', boundary='symm'))
    f, ax = plt.subplots(1, 5)
    ax[0].imshow(Lsub0, vmin=-1, vmax=1, cmap="BlueWhiteRed")
    ax[1].imshow(Lsub45, vmin=-1, vmax=1, cmap="BlueWhiteRed")
    ax[2].imshow(close_to_zero)
    ax[3].imshow(Lsub0 - Lsub45, cmap="BlueWhiteRed", vmin=-0.1, vmax=0.1)
    Lsub = Lsub0
    Lsub[close_to_zero] = Lsub45[close_to_zero]

    ax[4].imshow(Lsub - Lsub0, vmin=-.1, vmax=.1, cmap="BlueWhiteRed")
    # plt.show()
    # cos_arr = convolve2d(cosA, kernel, mode='same', boundary='symm')
    # sin_arr = convolve2d(sinA, kernel, mode='same', boundary='symm')
    # Lsub = cos_arr
    # Lsub[np.abs(cos_arr) < np.abs(sin_arr)] = sin_arr[np.abs(cos_arr) < np.abs(sin_arr)]
    L += Lsub

# L[500,500] = 0
# L = np.sin(np.pi/2 * L)

# from scipy.special import erf, erfinv
# endpoint = 0.99999999999
# factor = erfinv(endpoint)
# L = erf(factor * L) / endpoint
plt.figure()
plt.imshow(L, cmap="BlueWhiteRed", vmin=-1, vmax=1)

# plt.show()

n, m = A.shape

# kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
# L_F = fftpack.fft2(L)
# K_F = fftpack.fft2(kernel, shape=L.shape)
# f, ax = plt.subplots(1, 2)
# ax[0].imshow(np.real(L_F))
# ax[1].imshow(np.real(K_F))
# plt.show()
# U = fftpack.ifft2(L_F / K_F)
M = vispol.construct_matrix(A.shape, type='laplacian')
# M = vispol.construct_matrix(A.shape, type='lap_xy')

U = spsolve(M, L.reshape(n * m, 1)).reshape((n, m))
# U -= np.median(U)
#
# U /= 2 * np.max(np.abs(U))
# U += 0.5
#

U = (U - np.min(U)) / (np.max(U) - np.min(U))
U_enhanced = (U - np.percentile(U, 1)) / (np.percentile(U, 99) - np.percentile(U, 1))
U_enhanced = np.clip(wiener(U_enhanced, 3), 0, 1)
U = np.clip(U, 0, 1)
Ushrink = U_enhanced * (0.8 - 0.2) + 0.2

plt.figure()
A_slice = A[1000,:600]/np.pi
# A_slice[A_slice > 0.93] -= 1
# A_slice[564:] -= 1
U_slice = U[1000,:600]
slopes_A = ang_diff(np.convolve(A_slice, [-1, 1], mode='same'))
slopes_U = np.convolve(U_slice, [-1, 1], mode='same')
# A_slice = (A_slice - np.min(A_slice)) / (np.max(A_slice) - np.min(A_slice))
# U_slice = (U_slice - np.min(U_slice)) / (np.max(U_slice) - np.min(U_slice))
# A_delta = np.convolve(A_slice, [1, -1], mode='same')
# A_slice[A_delta > 0.5] -= 0.5
# A_slice[A_delta < -0.5] += 0.5
plt.plot(range(599), slopes_A[1:])
plt.plot(range(599), slopes_U[1:])
# plt.plot(range(600), slopes_A / slopes_U)
# plt.plot(range(600), np.abs(slopes_A / slopes_U))

P = 1 - np.clip(delta / np.max(delta), 0, 1)
# mask_params ={'thresh':0.4,
#               'smooth':True,
#               'morph':True}
# delta_params = {'mask_on':True, 'mask_params':mask_params}
# dmask = vispol.dmask(delta, thresh=0.3, smooth=True, morph=True)


# U *= dmask
# A += np.pi/2
RGB = vispol.IPAtoRGB(I = Ushrink, P=P, A=A, dependent="P")

f, ax = plt.subplots(1, 2)
# plt.imsave("C:/users/z5052714/documents/weekly_meetings/28-06-2019/U.png", U, cmap="gray")
# plt.imsave("C:/users/z5052714/documents/weekly_meetings/28-06-2019/U_enhanced.png", U_enhanced, cmap="gray")
# plt.imsave("C:/users/z5052714/documents/weekly_meetings/28-06-2019/RGB_rot.png", RGB)
ax[0].imshow(U_enhanced, cmap="gray")
ax[1].imshow(RGB)
# for idx, sig in enumerate([30, 20, 10, 5]):
#     ax[idx + 1].imshow(U - gaussian_filter(U, sigma=sig), cmap="gray")
    # ax[idx].imshow(Usig, cmap="gray")
plt.show()