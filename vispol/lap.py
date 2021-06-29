import sys
sys.path.append('C:/python scripts/ciecam02 plot')
import Read_Meredith as rm
from scipy.ndimage import binary_dilation, binary_closing, binary_opening, generate_binary_structure
import vispol
import numpy as np
from scipy.sparse.linalg import spsolve
# from scipy.sparse.linalg import spilu
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_erosion, grey_opening, grey_closing
from scipy.signal import medfilt2d
from scipy.signal import wiener
from scipy.sparse.linalg import inv as sparse_inverse
from scipy.sparse import save_npz as sparse_save
from scipy import sparse
from scipy import optimize
from scipy.sparse import diags
from scipy.sparse.linalg import splu



    # P = np.pi
    # return 1 - 2/P * np.abs((D + P/2) % (2 * P) - P)

def obj_fun(Uout, Lin):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    Lout = convolve2d(Uout.reshape(Lin.shape), kernel, mode='same', boundary='symm')
    print(np.sqrt(np.mean((Lin - Lout) ** 2)))
    return np.sqrt(np.mean((Lin - Lout)**2))

class gauss_seidel:
    def __init__(self, D, L, U, b, omega):
        # self.D = diags(A.diagonal(k=0), offsets=0)
        # print(sparse.isspmatrix_csc(A))
        # LU = splu(A)
        # self.L = LU.L - self.D
        # self.U = LU.U

        self.D = D
        self.U = U
        self.L = L

        print('beginning terms')
        DplusOmegaL = self.D + omega * self.L
        # DplusOmegaLinv = sparse_inverse(DplusOmegaL)
        # print(DplusOmegaLinv.nnz, DplusOmegaLinv.shape)
        #
        # plt.imshow(DplusOmegaLinv.todense())
        term1_LU = splu(DplusOmegaL)
        DplusOmegaLinv = sparse.csc_matrix(D.shape, dtype=D.dtype)

        for idx in range(L.shape[0]):
            array = np.zeros(L.shape[0], dtype=D.dtype)
            array[idx] = 1.0
            DplusOmegaLinv[idx] = sparse.csc_matrix(term1_LU.solve(array), dtype=D.dtype)
            # DplusOmegaLinv = sparse.csc_matrix(DplusOmegaLinv)
            # print(DplusOmegaLinv[idx] == 0)
        DplusOmegaLinv = DplusOmegaLinv.transpose()
        print(DplusOmegaLinv.nnz, DplusOmegaLinv.shape)
        # plt.figure()
        # plt.imshow(DplusOmegaLinv.todense())
        # plt.show()

        print('term 1 complete')
        omegab = omega * b
        print('term 2 complete')
        omegaUplusomegaminusoneD = omega * self.U + (omega - 1) * self.D
        print('term 3 complete')
        self.terms = [DplusOmegaLinv, omegab, omegaUplusomegaminusoneD]

    def iterate(self, x0, max_iter=None, convergence=None):
        iterations = 0
        rmse = 10000000.0
        xk = x0
        def run(xk, *terms):
            xkmin1 = xk
            xk = terms[0].dot(terms[1] - terms[2].dot(xkmin1))
            residuals = xk - xkmin1
            rmse = np.sqrt(np.mean(residuals ** 2))
            return xk, rmse



        if max_iter is None and convergence is None:
            while iterations < 100:
                xk, rmse = run(xk, *self.terms)
                print('iteration: {}\n'
                      'root mean square: {}'.format(iterations, rmse))
                iterations += 1
        elif max_iter is None and not convergence is None:
            while rmse > convergence:
                xk, rmse = run(xk, *self.terms)
                print('iteration: {}\n'
                      'root mean square: {}'.format(iterations, rmse))
                iterations += 1
        elif not max_iter is None and convergence is None:
            while iterations < max_iter:
                xk, rmse = run(xk, *self.terms)
                print('iteration: {}\n'
                      'root mean square: {}'.format(iterations, rmse))
                iterations += 1
        else:
            while iterations < max_iter and rmse > convergence:
                xk, rmse = run(xk, *self.terms)
                print('iteration: {}\n'
                      'root mean square: {}'.format(iterations, rmse))
                iterations += 1
        self.xk = xk
        self.rmse = rmse
        self.iterations = iterations

vispol.register_cmaps()
# data_folder = 'c:/users/z5052714/documents/unsw/unsw/data_sets/'
# filename = data_folder + 'forscottfrommeredith/162041.L1B2.v006.hdf5'
# filename = data_folder + 'forscottfrommeredith/162651.L1B2.v006.hdf5'
# filename = data_folder + 'forscottfrommeredith/094821.L1B2.v006.hdf5'

# I = np.clip(I/np.percentile(I,99), 0, 1)
A *= np.pi/180.0
A[A > np.pi] -= np.pi

# I = np.clip((I - np.percentile(I, 1))/(np.percentile(I, 99) - np.percentile(I, 1)), 0, 1)

delta = vispol.delta_aop(A)
delta = wiener(delta, 5)

dmask3 = vispol.dmask(delta, morph=True, struct=3, thresh=0.4)
dmask5 = vispol.dmask(delta, morph=True, struct=5, thresh=0.4, smooth=True)
dmask7 = vispol.dmask(delta, morph=True, struct=7, thresh=0.4)
# dmask5 = vispol.dmask(delta, struct=11, thresh=0.4)
# dmask7 = vispol.dmask(delta, struct=13, thresh=0.4)
# delta_er = grey_erosion(delta, (5,5))
# delta_open = grey_opening(delta, (5,5))
# delta_close = grey_closing(delta, (5,5))

# f, ax = plt.subplots(1,5)
# ax[0].imshow(delta, vmin=0, vmax=1)
# ax[1].imshow(dmask3, vmin=0, vmax=1)
# ax[2].imshow(dmask5, vmin=0, vmax=1)
# ax[3].imshow(dmask7, vmin=0, vmax=1)
# ax[4].imshow(A, cmap="AoP")

# cosA = medfilt2d(np.cos(2 * A), 5)
# sinA = medfilt2d(np.sin(2 * A), 5)
# A = np.arctan2(sinA, cosA) / 2

# ax[1].imshow(delta_er, vmin=0, vmax=1)
# ax[2].imshow(delta_open, vmin=0, vmax=1)
# ax[3].imshow(delta_close, vmin=0, vmax=1)
# plt.show()

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
# A, _ = vispol.histogram_eq(A,
#                                weighted=True,
#                                min_change=0.25,
#                                element=5,
#                                deltas = delta,
#                                suppress_noise=True,
#                                interval=[0.0,np.pi])#,
#                                # box=[[1100, A.shape[0]], [0, A.shape[1]]])
# plt.imsave("C:/users/z5052714/documents/weekly_meetings/28-06-2019/AoP_rot.png", A, cmap="AoP", vmin=0, vmax=np.pi)

# f, ax = plt.subplots(1, 4)
# ax[0].imshow(delta, vmin=0, vmax=1)
# ax[1].imshow(A, vmin=0, vmax=np.pi, cmap="AoP")
# ax[2].imshow(P, vmin=0, vmax=1)

# kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Ld = convolve2d(delta, kernel, mode='same', boundary='symm')
# Ld /= np.percentile(np.abs(Ld), cap)
# Ld = np.clip(Ld, -1, 1)




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
# close_to_zero = np.abs(np.cos(2 * A) - 1) < 0.000005

# A = dmask5.astype(float) * A
for kernel in [x_neg, x_pos, y_neg, y_pos]:
# for kernel in [x_neg, y_neg]:
#     f, ax = plt.subplots(1, 3)
    Lsub = ang_diff(convolve2d(A, kernel, mode='same', boundary='symm'))
    # LI = convolve2d(I, kernel, mode='same', boundary='symm')
    # Lsub = np.sign(LI) * np.abs(ang_diff(convolve2d(A, kernel, mode='same', boundary='symm')))
    # Lsub = ang_diff(convolve2d(A, kernel, mode='same', boundary='symm'))
    # Lsub0 = ang_diff(convolve2d(A, kernel, mode='same', boundary='symm'))
    # Lsub45 = ang_diff(convolve2d(A45, kernel, mode='same', boundary='symm'))
    # ax[0].imshow(Lsub, vmin=-.2, vmax=.2, cmap="BlueWhiteRed")
    # ax[1].imshow(LI, vmin=-.2, vmax=.2, cmap="BlueWhiteRed")
    # ax[2].imshow(np.sign(LI), cmap="BlueWhiteRed")
    # ax[3].imshow(Lsub0 - Lsub45, cmap="BlueWhiteRed", vmin=-0.1, vmax=0.1)
    # Lsub = Lsub0
    # Lsub[close_to_zero] = Lsub45[close_to_zero]
    #
    # ax[4].imshow(Lsub - Lsub0, vmin=-.1, vmax=.1, cmap="BlueWhiteRed")
    # plt.show()
    # cos_arr = convolve2d(cosA, kernel, mode='same', boundary='symm')
    # sin_arr = convolve2d(sinA, kernel, mode='same', boundary='symm')
    # Lsub = cos_arr
    # Lsub[np.abs(cos_arr) < np.abs(sin_arr)] = sin_arr[np.abs(cos_arr) < np.abs(sin_arr)]
    L += Lsub

# plt.show()


# opt_results = optimize.minimize(obj_fun,
#                                 x0 = np.ones_like(L),
#                                 args=L,
#                                 method='TNC')
# print(opt_results)
# U_min = opt_results.x
# plt.imshow(U_min.reshape(n, m))
# plt.show()
M, Dmat, Lmat, Umat = vispol.construct_matrix(A.shape, type='laplacian')

# b=L.reshape((n * m, 1))
# gs = gauss_seidel(D=Dmat, L=Lmat, U=Umat, b=b, omega = 1.1)
# gs.iterate(x0=np.zeros_like(b), max_iter=100, convergence=0.0001)
# U_gs = gs.xk.reshape((n, m))
# plt.imshow(U_gs)
# plt.figure()
# plt.imshow(A, cmap="AoP")
# plt.show()
# plt.imshow(M.todense())
# M_inv = sparse_inverse(M)
# print(type(M_inv))
# plt.figure()
# plt.imshow(M_inv.todense())
# plt.show()
# sparse_save('M_inverse.npz', matrix=M_inv)
# U_2 = M_inv.multiply(L.reshape(1, n * m)).todense().reshape((n, m))
# U_2 = sparse.csc_matrix.dot(M_inv, L.reshape(n * m, 1)).reshape((n, m))
# M = vispol.construct_matrix(A.shape, type='lap_xy')
# Id = M_inv.dot(M).todense()
# plt.figure()
# plt.imshow(Id)
# plt.show()
# U -= np.median(U)
#
# U /= 2 * np.max(np.abs(U))
# U += 0.5
#

# U = (U - np.min(U)) / (np.max(U) - np.min(U))
plt.figure()
plt.imshow(U, cmap="gray")
plt.figure()
plt.imshow(A, cmap="AoP")
# plt.figure()
# plt.imshow(U_2, cmap="gray")
plt.show()
U_enhanced = (U - np.percentile(U, 1)) / (np.percentile(U, 99) - np.percentile(U, 1))
U_enhanced = np.clip(wiener(U_enhanced, 3), 0, 1)
plt.imsave('A_cmapped.png', A, cmap="AoP")
plt.imsave('A_grayscale.png', U, cmap="gray")
plt.imsave('A_grayscale_enhanced.png', U_enhanced, cmap="gray")
# U = np.clip(U, 0, 1)
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