import vispol
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
from scipy.sparse.linalg import inv as sparse_inverse
from scipy.sparse import save_npz as sparse_save

def ang_diff(D):
    P = np.pi
    return 1 - 2/P * np.abs((D + P/2) % (2 * P) - P)

vispol.register_cmaps()
side = 36
A = np.zeros((side, side))

# A[0:100, 100:400] = np.tile(np.linspace(0, np.pi, 300),
#                              (100, 1))
# A[200:300, 0:300] = np.tile(np.linspace(np.pi/2, 3*np.pi/2, 300),
#                              (100, 1))
# A[50:500, 200:300] = np.tile(np.linspace(np.pi/4, -np.pi/2, 50),
#                                      (100, 9)).transpose()
# A[:50, :50] = np.pi / 2 * np.ones((50, 50))
# A[50:100, :50] = np.pi / 3 * np.ones((50, 50))
# A[:50, 50:100] = np.pi / 3 * np.ones((50, 50))
# A[50:100, 50:100] = np.pi / 4 * np.ones((50, 50))
# A[100:150, 50:100] = np.pi / 7 * np.ones((50, 50))
# A[150:200, :50] = 4 * np.pi / 5 * np.ones((50, 50))
# A[150:200, 50:100] = 2 * np.pi / 3 * np.ones((50, 50))
#
# A[400:, :100] = np.max(np.dstack(np.meshgrid(np.linspace(0,np.pi/2, 100),
#                                                 np.linspace(-np.pi/4, 3 * np.pi/4, 100))),
#                           axis=-1)
#
# A[100:, 300:] = gaussian_filter(2 * np.pi * np.random.random((400, 200)), sigma=3)
# number = 6
# length = int(side / number / 2)
# for y in range(number):
#     ystart = y*length*2
#     A0 = (y+1) / number * np.pi
#     for x in range(number):
#         xstart = x*length*2
#         A1 = (x+1) / number * np.pi
#         A[xstart:xstart+2*length, ystart:ystart+length] = A0 * np.ones((2*length,length))
#         A[xstart:xstart+2*length, ystart+length:ystart+2*length] = A1 * np.ones((2*length,length))
    # A[:, ystart:ystart+length] = np.tile(np.linspace(A0, A0 + np.pi, side), (length, 1)).transpose()
A = A + np.random.normal(0, .05, size=(side, side))

for y in range(side):
    for x in range(side):
        if x + 15 * np.sin(x/side * 2 * np.pi)<y:
            A[y, x] = x * y / (side**2) * 2.2 * np.pi
        else:
            # A[y, x] = np.pi/2
            A[y, x] = 0


# A += 3*np.pi/4


# delta = vispol.delta_aop(A)

cosA = np.cos(2 * A)
cosA135 = np.cos(2 * A + 3*np.pi/2)
sinA = np.sin(2 * A)

# n=3
# av_cos = convolve2d(cosA, np.ones((n,n)), boundary='symm', mode='same')
# av_sin = convolve2d(sinA, np.ones((n,n)), boundary='symm', mode='same')
# av_A = np.arctan2(av_sin, av_cos) / 2

while np.any(A>np.pi) or np.any(A<0):
    A[A>np.pi] -= np.pi
    A[A<0] += np.pi

A2 = np.copy(A)
while np.any(A2 > 3 * np.pi/4) or np.any(A2 < -np.pi/4):
    A2[A2 > 3 * np.pi/4] -= np.pi
    A2[A2 < -np.pi/4] += np.pi
# while np.any(av_A>np.pi) or np.any(av_A<0):
#     av_A[av_A>np.pi] -= np.pi
#     av_A[av_A<0] += np.pi

plt.imshow(A, cmap="AoP", vmin=0, vmax=np.pi)
plt.figure()
plt.imshow(A2, cmap="AoP", vmin=-np.pi/2, vmax=np.pi/2)
# plt.figure()
# plt.imshow(av_A, cmap="AoP", vmin=0, vmax=np.pi)
# P = 1 - np.clip(delta / np.max(delta), 0, 1)

# A = av_A

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

num = 0
for kernel in [x_neg, x_pos, y_neg, y_pos]:
# for kernel in [x_neg, y_neg]:
    Lsub = ang_diff(convolve2d(A, kernel, mode='same', boundary='symm'))
    Lsub2 = ang_diff(convolve2d(A2, kernel, mode='same', boundary='symm'))
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(Lsub, cmap="BlueWhiteRed", vmin=-1, vmax=1)
    # manual edits:
    # if num == 0 or num ==1:
    #     Lsub[24:, :16] *= -1
    #     Lsub[24:, 29:] *= -1
        # Lsub[24:, :] *= -1

    ax[1].imshow(Lsub2, cmap="BlueWhiteRed", vmin=-1, vmax=1)
    # cos_arr = convolve2d(cosA, kernel, mode='same', boundary='symm')
    # sin_arr = convolve2d(sinA, kernel, mode='same', boundary='symm')
    # cos135_arr = convolve2d(sinA, kernel, mode='same', boundary='symm')
    # cos45_arr = convolve2d(sinA, kernel, mode='same', boundary='symm')
    # Lsub = cos_arr
    # Lsub[np.abs(cos_arr) < np.abs(sin_arr)] = sin_arr[np.abs(cos_arr) < np.abs(sin_arr)]
    # Lsub[np.abs(cos_arr) < np.abs(cos135_arr)] = cos135_arr[np.abs(cos_arr) < np.abs(cos135_arr)]
    # Lsub[np.abs(Lsub) < np.abs(cos45_arr)] = cos45_arr[np.abs(Lsub) < np.abs(cos45_arr)]
    num += 1
    L += Lsub

plt.figure()
plt.imshow(L, cmap="BlueWhiteRed", vmin=-1, vmax=1)

def residuals(f, L, size):
    f2d = f.reshape(size)
    lap_f = convolve2d(f2d, [[0, 1, 0],[1, -4, 1], [0, 1, 0]], mode='same', boundary='symm')
    return np.abs(L) - np.abs(lap_f).reshape(-1)

from functools import partial
residuals_partial = partial(residuals, L=L.reshape(-1), size=L.shape)

# print(residuals_partial(np.zeros(np.product(L.shape))).shape)
M, _, _, _ = vispol.construct_matrix(A.shape, type='laplacian')

# from scipy.optimize import least_squares
# U_lq = least_squares(residuals_partial, np.random.random(np.product(L.shape)), jac_sparsity=M)
# U = U_lq.x.reshape(A.shape)
# res = U_lq.fun.reshape(A.shape)
# # U = U / np.max(np.abs(U))
# f, ax = plt.subplots(1,2)
# ax[0].imshow(U, cmap="gray")
# ax[1].imshow(res, cmap="gray")
# f, ax = plt.subplots(1,2)
# ax[0].imshow(np.abs(convolve2d(U, [[0, 1, 0],[1, -4, 1], [0, 1, 0]], mode='same', boundary='symm')),
#              cmap="BlueWhiteRed", vmin=-1, vmax=1)
# ax[1].imshow(np.abs(L), cmap="BlueWhiteRed", vmin=-1, vmax=1)
# plt.show()
n, m = A.shape
# M = vispol.construct_matrix(A.shape, type='laplacian')
print(type(M))
def solve(M, L):
    print(type(M))
    return spsolve(A=M, b=L.reshape(n * m, 1)).reshape((n, m))
U = solve(M, L)
# U = spsolve(A=csc_matrix(M), b=L.reshape(n * m, 1)).reshape((n, m))
U = (U - np.min(U)) / (np.max(U) - np.min(U))
U_enhanced = (U - np.percentile(U, 1)) / (np.percentile(U, 99) - np.percentile(U, 1))
# U_enhanced = np.clip(wiener(U_enhanced, 3), 0, 1)
U_enhanced = np.clip(U_enhanced, 0, 1)
U = np.clip(U, 0, 1)
Ushrink = U_enhanced * (0.8 - 0.2) + 0.2
RGB = vispol.IPAtoRGB(I = Ushrink, P=np.ones_like(A), A=A, dependent="P")

f, ax = plt.subplots(1, 2)
ax[0].imshow(U, cmap="gray")
ax[1].imshow(RGB)

L = np.zeros_like(A)

for kernel in [x_neg, x_pos, y_neg, y_pos]:
# for kernel in [x_neg, y_neg]:
#     Lsub = ang_diff(convolve2d(A, kernel, mode='same', boundary='symm'))
    cos_arr = convolve2d(cosA, kernel, mode='same', boundary='symm')
    # sin_arr = convolve2d(sinA, kernel, mode='same', boundary='symm')
    # cos135_arr = convolve2d(sinA, kernel, mode='same', boundary='symm')
    cos45_arr = convolve2d(sinA, kernel, mode='same', boundary='symm')
    Lsub = cos_arr
    # Lsub[np.abs(cos_arr) < np.abs(sin_arr)] = sin_arr[np.abs(cos_arr) < np.abs(sin_arr)]
    Lsub[np.abs(cos_arr) < np.abs(cos45_arr)] = cos45_arr[np.abs(cos_arr) < np.abs(cos45_arr)]
    # Lsub[np.abs(Lsub) < np.abs(cos45_arr)] = cos45_arr[np.abs(Lsub) < np.abs(cos45_arr)]
    L += Lsub

n, m = A.shape
M = vispol.construct_matrix(A.shape, type='laplacian')
U = spsolve(M, L.reshape(n * m, 1)).reshape((n, m))
U = (U - np.min(U)) / (np.max(U) - np.min(U))
U_enhanced = (U - np.percentile(U, 1)) / (np.percentile(U, 99) - np.percentile(U, 1))
U_enhanced = np.clip(wiener(U_enhanced, 3), 0, 1)
U = np.clip(U, 0, 1)
Ushrink = U_enhanced * (0.8 - 0.2) + 0.2
RGB = vispol.IPAtoRGB(I = Ushrink, P=np.ones_like(A), A=A, dependent="P")

f, ax = plt.subplots(1, 2)
ax[0].imshow(U, cmap="gray")
ax[1].imshow(RGB)

f, ax = plt.subplots(1)
ax.imshow(cosA, cmap="gray")
plt.show()