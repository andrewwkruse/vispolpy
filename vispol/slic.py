import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
# from .aolp import detect_range
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import vispol
vispol.register_cmaps()
class slic:
    def __init__(self, aop, grid_len=5, interval='auto', scale=True):
        self.grid_len = grid_len
        self.gradmag, self.cosA, self.sinA = gradmag_aop(aop, interval=interval, scale=scale)
        padded_gradmag = pad(self.gradmag)
        (self.xlen, self.ylen) = aop.shape
        x = np.tile(np.arange(self.xlen), (self.ylen, 1)).transpose()
        y = np.tile(np.arange(self.ylen), (self.xlen, 1))
        # this factor makes it so distances in cosine and sine are comparable to distances within a
        # square grid of length grid_len. This scales maximum difference in cosine and sine to the length
        # of the diagonal in the square grid
        self.factor = float(np.sqrt(2) * grid_len)
        self.array = np.dstack((self.factor * cosA, self.factor * sinA, x, y))
        self.labels = -1 * np.ones_like(aop, dtype=int)
        self.distances = np.inf * np.ones_like(aop, dtype=int)
        self.cluster_centers = self.array[::grid_len, ::grid_len].reshape((-1, 4))
        self.changes = np.inf * np.ones(self.cluster_centers.shape[0])
        for idx, m in enumerate(self.cluster_centers):
            xm, ym = m[2:].astype(int)

            # verbose indexing:
            # xpadded, ypadded = [xm + 1, ym + 1] # converting to the indeces of the padded array
            # xstart = xpadded - 1
            # xend = xpadded + 2
            # ystart = ypadded - 1
            # yend = ypadded + 2

            # efficient indexing:
            xstart, ystart, xend, yend = [xm, ym, xm + 3, ym + 3]

            region = padded_gradmag[xstart:xend, ystart:yend]
            region_min = np.array(np.unravel_index(np.argmin(region), region.shape), dtype=int)
            gradmin_padded = region_min - 1
            gradmin_x, gradmin_y = gradmin_padded - 1

            # set new clusters to minimum of 3x3 region and prevent out-of-bounds indeces
            new_xm = min(max(xm + gradmin_x, 0), int(self.xlen) - 1)
            new_ym = min(max(ym + gradmin_y, 0), int(self.ylen) - 1)

            new_m = self.array[new_xm, new_ym]

            self.cluster_centers[idx] = new_m
        self.superpixels = np.zeros_like(cosA)

    def update_labels(self, weight=1):
        def dist(pixels, cluster_center):
            da_sq = (pixels[:, :, 0] - cluster_center[0]) ** 2 + (pixels[:, :, 1] - cluster_center[1]) ** 2
            dxy_sq = (pixels[:, :, 2] - cluster_center[2]) **2 + (pixels[:, :, 2] - cluster_center[2]) ** 2
            return np.sqrt(weight ** 2 * da_sq + dxy_sq)

        for idx, m in  enumerate(self.cluster_centers):
            xstart, xend = np.array([max(m[2] - self.grid_len, 0), min(m[2] + self.grid_len + 1, self.xlen)], dtype=int)
            ystart, yend = np.array([max(m[3] - self.grid_len, 0), min(m[3] + self.grid_len + 1, self.ylen)], dtype=int)


            neighborhood = self.array[xstart:xend, ystart:yend]
            labels = self.labels[xstart:xend, ystart:yend]
            distances = self.distances[xstart:xend, ystart:yend]

            dist_idx = dist(neighborhood, m)
            # print(distances, dist_idx)
            new_pixels = np.where(distances - dist_idx > 0)
            labels[new_pixels] = idx
            distances[new_pixels] = dist_idx[new_pixels]
            self.labels[xstart:xend, ystart:yend] = labels
            self.distances[xstart:xend, ystart:yend] = distances
    def update_clusters(self):
        for idx, m in enumerate(self.cluster_centers):
            pixels = self.array[self.labels == idx]
            if int(pixels.shape[0]) > 0:
                new_m = np.sum(pixels, axis=0) / int(pixels.shape[0])
            else:
                new_m = m
            self.changes[idx] = np.sqrt(np.sum((m - new_m) ** 2))
            self.cluster_centers[idx] = new_m
            self.superpixels[self.labels == idx] = np.arccos(new_m[0] / self.factor) / 2.0

    def check_convergence(self, threshold = 1):
        threshold_scaled = threshold * float(self.cluster_centers.shape[0])
        total_change = np.sum(self.changes)
        print(total_change, threshold_scaled)
        return total_change < threshold_scaled

    def iterate(self, iterations=1000, weight=1, threshold=1):
        convergence = False
        ij = 0
        while(not convergence and ij < iterations):
            ij += 1
            self.update_labels(weight=weight)
            self.update_clusters()
            convergence = self.check_convergence(threshold=threshold)

