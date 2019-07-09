import numpy as np
import matplotlib.pyplot as plt
import vispol

# f, ax = plt.subplots(3,1)
f, ax = plt.subplots(1)
size = 100
patch = 12
rang = 0.1
bins_delta = np.linspace(0, 1, 100)
num_samples = 5000
# delta = np.zeros(num_samples)
# A = np.zeros((num_samples, n**2))
std = 0.05
mn = np.zeros((size - 2 * patch)**2)
true_a = 0.4
for sigmas, color in zip(np.linspace(0, 2, 6), ['b', 'orange', 'g', 'k', 'r', 'm']):
    # for samples in range(num_samples):
    S1 = np.random.normal((sigmas * std) * np.cos(2 * true_a), std, (size, size))
    S2 = np.random.normal((sigmas * std) * np.sin(2 * true_a), std, (size, size))
    # A[samples] = np.arctan2(S2, S1) / 2
    A = np.arctan2(S2, S1) / 2
    # delta[samples] = np.sqrt(1 - np.mean(np.cos(2 * A[samples]))**2 + np.mean(np.sin(2 * A[samples]))**2)
    delta = vispol.delta_aop(A)
    mn_idx = 0
    for idx, d in np.ndenumerate(delta):
        if np.all(np.array(idx) > patch) and np.all(size - np.array(idx) > patch):
            A_sub = A[idx[0]-patch:idx[0]+patch+1, idx[1]-patch:idx[1]+patch+1]
            d_sub = delta[idx[0]-patch:idx[0]+patch+1, idx[1]-patch:idx[1]+patch+1]
            mn[mn_idx] = np.mean(A_sub[np.abs(d_sub-d) < rang])
            mn_idx += 1
    hist_mn, edges_mn = np.histogram(mn, bins='fd')
    ax.plot(edges_mn[:-1], hist_mn, c=color)
    # hist_delta, edges_delta = np.histogram(delta, bins=bins_delta)
    # pdf_delta = hist_delta / np.sum(hist_delta)
    # cdf_delta = np.cumsum(pdf_delta)

    # hist_A, edges_A = np.histogram(A.reshape((-1, 1)), bins=10)
    # pdf_A = hist_A / np.sum(hist_A)
    #
    # ax[0].plot(edges_delta[:-1], pdf_delta, c=color)
    # ax[1].plot(edges_delta[:-1], cdf_delta, c=color)
    # ax[2].plot(edges_A[:-1], pdf_A, c=color)
plt.show()