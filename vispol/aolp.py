import numpy as np

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
