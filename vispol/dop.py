import numpy as np

def StokestoDoLP(S):
# function to calculate AoLP of 2D array of Stokes vectors (as in an image)
# S = input stokes vector
# S can be nxmx2, nxmx3, nxmx4 array
# nxmx2:
#   S[0] = S1, S[1] = S2
# nxmx3:
#   S[0] = S0, S[1] = S1, S[2] = S2
# nxmx4:
#   S[0] = S0, S[1] = S1, S[2] = S2, S[3] = S3
#
# outputs:
#   d: nxm vector of angles
#

# Last edited 15:50 08/02/2018 by AWK



    if S.shape[2] == 2:
        S1 = S[:, :, 0]
        S2 = S[:, :, 1]
    else:
        S1 = S[:, :, 1]
        S2 = S[:, :, 2]


    d = np.sqrt(S1**2 + S2**2)
    d[np.isnan(S1)] = np.nan
    d[np.isnan(S2)] = np.nan

    return d