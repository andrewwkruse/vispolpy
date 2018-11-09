import numpy as np
import vispol
import matplotlib.pyplot as plt
import colorspacious as clr

class color_wheel:
    def __init__(self,
                 size=512,
                 xspline=[0, 5, 20, 40, 73, 77, 100],
                 yspline=[0.0, 6.6, 13.7, 19.4, 26.4, 24.1, 0.0]):
        self.alpha_channel = np.zeros((size,size))
        M_array = np.zeros((size, size))
        self.rgb = np.zeros((size, size, 3))
        radius = size / 2

        # xdist = [idx - radius for idx, x in enumerate(M_array[:,0])]
        # ydist = [idx - radius for idx, x in enumerate(M_array[:,0])]
        xdist = np.tile(np.arange(0,size,1) - radius, (size,1)).transpose()
        ydist = np.tile(np.arange(0,size,1) - radius, (size,1))
        rdist = np.sqrt(xdist**2 + ydist**2) / radius

        # rdist = np.copy(M_array)
        # for idx, xy in np.ndenumerate(M_array):
        #     rdist[idx] = np.sqrt((idx[0] - radius)**2 + (idx[1] - 0)**2/radius )
        fade_start = 0.995
        fade_stop = 1.005
        where_good = np.where(rdist <= fade_start)
        self.alpha_channel[where_good] = 1
        M_array[where_good] = rdist[where_good] * max(yspline)


        where_fade = np.where((rdist > fade_start) & (rdist <= fade_stop))
        self.alpha_channel[where_fade] = (-1 / (fade_stop - fade_start)) * (rdist[where_fade] - fade_stop)
        M_array[where_fade] = max(yspline)

        h_array = np.arctan2(ydist, xdist) * 2

        bounds = vispol.Jbounds(M_array, xspline, yspline)
        # J_array = (bounds[:,:, 1] + bounds[:,:, 0])/ 2
        J_array = bounds[:,:,1]
        J_array[np.where(J_array == 0)] = 0.0001
        J_array, a_array, b_array = vispol.JMhtoJab(J_array, M_array, h_array)
        ucs = np.dstack((J_array, a_array, b_array))
        self.rgb = vispol.JabtoRGB(J_array, a_array, b_array)
        self.xyz = clr.cspace_convert(ucs.reshape((-1,3)), "CAM02-UCS", "XYZ100").reshape(self.rgb.shape)
        self.M = M_array
        self.h = h_array
    def create_fig(self,
                   fignum= -1,
                   RTick= np.linspace(0, 1, 5),
                   RLabel= np.linspace(0, 1, 5),
                   ThetaTick= np.linspace(0, 2 * np.pi, 12, endpoint=False),
                   ThetaLabel= np.concatenate((np.linspace(0, 180, 6, endpoint=False, dtype=int),
                                               np.linspace(0, 180, 6, endpoint=False, dtype=int)),
                                              axis=0)):

        if fignum < 0 or not fignum % 1 == 0:
            self.fig = plt.figure()
        else:
            self.fig = plt.figure(fignum)

        # for some reason you can't set axis coords the same for both. Setting ax and pax to within 0.00001
        self.ax = self.fig.add_axes([0.10001, 0.1, 0.8, 0.8])
        self.pax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
        self.im = self.ax.imshow(np.dstack((self.rgb, self.alpha_channel)), aspect='equal')
        self.ax.set_axis_off()
        self.pax.set_yticks(RTick)
        self.pax.set_yticklabels(RLabel)
        self.pax.set_xticks(ThetaTick)
        self.pax.set_xticklabels(ThetaLabel)
        self.pax.set_facecolor([0,0,0,0])
        self.pax.set_theta_zero_location('N')
        return