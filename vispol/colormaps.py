import colorspacious as clr
import numpy as np
from matplotlib.cm import register_cmap as rc

def JMhtoJab(J,M,h):
    a = M * np.cos(h)
    b = M * np.sin(h)
    return J, a, b

def diverging(start, stop, critical, steps = 256):
    to_crit = round(steps/2)
    to_stop = steps - to_crit
    t1 = np.linspace(0, 1, to_crit)
    t2 = np.linspace(0, 1, to_stop)

    abc1 = np.array(critical) - np.array(start)
    abc2 = np.array(stop) - np.array(critical)
    # colors1 = np.array(start) + np.dot(np.matrix(abc1).swapaxes(0,1), t1.reshape((1,to_crit)) )
    jab1 = np.array([abc1[idx] *t1 + start[idx] for idx in range(3)])
    jab2 = np.array([abc2[idx] *t2 + critical[idx] for idx in range(3)])
    jab = np.concatenate((jab1,jab2), axis=1)
    return np.swapaxes(jab, 0, 1)

def UCStoCmap(Jab, name = 'custom'):
    from matplotlib.colors import LinearSegmentedColormap as lsc

    RGB = clr.cspace_convert(Jab, "CAM02-UCS", "sRGB1")
    # for idx, color in enumerate(RGB):
    #     if len(color[np.where(color > 1)]) >= 1 or len(color[np.where(color < 0)]) >= 1:
    #         print(str(color) + ' outside of bounds. converted from ' + str(Jab[idx]))

    RGB = np.clip(RGB, 0, 1)
    cmap = lsc.from_list(name, RGB, N = len(RGB))
    return cmap

def register_cmaps(printout=True):
    c1 = DarkBlueLightRed()
    c2 = BlueWhiteRed()
    c3 = BlueBlackRed()
    c4 = BlueGrayRed()
    c5 = GrayBlackRed()
    c6 = GrayWhiteRed()
    c7 = BlueBlackGray()
    c8 = BlueWhiteGray()
    from .aolp import colormap
    c9 = colormap()
    c10 = colormap(isoluminant=True)
    if printout:
        print("Following maps have been registered:\n"
              "\"AoP\"\n"
              "\"isoluminantAoP\"\n"
              "\"DarkBlueLightRed\"\n"
              "\"BlueWhiteRed\"\n"
              "\"BlueBlackRed\"\n"
              "\"BlueGrayRed\"\n"
              "\"GrayBlackRed\"\n"
              "\"GrayWhiteRed\"\n"
              "\"BlueBlackGray\"\n"
              "\"BlueWhiteGray\"\n"

              )
    return


def DarkBlueLightRed():
    h1 = .1 * np.pi
    h2 = 1.5 * np.pi
    J1 = 60
    J2 = 20
    Jc = (J1 + J2) / 2
    M = 30
    start = JMhtoJab(J2, M, h2)
    stop = JMhtoJab(J1, M, h1)
    crit = [Jc, 0, 0]
    cmap = UCStoCmap(diverging(start, stop, crit), name="DarkBlueLightRed")
    rc("DarkBlueLightRed", cmap)
    return cmap

def BlueWhiteRed():
    h1 = .1 * np.pi
    h2 = 1.5 * np.pi
    J = 40
    M = 35

    M = 30
    start = JMhtoJab(J, M, h2)
    stop = JMhtoJab(J, M, h1)
    crit = [100, 0, 0]
    cmap = UCStoCmap(diverging(start, stop, crit), name='BlueWhiteRed')
    rc("BlueWhiteRed", cmap)
    return cmap

def BlueBlackRed():
    h1 = .1 * np.pi
    h2 = 1.5 * np.pi
    J = 40
    M = 35
    start = JMhtoJab(J, M, h2)
    stop = JMhtoJab(J, M, h1)
    crit = [0, 0, 0]
    cmap = UCStoCmap(diverging(start, stop, crit), name='BlueBlackRed')
    rc("BlueBlackRed", cmap)
    return cmap

def BlueGrayRed():
    h1 = .1 * np.pi
    h2 = 1.5 * np.pi
    J = 40
    M = 35
    start = JMhtoJab(J, M, h2)
    stop = JMhtoJab(J, M, h1)
    crit = [J, 0, 0]
    cmap = UCStoCmap(diverging(start, stop, crit), name='BlueGrayRed')
    rc("BlueGrayRed", cmap)
    return cmap

def GrayBlackRed():
    h = 1.5 * np.pi
    J = 40
    M = 35

    start = JMhtoJab(J,M,h)
    stop = JMhtoJab(J,0,0)
    crit = [0, 0, 0]
    cmap = UCStoCmap(diverging(start, stop, crit), name="GrayBlackRed")
    rc("GrayBlackRed", cmap)
    return cmap

def GrayWhiteRed():
    h = 1.5 * np.pi
    J = 40
    M = 35

    start = JMhtoJab(J,M,h)
    stop = JMhtoJab(J,0,0)
    crit = [100, 0, 0]
    cmap = UCStoCmap(diverging(start, stop, crit), name="GrayWhiteRed")
    rc("GrayWhiteRed", cmap)
    return cmap

def BlueBlackGray():
    h = .1 * np.pi
    J = 40
    M = 35

    start = JMhtoJab(J,M,h)
    stop = JMhtoJab(J,0,0)
    crit = [0, 0, 0]
    cmap = UCStoCmap(diverging(start, stop, crit), name="BlueBlackGray")
    rc("BlueBlackGray", cmap)
    return cmap

def BlueWhiteGray():
    h = .1 * np.pi
    J = 40
    M = 35

    start = JMhtoJab(J,M,h)
    stop = JMhtoJab(J,M,0)
    crit = [100, 0, 0]
    cmap = UCStoCmap(diverging(start, stop, crit), name="BlueWhiteGray")
    rc("BlueWhiteGray", cmap)
    return cmap

# class cbar(im, ax, aspect=20, pad_fraction=0.5, **cbar_kwargs):
#     def __init__(self):
#

def cbar(ax, aspect=20, pad_fraction=0.5, **cbar_kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
    from matplotlib.pyplot import colorbar
    from matplotlib.colorbar import ColorbarBase

    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1. / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    c = ColorbarBase(cax, **cbar_kwargs)
    # c = colorbar(im, cax=cax, norm=norm, **cbar_kwargs)
    return c
