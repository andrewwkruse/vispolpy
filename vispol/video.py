import numpy as np
from .UCSvis import StokestoRGB

try:
    import imageio
except ImportError:
    print('Package \'imageio\' must be installed for using this method of producing video')

def StokestoFrames(Stokes_frames, progress=True, **kwargs):
    sz = Stokes_frames.shape
    rgb_sz = [sz[0], sz[1], sz[2], 3]
    returnUCS = kwargs['returnUCS'] if 'returnUCS' in kwargs.keys() else False
    if returnUCS:
        UCS_frames = np.zeros(rgb_sz)
    RGB_frames = np.zeros(rgb_sz)
    print('Beginning color conversions...')
    for number, frame in enumerate(Stokes_frames):
        if progress:
            print('{:.2%}% completed\n'.format(number/len(Stokes_frames)))
        if returnUCS:
            RGB_frames[number,:,:,:], UCS_frames[number,:,:,:] = StokestoRGB(frame, **kwargs)
        else:
            RGB_frames[number,:,:,:] = StokestoRGB(frame, **kwargs)
    if returnUCS:
        return RGB_frames, UCS_frames
    else:
        return RGB_frames

def IPAtoFrames(IPA_frames, progress=True, **kwargs):
    Stokes_frames = np.zeros_like(IPA_frames)
    Stokes_frames[:,:,:,0] = IPA_frames[:,:,:,0]
    Stokes_frames[:,:,:,1] = IPA_frames[:,:,:,1] * np.cos(2 * IPA_frames[:,:,:,2])
    Stokes_frames[:,:,:,2] = IPA_frames[:,:,:,1] * np.sin(2 * IPA_frames[:,:,:,2])
    return StokestoFrames(Stokes_frames, progress=progress, **kwargs)

def StokestoVideo(out_file,
                  Stokes_frames,
                  fps = 30,
                  progress=True,
                  **kwargs):
    returnUCS = kwargs['returnUCS'] if 'returnUCS' in kwargs.keys() else False
    if returnUCS:
        RGB_frames, UCS_frames = StokestoFrames(Stokes_frames, progress=progress, **kwargs)
    else:
        RGB_frames = StokestoFrames(Stokes_frames, progress=progress, **kwargs)

    try:
        RGB_frames = np.asarray(256 * RGB_frames, dtype='uint8')
    except MemoryError:
        for number, frame in enumerate(RGB_frames):
            RGB_frames[number, :,:,:] = np.array(256 * frame, dtype='uint8')
    print('Attempting to write video to ' + out_file)
    try:
        imageio.mimwrite(out_file, RGB_frames, fps=fps, macro_block_size=None)
        print('Video written to ' + out_file)
    except IOError:
        print('Error writing video to file. Check if the ffmpeg binary is installed correctly.\n'
              'This is necessary for the plugin within imageio. Information on installing found here:\n'
              'https://imageio.readthedocs.io/en/stable/format_ffmpeg.html')
    return RGB_frames

def vid_example(outfile):
    from .example import generate
    stokes = np.asarray([generate() for x in range(30*20)])
    StokestoVideo(outfile, stokes)
    return