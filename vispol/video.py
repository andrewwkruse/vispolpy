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

def FramestoVideo(out_file,
                  input_frames,
                  input_type='Stokes',
                  fps = 30,
                  progress=True,
                  **kwargs):
    # input can be Stokes, IPA, or processed RGB
    if input_type in ['RGB', 'rgb']:
        RGB_frames = input_frames
    else:
        toFrames = StokestoFrames if input_type in ['Stokes', 'stokes'] else IPAtoFrames
        returnUCS = kwargs['returnUCS'] if 'returnUCS' in kwargs.keys() else False
        if returnUCS:
            RGB_frames, UCS_frames = toFrames(input_frames, progress=progress, **kwargs)
        else:
            RGB_frames = toFrames(input_frames, progress=progress, **kwargs)
    width, height = RGB_frames.shape[1:-1]
    if width%2:
        print('width {} not divisible by 2. Clipping edge to {} pixels.'.format(width, width-1))
        RGB_frames = RGB_frames[:, :-1, :]
    if height%2:
        print('height {} not divisible by 2. Clipping edge to {} pixels.'.format(height, height-1))
        RGB_frames = RGB_frames[:, :, :-1]

    try:
        RGB_frames = np.asarray(np.round(255 * RGB_frames), dtype='uint8')
    except MemoryError:
        for number, frame in enumerate(RGB_frames):
            RGB_frames[number, :,:,:] = np.round(255 * frame)
        RGB_frames = RGB_frames.astype('uint8', copy=False)
    print('Attempting to write video to ' + out_file)
    try:
        imageio.mimwrite(out_file, RGB_frames, fps=fps, quality=10, macro_block_size=None)
        print('Video written to ' + out_file)
    except IOError:
        print('Error writing video to file. Check if the ffmpeg binary is installed correctly.\n'
              'This is necessary for the plugin within imageio. Information on installing found here:\n'
              'https://imageio.readthedocs.io/en/stable/format_ffmpeg.html')
    return RGB_frames

def vid_example(outfile):
    from .example import generate
    stokes = np.asarray([generate() for x in range(30*20)])
    FramestoVideo(outfile, stokes)
    return