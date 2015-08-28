import numpy as np
from scipy import fftpack

def hz2mel(f):
    """mel scale"""
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel2hz(m):
    """inverse mel scala"""
    return 700.0 * (np.power(10.,  m / 2595.) - 1.)

def preemphasis(signal, alpha=0.97):
    """emphasis the high frequency part to make the spectrum more flatten"""
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def enframe(signal, frame_len, frame_step, window_func=lambda x: np.ones((1,x))):
    """frame the signal, may specify a analysis window function"""
    #FIXME add the frame length and frame step check
    # N: number samples; n: frame length; s: frame step; nf: number of frames
    N = len(signal)
    n = int(round(frame_len))
    s = int(round(frame_step))
    nf = int(np.ceil((N-n)/s)) + 1
    # pad the signal with zeros
    pad = np.zeros( s * (nf-1) + n - N)
    sigpad = np.append(signal, pad)
    win = np.tile(window_func(n),(nf,1))
    index1 = np.array(range(0, nf)) * np.tile(s, nf)
    index2 = index1 + np.tile(n-1, nf) + 1
    mp = map(lambda x,y: range(x, y), index1, index2)
    index = [x for x in mp]
    frames = sigpad[index] * win
    return frames

def stft(frames, nfft=512):
    """short time Fourier transform, rfft is for real input, 
    faster than fft with complex input"""
    return np.fft.rfft(frames, nfft)

def specpower(sigfft, nfft=512):
    """using the result of stft to calculate the power"""
    mag = np.absolute(sigfft) # Magnitude spectrum computation
    return 1. / nfft * np.square(mag) 

def logspecpower(specpow, threshold='y'):
    """log10 of spectrum power"""
    #FIXME
    pth = max(specpow) * 1e-20
    logspec = np.log10(np.array([max(x, pth) for x in specpow]))
    return logspec

def meltrifilter(n=26, nfft=512, framerate=16000, lower=300, upper=8000):
    """Compute Mel-spaced filterbank"""
    # upper must less or equal than Nyquist frequency
    mlow = hz2mel(lower)
    mup  = hz2mel(upper)
    
    mstep = (mup - mlow) / (n + 1)
    # num of points = num of triangulars/filters + 2
    N = n + 2
    
    mel = np.array(range(0, N)) * mstep + mlow
    f = mel2hz(mel)
    
    bin = np.floor((nfft + 1) * f / framerate)
    
    fbank = np.zeros([n,int(nfft/2+1)])
    # using bins of fft 
    for j in range(0, n):
        for i in range(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in range(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
         
    return (mel, f, fbank)

def bankenergy(specpow, fbank):
    return np.dot(specpow, fbank.T)

def DCT(logbankE):
    """using dct to get final ceptral coefficients"""
    return fftpack.dct(logbankE)

def mfcc_compute(signal, wlen=0.0250,step=0.010, alpha=0.97, samplerate=16000,
                 nfft=512,nfilter=26, lower=300, upper=8000):
    """compute mfcc"""
    # window length: wlen=0.0250 seconds, the wlength will be the
    # sample rate times wlen
    frame_len = int(samplerate * wlen)
    frame_step = int(samplerate * step)
    sig = preemphasis(signal, alpha)
    frames = enframe(sig, frame_len, frame_step, window_func=lambda x: np.ones((1,x)))
    sigfft = stft(frames, nfft)
    specp = specpower(sigfft, nfft)
    (M,F,fbank) = meltrifilter(nfilter, nfft, samplerate, lower, upper)
    bankE = bankenergy(specp, fbank)
    coeff = DCT(np.log10(bankE))
    logE  = logspecpower(np.sum(specp, 1))
    return (logE, coeff)
    
    
