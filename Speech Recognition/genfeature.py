# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:01:28 2015

@author: wanghuaq
"""
from mfcc import mfcc_compute

import wave
import struct
import numpy as np
import os
import tarfile
#import re
import tables

def wave2signal(wave, nframes, samplewidth):
    # Only support 8bit, 16bit, 32bit decoding
    codecs = {1: 'b', 2: 'h', 4:'i'}
    code = codecs[samplewidth]
    return np.array(struct.unpack(nframes * code, wave))/pow(2, 8*samplewidth - 1)
                                  
def wave_parse(infile):
    """parse the wave file, get the params and frame data, 
    only support one channel current now"""
    wav = wave.Wave_read(infile)
    params = wav.getparams()
    frames = wav.readframes(params.nframes)
    wav.close()
    signal = wave2signal(frames, params.nframes, params.sampwidth)
    return (params.framerate, signal)

def untar(indir, outdir):
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            tar = tarfile.open(subdir + '/' + file)
            tar.extractall(path=outdir)
            tar.close()

#def downsample(indata, outdata):

def genfeature(indir):
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            if file.find(".wav") > 0:
                (framerate, signal) = wave_parse(subdir + '/' + file)
                (logE, coeff) = mfcc_compute(signal)
                feat = np.column_stack((logE, coeff[:, 1:13]))
                savefile = file.replace(".wav", ".mfcc")
                # save feature file
                mfcdir = os.path.abspath(os.path.join(subdir, os.pardir)) + '/mfc';
                if not(os.path.isdir(mfcdir)):
                    os.mkdir(mfcdir)
                f=tables.open_file(mfcdir  + '/' + savefile, mode='w')
                root = f.root
                f.create_array(root, "mfcc", feat)
                f.close()
                
def readfeature(file):
    f = tables.open_file(file)
    feat = f.root.mfcc.read()
    f.close()    
    return feat

def load_phonedict(filename):
    fd = open(filename)
    lines = fd.readlines()
    fd.close()
    #phonedict = dict(map(lambda x: x.upper().rstrip().split(maxsplit=1), lines))
    phonedict =dict([x.upper().rstrip().split(maxsplit=1) for x in lines])
    return(phonedict)
    
def gen_phone_seq(instring, phonedict):
    '''this is the function to generate phoneme sequence'''
    '''and the instring and phonedict are both upcase'''
    words = instring.upper().rstrip().split()
    #out = map(lambda x: phonedict[x], words)
    # FIXME if the word is not in the phone dictionary
    out = [phonedict[x] for x in words]
    return(out)
    

if __name__ == '__main__':
    indir = 'c:/app/sharefolder/vox'
    outdir = 'c:/aip'

    #untar(indir, outdir)

    # Read the phone dictionary
    phonedict_file = "C:/AA/speech recognition/cmu/xvoice-cmudict-unstressed"
    phonedict = load_phonedict(phonedict_file)
    # genfeature("C:/AIP/")
    
    



















