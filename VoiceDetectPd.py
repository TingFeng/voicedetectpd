# -*- coding: utf-8 -*-
"""
Created for Insight project
@author: ting
"""
# This is the main program to extract feature of the voice
# and to build logistic regression model

# # for download the data, use the R script DownloadData.R

# # the voice data file is apple loseless compression (ALAC) encodeed
# # the original .m4a file has 32 bit precision, so common tools 
# # like python audiotool package can not proceed it
# # so I convert the .m4a file into .wav format using ffmpeg
# # with zero compression though

# import packages
import os # for find files in a path
import numpy as np # for numpy array and related
from numpy.fft import fft, fftfreq # for Fourier transform
import soundfile as sf # for extracting converted wave file
import csv # for read in csv files

# navigate to the folder of .wav files, each file is named with the audio ID
os.chdir('/mnt/hgfs/InsightVoiceData/output')
# List all the files so we can read them
AudioFile=os.listdir('/mnt/hgfs/InsightVoiceData/output')

# initiate the numpy array for eight features of each voice sample
Prop=np.zeros(shape=(len(AudioFile),8))

for Audio in AudioFile:
    data,Fs=sf.read(Audio) # read audio data
    Trans=np.zeros(shape=(200,251)) # we only look at 0-5000Hz
    for Index in np.arange(1,201,step=1):
        # here we are looking at fourier transform at every 50ms 
        # so we can check the variance/consistency across time
        t=np.arange(Fs/20*(Index-1),min(len(data),Fs/20*Index),step=1)
        sp=fft(data[t].T)
        sp=sp.T
        freq=fftfreq(t.size,d=1.0/Fs)
        Trans[Index-1,]=2*abs(sp[0:251].T)
    Power=Trans.sum(axis=1) # note this is not the actual power
                       # rather it is the summed amplitude
                       # power is the square of amplitude
    Power[np.where(np.isnan(Power))]=0 
                      # so if there is no recording in the beginning or end
                      # the volume/power/amplitude is actually zero rather than NaN
    c=np.rint(Power>np.mean(Power)-np.std(Power))
                      # if power is less than the threshold
                      # consider the volume is really low at that point
    c=c.astype(int)
    a=np.array([0])
    check=np.diff(np.concatenate((a,c,a), axis=1))
    SS=np.where(check==1)
    EE=np.where(check==-1)
    Prop[audio,0:3]=np.array([[50*(SS[0][0]+1),50*(201-EE[0][-1]),std(Power[SS[0]:EE[0][-1])]])
    # program continues... I'm updating all my Matlab code in Phython
    # I will try to get it done as soon as possible
  


    

