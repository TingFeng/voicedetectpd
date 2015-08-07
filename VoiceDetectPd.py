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
from scipy import stats, signal # for zsore and find peaks
import csv # for read in csv files

# navigate to the folder of .wav files, each file is named with the audio ID
os.chdir('/mnt/hgfs/InsightVoiceData/output')
# List all the files so we can read them
AudioFile=os.listdir('/mnt/hgfs/InsightVoiceData/output')

# initiate the numpy array for eight features of each voice sample
Prop=np.zeros(shape=(len(AudioFile),8))

count=0
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
    Power=Trans.sum(axis=1)# note this is not the actual power
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
    # the first three features are the timing of voice onset, the early stop time, and the loudness variance
    Prop[count,0:3]=np.array([50*(SS[0][0]),50*(200-EE[0][-1]),np.std(Power[SS[0][0]:EE[0][-1]])])
    
    a=np.polyfit(np.arange(SS[0][0],EE[0][-1],1), Power[SS[0][0]:EE[0][-1]], 1)
    Prop[count,3]=a[0]
    
    Pitch=np.zeros(shape=(200,1)) # pitch in every 50ms
    for Index in np.arange(0,200,step=1): 
        a=signal.find_peaks_cwt(Trans[Index,],np.arange(1,15)) 
        # each peak cover no more than 200Hz, minimum distance between peaks has to be more than 50Hz 
        # this function is similar as findpeaks in MATLAB, but I do not think it is working as well
        a=np.asarray(a)
        b=np.where(Trans[Index,a]>np.mean(Trans[Index,])+3*np.std(Trans[Index,]))
        # I want to remove peaks in frequency domain that is too small
        b=np.asarray(b)
        a=a[b[0]]
        if len(a)>1:
            Pitch[Index]=np.min([a[0],np.min(np.diff(a))])
        elif len(a)==1:
            Pitch[Index]=a[0] # pitch should be the first harmonic frequency
                          # but just in case python find_peaks_cwt is not accurate
                          # I also calculated the (minimum) pitch difference between harmonics (which should be equivanlent with pitch)
        else:
            Pitch[Index]=np.nan
        
    # again, remove frequency that are outlier, presumably from inaccurate measurement of peak function
    t=np.where(Pitch[SS[0][0]:EE[0][-1]]-np.nanmean(Pitch[SS[0][0]:EE[0][-1]])<25)  # remove +500 Hz outlier 
    # need to calculate zscore variantion because every person's pitch is different
    Prop[count,4]=np.std(stats.zscore(Pitch[SS[0][0]+t[0]]))
    
    
    count=count+1
    # program continues... I'm updating all my Matlab code in Phython
    # I will try to get it done as soon as possible
