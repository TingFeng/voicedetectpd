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
from scipy import stats, signal # for zsore, pearson correlation and find peaks
from sklearn.linear_model import LogisticRegression # for logistic regression
import csv # for read in csv files

# navigate to the folder of .wav files, each file is named with the audio ID
os.chdir('/mnt/hgfs/InsightVoiceData/output')
# List all the files so we can read them
AudioFile=os.listdir('/mnt/hgfs/InsightVoiceData/output')

# initiate the numpy array for eight features of each voice sample
Prop=np.zeros(shape=(len(AudioFile),8))
VoiceID=np.zeros(shape=(len(AudioFile),1)) # audio file name

count=0
for Audio in AudioFile:
    VoiceID[count]=np.asarray(Audio[0:7],dtype='float64')
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
    
    
    Coff=np.zeros(shape=(199,1)) # qantify the frequency consistency
    for i in np.arange(0,199):
        a=stats.pearsonr(Trans[i,],Trans[i+1,])
        Coff[i]=a[0]
    Prop[count,5]=np.mean(Coff) 
        
    # next we quantify the variance of peak frequency within each formant frequency    
    Formant=np.zeros(shape=(200,2));
    Range=[[500, 925], [925, 1225], [2000, 3000], [3000, 4000]];
    # putting four formant frequency range in a List
    # note these ranges are setted by experience, and only the 3rd and 4th formant are used as features
    for i in np.arange(0,200):
        for j in np.arange(0,2):
            t=np.where((freq>Range[j+2][0])*(freq<Range[j+2][1]))
            b=np.where(Trans[i,t]==np.max(Trans[i,t]))
            Formant[i][j]=t[0][b[1]]            
    Prop[count,6:8]=np.array([np.std(Formant[SS[0][0]:EE[0][-1],0]),np.std(Formant[SS[0][0]:EE[0][-1],1])])   
  
    count=count+1
    
# Great, now that we calculated all the eight features of voice
# let's load the meta data, mark the voice with people and medication stage

RecordID=np.zeros(shape=(15744,1)) # audio file name
MedID=np.zeros(shape=(15744,1)) # medication stage
HealthCode=np.empty(15744,dtype='S36') # user ID
cr=csv.reader(open('/mnt/hgfs/InsightVoiceData/VoiceInfo.csv')) # this table contains e.g. time, people, recoring ID 
count=0
for row in cr:    
    # print row[2],row[-2] 
    if count>0:  
        HealthCode[count-1]=row[3]
        if len(row[9])>0: # else it can be np.nan, for now we use 0
           RecordID[count-1]=np.asarray(row[9])                 
        # row[19] is the medication stage
        if len(row[19])==39:
            MedID[count-1]=np.array([4]) # immediately before medication
        elif len(row[19])==46:
            MedID[count-1]=np.array([2]) # immediately after medication (at the best)
        elif len(row[19])==12:
            MedID[count-1]=np.array([3]) # another time - intermedian stage
        elif len(row[19])>0:
            MedID[count-1]=np.array([1]) # do not take medication                       
    count=count+1
    
PatientID=np.empty(54,dtype='S36') # user ID
Diagnosis=np.zeros(shape=(54,1)) # professionally diagonosis 1; otherwise 0
cr=csv.reader(open('/mnt/hgfs/InsightVoiceData/PatientInfo.csv')) # this table contains patients' meta data
count=0
for row in cr:
    if count>0:
        PatientID[count-1]=row[3]
        if len(row[29])==6:
            Diagnosis[count-1]=np.array([1])
    count=count+1

# match PatientID with HealthCode
DiagnosisID=np.zeros(shape=(15744,2)) # patient ID, medication state                                     
Totalppl=np.unique(HealthCode) # the voice recordings come from 50 ppl (PatientInfo table has replicates)
pplDiag=np.zeros(len(Totalppl)) # 50ppl patient as 1, control as 0
for i in np.arange(0,len(Totalppl)):
    tt=np.where(PatientID==Totalppl[i])
    t=np.where(HealthCode==PatientID[tt[0][0]])
    DiagnosisID[t[0],0]=i
    DiagnosisID[t[0],1]=Diagnosis[tt[0][0]]
    pplDiag[i]=Diagnosis[tt[0][0]]

# match voice recordings with index (because some of the voice recordings are not download-able)

ID_Diagnosis=np.zeros(shape=(len(AudioFile),2))
count=0
for i in VoiceID:
    t=np.where(RecordID==i)
    if len(t[0])==0:
        VoiceID=np.delete(VoiceID,count,axis=0)
        Prop=np.delete(Prop,count,axis=0)
    else:               
        ID_Diagnosis[count,]=DiagnosisID[t[0],]
        count=count+1
if ID_Diagnosis[:,0].size>count:
    ID_Diagnosis=np.delete(ID_Diagnosis,np.arange(count,ID_Diagnosis[:,0].size),axis=0)
        
# a=[x for x in RecordID if not x in VoiceID]


# use logistic regression

# Note I'm only using voice recordings from control subjects
# and a subset from patients' voice recordings that are immediately taken before medication
# because there should be their personal worst

# Also I make sure that recordings from different people went into
# training and testing datasets, to avoide the common mistake that 
# using a subset of an individual's voice recording to predict the other subset recording from the same person

# traning set: t11 for control subjects, t21 for patients
# testing set: t12 for control subjects, t22 for patients

TT=np.where(pplDiag==0)
t11=np.where((ID_Diagnosis[:,1]==0)*(ID_Diagnosis[:,0]<=TT[0][3])*(~np.isnan(Prop.sum(axis=1)))*(Prop.sum(axis=1)!=0))
t12=np.where((ID_Diagnosis[:,1]==0)*(ID_Diagnosis[:,0]>TT[0][3])*(~np.isnan(Prop.sum(axis=1)))*(Prop.sum(axis=1)!=0))

TT=np.where(pplDiag==1)
t21=np.where((ID_Diagnosis[:,1]==4)*(ID_Diagnosis[:,0]<=TT[0][14])*(~np.isnan(Prop.sum(axis=1)))*(Prop.sum(axis=1)!=0))
t22=np.where((ID_Diagnosis[:,1]==4)*(ID_Diagnosis[:,0]>TT[0][19])*(ID_Diagnosis[:,0]<=TT[0][29])*(~np.isnan(Prop.sum(axis=1)))*(Prop.sum(axis=1)!=0))

x=np.append(Prop[t11[0]],Prop[t21[0]],axis=0)
y=np.append(np.zeros(shape=(t11[0].size,1)),np.ones(shape=(t21[0].size,1)),axis=0)

model = LogisticRegression()
model = model.fit(X, y)
# check the accuracy on the training set
model.score(X, y)

# for testing datasets
x=np.append(Prop[t12[0]],Prop[t22[0]],axis=0)
y=np.append(np.zeros(shape=(t12[0].size,1)),np.ones(shape=(t22[0].size,1)),axis=0)

# check the accuracy on the testing set
model.score(x, y)
