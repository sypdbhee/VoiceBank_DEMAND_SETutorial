#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun. Aug. 1 22:21 2021

@author: sypdbhee
"""

import numpy as np
import scipy.signal as ssg

import scipy
import csv
import scipy.io
import librosa
import random
import os
import sys
import ipdb
import h5py
import torch

from pesq import pesq
from pystoi.stoi import stoi
from os.path import join
from os.path import sep as filesep
from scipy.io.wavfile import read as audioread
from scipy.io.wavfile import write as audiowrite
from librosa.util import frame as libframing

epsilon = np.finfo(float).eps
np.random.seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)

def cal_score(clean,enhanced,noyData):

    enhanced = enhanced/np.std(enhanced) * np.std(clean)

    s_stoi = stoi(clean, enhanced, 16000)
    s_pesq = pesq(clean, enhanced, 16000)
    s_e_stoi = stoi(clean, enhanced, 16000, extended=True)
    s_ssnri = ssnri(clean, enhanced, noyData, 512)

    return round(s_pesq,5), round(s_stoi,5), round(s_e_stoi,5), round(s_ssnri,5)

def get_filepaths(directory,ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)
    
def MakingContextureFeature(InpFea, NUM_FRAME):

    InpFea = InpFea.T   #np.log10(abs(D)**2 + epsilon)
    Rt_Frames, Dims = InpFea.shape

    Frames = Rt_Frames
    if NUM_FRAME == 0 or NUM_FRAME == 1:
        NUM_FRAME = 1
        idx = 1
    elif NUM_FRAME == 2:
        BfSxx = np.expand_dims(InpFea[0,],axis=0)
        InpFea = np.concatenate((BfSxx, InpFea), axis=0)
        Frames += int(NUM_FRAME - 1)
        idx = NUM_FRAME
    else:
        BfSxx = np.dot(np.ones((int(np.floor(NUM_FRAME/2)), 1)), np.expand_dims(InpFea[0,],axis=0))
        AfSxx = np.dot(np.ones((int(np.floor(NUM_FRAME/2)), 1)), np.expand_dims(InpFea[-1,],axis=0))
        #BfSxx = np.dot(np.ones((NUM_FRAME - 2, 1)), np.expand_dims(InpFea[0,],axis=0))
        #AfSxx = np.expand_dims(InpFea[-1,],axis=0)
        InpFea = np.concatenate((BfSxx, InpFea, AfSxx), axis=0)
        Frames += int(NUM_FRAME - 1)
        idx = NUM_FRAME

    return_data = np.empty((Rt_Frames, int(NUM_FRAME), np.int32(Dims)), dtype=InpFea.dtype)

    while idx <= Frames:
        idx_start = idx - NUM_FRAME
        idx_end = idx_start + NUM_FRAME

        return_data[idx_start] = InpFea[idx_start:idx_end,]
        idx += 1
        
    return return_data.reshape(Rt_Frames, NUM_FRAME * Dims)    

def wav2spec(WavData, WinLen=512, NUM_FFT = 512, hop_length=256, NUM_FRAME = 1, returnPhase = False, returnCmplex = False):
    '''
    D = librosa.stft(y = np.float32(WavData),
                     n_fft = NUM_FFT,
                     hop_length = hop_length,
                     win_length = WinLen,
                     window = ssg.windows.hann, 
                     center = cTer)
                     
    ContexFeat = MakingContextureFeature(D, NUM_FRAME)
    '''
    Hann = torch.from_numpy(ssg.windows.hann(WinLen))
                     
    Spec = torch.stft(WavData,
                      n_fft = NUM_FFT,
                      hop_length = hop_length,
                      win_length = WinLen,
                      window = Hann.to(device),
                      center = True,
                      return_complex = True)

    rl = torch.real(Spec)
    im = torch.imag(Spec)

    if returnCmplex:
        return_data = np.concatenate((rl,im), axis=0)
    else:
        return_data = torch.abs(Spec)

    if returnPhase:
        return return_data, torch.atan2(im,rl)
    else:
        return return_data
                                        
def spec2wav(Inp_1, Inp_2, WinLen = 512, NUM_FFT = 512, hop_length = 256, UttLen = 16000, cmplxInp = False):

    Hann = torch.from_numpy(ssg.windows.hann(WinLen))
    InpSpec = make_spec(Inp_1.to(device), Inp_2.to(device), cmplxInp)

    try:
        return np.real(torch.istft(InpSpec,
                                   n_fft = NUM_FFT,
                                   hop_length = hop_length,
                                   win_length = WinLen,
                                   window = Hann.to(device),
                                   length = UttLen
                                   )
                       )
    except:
        return np.real(torch.istft(InpSpec,
                                   n_fft = NUM_FFT,
                                   hop_length = hop_length,
                                   win_length = WinLen,
                                   window = Hann.to(device)
                                   )
                       )

def make_spec(Inp_1, Inp_2, cmplxInp = False):
    if cmplxInp:
        return Inp_1 + 1j*Inp_2
    else:        
        return Inp_1 * torch.exp(1j * Inp_2)

def collate_fn_LSTM(data):
    """
    data: is a list of tuples 
    """

    epsilon = torch.finfo(float).eps
    bthsize = len(data)    
    points = data[0][0].shape[1]

    mode = data[0][-1]
    FFTInfo = data[0][-2]

    InpWavf = torch.zeros(bthsize, points, dtype = data[0][0].dtype)
    RCnWavf = torch.zeros(bthsize, points, dtype = data[0][1].dtype)

    for idx, dt in enumerate(data):
        InpWavf[idx] = dt[0]
        RCnWavf[idx] = dt[1]
    
    InpAmpl, InpPhas = wav2spec(InpWavf.to(device), 
                                WinLen = FFTInfo['winsize'], 
                                NUM_FFT = FFTInfo['fftsize'], 
                                hop_length = FFTInfo['hopsize'], 
                                NUM_FRAME = 1, 
                                returnPhase = True, 
                                returnCmplex = False
                               )

    RCnAmpl, RCnPhas = wav2spec(RCnWavf.to(device), 
                                WinLen = FFTInfo['winsize'], 
                                NUM_FFT = FFTInfo['fftsize'], 
                                hop_length = FFTInfo['hopsize'], 
                                NUM_FRAME = 1, 
                                returnPhase = True, 
                                returnCmplex = False
                               )

    if mode == 'train':
    
        RtDta = {'DegFeat': {'inplps': torch.permute(torch.log10(torch.pow(InpAmpl,2) + epsilon), (0,2,1)), 'inppha': InpPhas},
                 'TarFeat': {'tarlps': torch.permute(torch.log10(torch.pow(RCnAmpl,2) + epsilon), (0,2,1)), 'tarpha': RCnPhas},
                 'FFTInfo': FFTInfo,
                 'UtteLen': points
                }
        
    elif mode == 'valid':

        RtDta = {'DegFeat': {'inplps': torch.permute(torch.log10(torch.pow(InpAmpl,2) + epsilon), (0,2,1)), 'inppha': InpPhas},
                 'TarFeat': {'tarlps': torch.permute(torch.log10(torch.pow(RCnAmpl,2) + epsilon), (0,2,1)), 'tarpha': RCnPhas},
                 'FFTInfo': FFTInfo,
                 'UtteLen': points
                }
        
    elif mode == 'test':
    
        E_Pth_List = []
        for idx, dt in enumerate(data):
            E_Pth_List.append(dt[2])
            
        RtDta = {'DegFeat': {'inplps': torch.permute(torch.log10(torch.pow(InpAmpl,2) + epsilon), (0,2,1)), 'inppha': InpPhas},
                 'TarFeat': {'tarlps': torch.permute(torch.log10(torch.pow(RCnAmpl,2) + epsilon), (0,2,1)), 'tarpha': RCnPhas},
                 'FFTInfo': FFTInfo,
                 'UtteLen': points,
                 'EhPthls': E_Pth_List
                }

    return RtDta

class Dataset_LSTM():

    def __init__(self, PthDict, args, mode):

        self.n_paths = PthDict['noisy_path']
        self.c_paths = PthDict['clean_path']
        self.e_paths = PthDict['outpt_path']

        self.FFTInfo = {'fftsize': args.n_fft,
                        'winsize': args.frame_size,
                        'hopsize': args.hop_length
                        }

        self.mode = mode

        self.batch_wav_length = args.sampling_rate * 2 # three seconds
        
    def _extend_waveform_dim(self, wavData):
        
        Dtype = torch.double
       
        return torch.from_numpy(np.expand_dims(wavData, axis=0)/(2.0**15)).type(Dtype)

    def _concat_Wav_to_Defined_size(self,):

        points = self.RCnWavf.shape[1]
        while points <= self.batch_wav_length:
            self.InpWavf = torch.cat((self.InpWavf, self.InpWavf), axis = 1)
            self.RCnWavf = torch.cat((self.RCnWavf, self.RCnWavf), axis = 1)            
            points = self.InpWavf.shape[1]

    def _get_train_fea(self,index):
    
        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self._concat_Wav_to_Defined_size()
        
        points = self.RCnWavf.shape[1]
        self.St_Fr = random.randint(0, points - self.batch_wav_length)
        self.Ed_Fr = self.St_Fr + self.batch_wav_length

    def _get_valid_fea(self,index):

        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self._concat_Wav_to_Defined_size()
        
        points = self.RCnWavf.shape[1]
        self.St_Fr = random.randint(0, points - self.batch_wav_length)
        self.Ed_Fr = self.St_Fr + self.batch_wav_length

    def _get_test_fea(self,index):

        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self.St_Fr = 0
        self.Ed_Fr = self.St_Fr + self.InpWavf.shape[1]

    def __getitem__(self, index):

        if self.mode == 'train':
            self._get_train_fea(index)
           
            return self.InpWavf[:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,self.St_Fr:self.Ed_Fr], self.FFTInfo, self.mode
        elif self.mode == 'valid':
            self._get_valid_fea(index)

            return self.InpWavf[:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,self.St_Fr:self.Ed_Fr], self.FFTInfo, self.mode

        elif self.mode == 'test':
            self._get_test_fea(index)

            return self.InpWavf[:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,self.St_Fr:self.Ed_Fr], self.e_paths[index].strip(), self.FFTInfo, self.mode

    def __len__(self):  # return count of sample we have

        return len(self.n_paths)

  
def collate_fn_FCN(data):
    """
    data: is a list of tuples 
    """

    mode = data[0][-1]
    bthsize = len(data)    
    
    if mode == 'train':
    
        points = data[0][0].shape[2]
    
        InpWavf = torch.zeros(bthsize, 1, points, dtype = data[0][0].dtype)
        RCnWavf = torch.zeros(bthsize, 1, points, dtype = data[0][1].dtype)
        
        for idx, dt in enumerate(data):
            InpWavf[idx] = dt[0]
            RCnWavf[idx] = dt[1]
            
        RtDta = {'DegFeat': {'inpfeat':InpWavf},
                 'TarWavf': {'rcnwavf':RCnWavf}
                }
        
        return RtDta
                   
    elif mode == 'valid':
    
        points = data[0][0].shape[2]
    
        InpWavf = torch.zeros(bthsize, 1, points, dtype = data[0][0].dtype)
        RCnWavf = torch.zeros(bthsize, 1, points, dtype = data[0][1].dtype)
        
        for idx, dt in enumerate(data):
            InpWavf[idx] = dt[0]
            RCnWavf[idx] = dt[1]
            
        RtDta = {'DegFeat': {'inpfeat':InpWavf},
                 'TarWavf': {'rcnwavf':RCnWavf}
                }
        
        return RtDta
    
    elif mode == 'test':
    
        points = data[0][0].shape[2]
    
        InpWavf = torch.zeros(bthsize, 1, points, dtype = data[0][0].dtype)
        RCnWavf = torch.zeros(bthsize, 1, points, dtype = data[0][1].dtype)
        
        E_Pth_List = []
        for idx, dt in enumerate(data):
            InpWavf[idx] = dt[0]
            RCnWavf[idx] = dt[1]

            E_Pth_List.append(dt[2])
            
        RtDta = {'DegFeat': {'inpfeat':InpWavf},
                 'TarWavf': {'rcnwavf':RCnWavf},
                 'EhPthls': E_Pth_List
                }
        
        return RtDta

class Dataset_FCN():

    def __init__(self, PthDict, args, mode):

        self.n_paths = PthDict['noisy_path']
        self.c_paths = PthDict['clean_path']
        self.e_paths = PthDict['outpt_path']

        self.mode = mode

        self.batch_wav_length = args.sampling_rate * 2 # three seconds
        
    def _extend_waveform_dim(self, wavData):
        
        Dtype = torch.double
       
        return torch.from_numpy(np.expand_dims(np.expand_dims(wavData, axis=0), axis=0)/(2.0**15)).type(Dtype)

    def _concat_Wav_to_Defined_size(self,):

        points = self.RCnWavf.shape[2]
        while points <= self.batch_wav_length:
            self.InpWavf = torch.cat((self.InpWavf, self.InpWavf), axis = 2)
            self.RCnWavf = torch.cat((self.RCnWavf, self.RCnWavf), axis = 2)            
            points = self.InpWavf.shape[2]

    def _get_train_fea(self,index):
    
        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self._concat_Wav_to_Defined_size()
        
        points = self.RCnWavf.shape[2]
        self.St_Fr = random.randint(0, points - self.batch_wav_length)
        self.Ed_Fr = self.St_Fr + self.batch_wav_length

    def _get_valid_fea(self,index):

        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self._concat_Wav_to_Defined_size()
        
        points = self.RCnWavf.shape[2]
        self.St_Fr = random.randint(0, points - self.batch_wav_length)
        self.Ed_Fr = self.St_Fr + self.batch_wav_length

    def _get_test_fea(self,index):

        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self.St_Fr = 0
        self.Ed_Fr = self.St_Fr + self.InpWavf.shape[2]

    def __getitem__(self, index):

        if self.mode == 'train':
            self._get_train_fea(index)
           
            return self.InpWavf[:,:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,:,self.St_Fr:self.Ed_Fr], self.mode
        elif self.mode == 'valid':
            self._get_valid_fea(index)

            return self.InpWavf[:,:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,:,self.St_Fr:self.Ed_Fr], self.mode

        elif self.mode == 'test':
            self._get_test_fea(index)

            return self.InpWavf[:,:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,:,self.St_Fr:self.Ed_Fr], self.e_paths[index].strip(), self.mode

    def __len__(self):  # return count of sample we have

        return len(self.n_paths)

def collate_fn_DNN(data):
    """
    data: is a list of tuples 
    """

    epsilon = torch.finfo(float).eps
    bthsize = len(data)    
    points = data[0][0].shape[1]

    mode = data[0][-1]
    FFTInfo = data[0][-2]

    InpWavf = torch.zeros(bthsize, points, dtype = data[0][0].dtype)
    RCnWavf = torch.zeros(bthsize, points, dtype = data[0][1].dtype)

    for idx, dt in enumerate(data):
        InpWavf[idx] = dt[0]
        RCnWavf[idx] = dt[1]
    
    InpAmpl, InpPhas = wav2spec(InpWavf.to(device), 
                                WinLen = FFTInfo['winsize'], 
                                NUM_FFT = FFTInfo['fftsize'], 
                                hop_length = FFTInfo['hopsize'], 
                                NUM_FRAME = 1, 
                                returnPhase = True, 
                                returnCmplex = False
                               )

    RCnAmpl, RCnPhas = wav2spec(RCnWavf.to(device), 
                                WinLen = FFTInfo['winsize'], 
                                NUM_FFT = FFTInfo['fftsize'], 
                                hop_length = FFTInfo['hopsize'], 
                                NUM_FRAME = 1, 
                                returnPhase = True, 
                                returnCmplex = False
                               )

    if mode == 'train':
    
        RtDta = {'DegFeat': {'inplps': torch.permute(torch.log10(torch.pow(InpAmpl,2) + epsilon), (0,2,1)), 'inppha': InpPhas},
                 'TarFeat': {'tarlps': torch.permute(torch.log10(torch.pow(RCnAmpl,2) + epsilon), (0,2,1)), 'tarpha': RCnPhas},
                 'FFTInfo': FFTInfo,
                 'UtteLen': points
                }
        
    elif mode == 'valid':

        RtDta = {'DegFeat': {'inplps': torch.permute(torch.log10(torch.pow(InpAmpl,2) + epsilon), (0,2,1)), 'inppha': InpPhas},
                 'TarFeat': {'tarlps': torch.permute(torch.log10(torch.pow(RCnAmpl,2) + epsilon), (0,2,1)), 'tarpha': RCnPhas},
                 'FFTInfo': FFTInfo,
                 'UtteLen': points
                }
        
    elif mode == 'test':
    
        E_Pth_List = []
        for idx, dt in enumerate(data):
            E_Pth_List.append(dt[2])
            
        RtDta = {'DegFeat': {'inplps': torch.permute(torch.log10(torch.pow(InpAmpl,2) + epsilon), (0,2,1)), 'inppha': InpPhas},
                 'TarFeat': {'tarlps': torch.permute(torch.log10(torch.pow(RCnAmpl,2) + epsilon), (0,2,1)), 'tarpha': RCnPhas},
                 'FFTInfo': FFTInfo,
                 'UtteLen': points,
                 'EhPthls': E_Pth_List
                }

    return RtDta

class Dataset_DNN():

    def __init__(self, PthDict, args, mode):

        self.n_paths = PthDict['noisy_path']
        self.c_paths = PthDict['clean_path']
        self.e_paths = PthDict['outpt_path']

        self.FFTInfo = {'fftsize': args.n_fft,
                        'winsize': args.frame_size,
                        'hopsize': args.hop_length
                        }

        self.mode = mode

        self.batch_wav_length = args.sampling_rate * 2 # three seconds
        
    def _extend_waveform_dim(self, wavData):
        
        Dtype = torch.double
       
        return torch.from_numpy(np.expand_dims(wavData, axis=0)/(2.0**15)).type(Dtype)

    def _concat_Wav_to_Defined_size(self,):

        points = self.RCnWavf.shape[1]
        while points <= self.batch_wav_length:
            self.InpWavf = torch.cat((self.InpWavf, self.InpWavf), axis = 1)
            self.RCnWavf = torch.cat((self.RCnWavf, self.RCnWavf), axis = 1)            
            points = self.InpWavf.shape[1]

    def _get_train_fea(self,index):
    
        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self._concat_Wav_to_Defined_size()
        
        points = self.RCnWavf.shape[1]
        self.St_Fr = random.randint(0, points - self.batch_wav_length)
        self.Ed_Fr = self.St_Fr + self.batch_wav_length

    def _get_valid_fea(self,index):

        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self._concat_Wav_to_Defined_size()
        
        points = self.RCnWavf.shape[1]
        self.St_Fr = random.randint(0, points - self.batch_wav_length)
        self.Ed_Fr = self.St_Fr + self.batch_wav_length

    def _get_test_fea(self,index):

        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs, ClnData = audioread(self.c_paths[index].strip())
        
        self.InpWavf = self._extend_waveform_dim(NoyData)
        self.RCnWavf = self._extend_waveform_dim(ClnData)
        
        self.St_Fr = 0
        self.Ed_Fr = self.St_Fr + self.InpWavf.shape[1]

    def __getitem__(self, index):

        if self.mode == 'train':
            self._get_train_fea(index)
           
            return self.InpWavf[:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,self.St_Fr:self.Ed_Fr], self.FFTInfo, self.mode
        elif self.mode == 'valid':
            self._get_valid_fea(index)

            return self.InpWavf[:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,self.St_Fr:self.Ed_Fr], self.FFTInfo, self.mode

        elif self.mode == 'test':
            self._get_test_fea(index)

            return self.InpWavf[:,self.St_Fr:self.Ed_Fr], self.RCnWavf[:,self.St_Fr:self.Ed_Fr], self.e_paths[index].strip(), self.FFTInfo, self.mode

    def __len__(self):  # return count of sample we have

        return len(self.n_paths)
