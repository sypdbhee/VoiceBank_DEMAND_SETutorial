#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun. Aug. 1 22:21 2021

@author: sypdbhee
"""
import ipdb
import torch

import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm

torch.autograd.set_detect_anomaly(True)

class FCNBlock(nn.Module):
    def __init__(self, InpChal = 30, OutChal = 30, KerSize = 55):
        super(FCNBlock,self).__init__()

        self.Conv1DLayer = nn.Conv1d(InpChal, OutChal, KerSize, padding = 'same')
        self.InsNorLayer = nn.InstanceNorm1d(OutChal)
        self.ActivaLayer = nn.LeakyReLU()

    def forward(self, Inp):

        LyrOut = self.Conv1DLayer(Inp)
        LyrOut = self.InsNorLayer(LyrOut)
        MdlOut = self.ActivaLayer(LyrOut)

        return MdlOut

class OutFCNBlock(nn.Module):
    def __init__(self, InpChal = 30, OutChal = 1, KerSize = 55):
        super(OutFCNBlock,self).__init__()

        self.Conv1DLayer = nn.Conv1d(InpChal, OutChal, KerSize, padding = 'same')
        self.ActivaLayer = nn.Tanh()

    def forward(self, Inp):

        LyrOut = self.Conv1DLayer(Inp)
        MdlOut = self.ActivaLayer(LyrOut)

        return MdlOut

class FCN(nn.Module):
    def __init__(self, args):
        super(FCN,self).__init__()
        
        self.FCNBlockInp = FCNBlock(1,30,55)
        self.FCNBlockFir = FCNBlock(30,30,55)
        self.FCNBlockSec = FCNBlock(30,30,55)
        self.FCNBlockThi = FCNBlock(30,30,55)
        self.FCNBlockFou = FCNBlock(30,30,55)
        self.FCNBlockFiv = FCNBlock(30,30,55)
        self.FCNBlockSix = FCNBlock(30,30,55)
        self.OutFCNBlock = OutFCNBlock(30,1,55)
            
    def forward(self, Inp):

        LyrOut = self.FCNBlockInp(Inp)
        LyrOut = self.FCNBlockFir(LyrOut)  
        LyrOut = self.FCNBlockSec(LyrOut)  
        LyrOut = self.FCNBlockThi(LyrOut)      
        LyrOut = self.FCNBlockFou(LyrOut)  
        LyrOut = self.FCNBlockFiv(LyrOut) 
        LyrOut = self.FCNBlockSix(LyrOut)
        MdlOut = self.OutFCNBlock(LyrOut)  

        return MdlOut

class LSTMBlock(nn.Module):
    def __init__(self, InpSize = 512, OutSize = 512, BiDir = False):
        super(LSTMBlock,self).__init__()

        self.LSTMLayer = nn.LSTM(InpSize, OutSize, batch_first = True, bidirectional = BiDir)
        self.ActivaLayer = nn.ReLU()

    def forward(self, Inp):

        LyrOut, _ = self.LSTMLayer(Inp)
        MdlOut = self.ActivaLayer(LyrOut)

        return MdlOut

class OutFFBlock(nn.Module):
    def __init__(self, InpSize = 512, OutSize = 512):
        super(OutFFBlock,self).__init__()

        self.LinearLayer = nn.Linear(InpSize, OutSize)

    def forward(self, Inp):

        MdlOut = self.LinearLayer(Inp)

        return MdlOut


class LSTM(nn.Module):
    def __init__(self, args, BiDir = False):
        super(LSTM,self).__init__()

        InitLayerSize = 384 
        if BiDir:
            LayerSize = InitLayerSize * 2
        else:
            LayerSize = InitLayerSize

        self.LSTMBlockInp = LSTMBlock(257,InitLayerSize,BiDir)
        self.LSTMBlockFir = LSTMBlock(LayerSize,InitLayerSize,BiDir)
        self.LSTMBlockSec = LSTMBlock(LayerSize,InitLayerSize,BiDir)
        self.LSTMBlockThi = LSTMBlock(LayerSize,InitLayerSize,BiDir)
        self.LSTMBlockFou = LSTMBlock(LayerSize,InitLayerSize,BiDir)
        self.LSTMBlockFiv = LSTMBlock(LayerSize,InitLayerSize,BiDir)
        self.OutFFBlock = OutFFBlock(LayerSize, 257)
            
    def forward(self, Inp):
        
        LyrOut = self.LSTMBlockInp(Inp)
        LyrOut = self.LSTMBlockFir(LyrOut)  
        LyrOut = self.LSTMBlockSec(LyrOut)  
        LyrOut = self.LSTMBlockThi(LyrOut)      
        LyrOut = self.LSTMBlockFou(LyrOut)      
        LyrOut = self.LSTMBlockFiv(LyrOut)      
        MdlOut = self.OutFFBlock(LyrOut)  

        return MdlOut

class BLSTM(nn.Module):
    def __init__(self, args):
        super(BLSTM,self).__init__()
        
        self.LSTMMdl = LSTM(args, True)
            
    def forward(self, Inp):
    
        return self.LSTMMdl(Inp)

class DNNBlock(nn.Module):
    def __init__(self, InpSize = 512, OutSize = 512):
        super(DNNBlock,self).__init__()

        self.DNNLayer = nn.Linear(InpSize, OutSize)
        self.ActivaLayer = nn.ReLU()

    def forward(self, Inp):

        LyrOut = self.DNNLayer(Inp)
        MdlOut = self.ActivaLayer(LyrOut)

        return MdlOut

class OutFFBlock(nn.Module):
    def __init__(self, InpSize = 512, OutSize = 512):
        super(OutFFBlock,self).__init__()

        self.LinearLayer = nn.Linear(InpSize, OutSize)

    def forward(self, Inp):

        MdlOut = self.LinearLayer(Inp)

        return MdlOut

class DNN(nn.Module):
    def __init__(self, args):
        super(DNN,self).__init__()

        LayerSize = 1024

        self.DNNBlockInp = DNNBlock(257,LayerSize)
        self.DNNBlockFir = DNNBlock(LayerSize,LayerSize)
        self.DNNBlockSec = DNNBlock(LayerSize,LayerSize)
        self.DNNBlockThi = DNNBlock(LayerSize,LayerSize)
        self.DNNBlockFou = DNNBlock(LayerSize,LayerSize)
        self.DNNBlockFif = DNNBlock(LayerSize,LayerSize)
        self.DNNBlockSix = DNNBlock(LayerSize,LayerSize)
        self.OutFFBlock = OutFFBlock(LayerSize, 257)
            
    def forward(self, Inp):
        
        LyrOut = self.DNNBlockInp(Inp)
        LyrOut = self.DNNBlockFir(LyrOut)  
        LyrOut = self.DNNBlockSec(LyrOut)  
        LyrOut = self.DNNBlockThi(LyrOut)      
        LyrOut = self.DNNBlockFou(LyrOut)      
        LyrOut = self.DNNBlockFif(LyrOut)      
        LyrOut = self.DNNBlockSix(LyrOut)      
        MdlOut = self.OutFFBlock(LyrOut)  

        return MdlOut

class EncLSTM(nn.Module):
    def __init__(self, InpSize = 257, LaySize = 384, OutSize = 384, BiDir = False):
        super(EncLSTM,self).__init__()

        if BiDir:
            LayerSize = LaySize * 2
        else:
            LayerSize = LaySize

        self.LSTMBlockInp = LSTMBlock(InpSize,LaySize,BiDir)
        self.LSTMBlockFir = LSTMBlock(LayerSize,LaySize,BiDir)
        self.LSTMBlockSec = LSTMBlock(LayerSize,OutSize,BiDir)
            
    def forward(self, Inp):
        
        LyrOut = self.LSTMBlockInp(Inp)
        LyrOut = self.LSTMBlockFir(LyrOut)  
        MdlOut = self.LSTMBlockSec(LyrOut)  

        return MdlOut

class DecLSTM(nn.Module):
    def __init__(self, InpSize = 384, LaySize = 384, OutSize = 257, BiDir = False):
        super(DecLSTM,self).__init__()

        if BiDir:
            LayerSize = LaySize * 2
        else:
            LayerSize = LaySize

        self.LSTMBlockThi = LSTMBlock(InpSize,LaySize,BiDir)
        self.LSTMBlockFou = LSTMBlock(LayerSize,LaySize,BiDir)
        self.LSTMBlockFiv = LSTMBlock(LayerSize,LaySize,BiDir)
        self.OutFFBlock = OutFFBlock(LayerSize, OutSize)
            
    def forward(self, Inp):
        
        LyrOut = self.LSTMBlockThi(Inp)      
        LyrOut = self.LSTMBlockFou(LyrOut)      
        LyrOut = self.LSTMBlockFiv(LyrOut)      
        MdlOut = self.OutFFBlock(LyrOut)  

        return MdlOut

class EnsLSTM(nn.Module):
    def __init__(self, args, InpSize = 257, BiDir = False):
        super(EnsLSTM,self).__init__()

        InitLayerSize = 384
        if BiDir:
            MulFactor = 4
        else:
            MulFactor = 2

        self.FEncLSTM = EncLSTM(InpSize,InitLayerSize,InitLayerSize,BiDir)
        self.SEncLSTM = EncLSTM(InpSize,InitLayerSize,InitLayerSize,BiDir)
        self.ODecLSTM = DecLSTM(InitLayerSize * MulFactor,InitLayerSize,257,BiDir)
            
    def forward(self, Inp):
        
        FirOut = self.FEncLSTM(Inp)      
        SecOut = self.SEncLSTM(Inp)      
        LayOut = torch.cat((FirOut, SecOut), -1)
        MdlOut = self.ODecLSTM(LayOut)  

        return MdlOut

class EncDNN(nn.Module):
    def __init__(self, InpSize = 257, LaySize = 1024, OutSize = 257):
        super(EncDNN,self).__init__()

        LayerSize = LaySize

        self.DNNBlockInp = DNNBlock(InpSize,LayerSize)
        self.DNNBlockFir = DNNBlock(LayerSize,LayerSize)
        self.DNNBlockSec = DNNBlock(LayerSize,LayerSize)
        self.DNNBlockThi = DNNBlock(LayerSize,OutSize)
            
    def forward(self, Inp):
        
        LyrOut = self.DNNBlockInp(Inp)
        LyrOut = self.DNNBlockFir(LyrOut)  
        LyrOut = self.DNNBlockSec(LyrOut)  
        MdlOut = self.DNNBlockThi(LyrOut)      

        return MdlOut

class DecDNN(nn.Module):
    def __init__(self,  InpSize = 512, LaySize = 1024, OutSize = 257):
        super(DecDNN,self).__init__()

        LayerSize = LaySize

        self.DNNBlockFou = DNNBlock(InpSize,LayerSize)
        self.DNNBlockFif = DNNBlock(LayerSize,LayerSize)
        self.DNNBlockSix = DNNBlock(LayerSize,LayerSize)
        self.OutFFBlock = OutFFBlock(LayerSize, OutSize)
            
    def forward(self, Inp):
        
        LyrOut = self.DNNBlockFou(Inp)      
        LyrOut = self.DNNBlockFif(LyrOut)      
        LyrOut = self.DNNBlockSix(LyrOut)      
        MdlOut = self.OutFFBlock(LyrOut)  

        return MdlOut

class EnsDNN(nn.Module):
    def __init__(self, args):
        super(EnsDNN,self).__init__()

        LayerSize = 1024

        self.FEncDNN = EncDNN(257,LayerSize,LayerSize)
        self.SEncDNN = EncDNN(257,LayerSize,LayerSize)
        self.ODecDNN = DecDNN(LayerSize * 2,LayerSize,257)
            
    def forward(self, Inp):
        
        FirOut = self.FEncDNN(Inp)      
        SecOut = self.SEncDNN(Inp)      
        LayOut = torch.cat((FirOut, SecOut), -1)
        MdlOut = self.ODecDNN(LayOut)  

        return MdlOut

class EnsBLSTM(nn.Module):
    def __init__(self, args, InpSize = 257):
        super(EnsBLSTM,self).__init__()
        
        self.EnsLSTMMdl = EnsLSTM(args, InpSize, True)
            
    def forward(self, Inp):
    
        return self.EnsLSTMMdl(Inp)


