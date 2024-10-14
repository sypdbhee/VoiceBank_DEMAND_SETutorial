#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:23:43 2018

@author: winstonlin
"""
import matplotlib
import ipdb
import csv
import subprocess
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np
import h5py as hf
import multiprocessing as mp

from pysepm import fwSNRseg, SNRseg, composite, srmr
from pystoi import stoi
from pesq import pesq
from os.path import join
from os.path import sep as filesep
from scipy.io.wavfile import read as audioread
from scipy.io.wavfile import write as audiowrite

epsilon = np.finfo(float).eps
np.random.seed(0)

# 副程式
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', default=3, type=int)

    # CSV Output
    parser.add_argument('--output_csv', '-o', default='./enhanced.csv', type=str)

    # List Input
    parser.add_argument('listPath', type=str, default='test.txt')
    parser.add_argument('dgrdPath', type=str, default='./noisy')

    args = parser.parse_args()

    return args

def making_list_config(lst_pth_dict):

    fp = open(lst_pth_dict['lstpth'],'r')
    pth_lst = fp.readlines()

    MP_List = []
    for lin in pth_lst:
        MP_List.append([lin.strip().split(' ')[1], join(lst_pth_dict['enhpth'], lin.strip().split(' ')[2]), lin.strip().split(' ')[2]])

    fp.close()

    return {'mplstpth':MP_List, 'outcsvpath': lst_pth_dict['outpth'], 'size': len(MP_List)}

def ssnri(ref, enh, noy, winlen):

    refMat = libframing(ref, winlen, winlen).T
    enhMat = libframing(enh, winlen, winlen).T
    noyMat = libframing(noy, winlen, winlen).T

    frames = enhMat.shape[0]

    e_snr_dB = np.mean(snr(refMat[:frames], enhMat))
    n_snr_dB = np.mean(snr(refMat[:frames], noyMat[:frames]))
    return e_snr_dB - n_snr_dB

def snr(ref, deg):
    err = ref[:deg.shape[0]] - deg
    ref_pow = np.sum(np.abs(ref[:deg.shape[0]])**2, axis=1)
    err_pow = np.sum(np.abs(err)**2, axis=1)

    esnr = ref_pow/(err_pow + epsilon * (err_pow == 0).astype('int'))
    return 10 * np.log10(esnr + epsilon * (esnr == 0).astype('int'))

def MP_CalScore(CnPath, DgPath, FileName):

    Fs, degData = audioread(DgPath.strip())
    Fs, grtData = audioread(CnPath.strip())

    #[PESQScor, CsigScor, CbakScor, CovlScor, SSNRScor]=composite(CnPath, DgPath)
    
    points = min(grtData.shape[0], degData.shape[0])
    
    PESQScor = pesq(Fs, grtData[:points], degData[:points], 'wb')
    FwSSNRSc = fwSNRseg(grtData[:points], degData[:points], Fs)    
    SSNRScor = SNRseg(grtData[:points], degData[:points], Fs)
    CsigScor, CbakScor, CovlScor = composite(grtData[:points], degData[:points], Fs)
    StoiScor = stoi(grtData[:points], degData[:points], Fs)
    ESTOIScr = stoi(grtData[:points], degData[:points], Fs, extended=True)
    SRMRIScr = np.abs(srmr(degData[:points], Fs) - srmr(grtData[:points], Fs))

    EvlRes = {'pesq': PESQScor,
            'stoi': StoiScor,
            'estoi': ESTOIScr,
            'ssnr': SSNRScor,
            'csig': CsigScor,
            'cbak': CbakScor,
            'covl': CovlScor,
            'fwssnr': FwSSNRSc,
            'srmri': SRMRIScr,
            'name': FileName
            }

    return EvlRes

def MpScor2Dict(MpResults, lstSize):

    ScorMat = np.zeros((lstSize, 9)) # 9: PESQ, STOI, ESTOI, CSIG, CBAK, COVL, SSNR, fwSSNR, SRMRI
    AvegScor = np.zeros((7,))
    FileName = []

    for scrInd, ScrDict in enumerate(MpResults):
        ScorMat[scrInd, 0] = ScrDict['pesq']
        ScorMat[scrInd, 1] = ScrDict['stoi']
        ScorMat[scrInd, 2] = ScrDict['estoi']        
        ScorMat[scrInd, 3] = ScrDict['csig']
        ScorMat[scrInd, 4] = ScrDict['cbak']
        ScorMat[scrInd, 5] = ScrDict['covl']
        ScorMat[scrInd, 6] = ScrDict['ssnr']
        ScorMat[scrInd, 7] = ScrDict['fwssnr']
        ScorMat[scrInd, 8] = ScrDict['srmri']
        FileName.append(ScrDict['name'])

    AvegScor = np.mean(ScorMat, axis = 0)
    EvlRes = {'scor': ScorMat,
              'aveg': AvegScor,
              'name': FileName
             }

    return EvlRes

def SaveCsvFile(ScorPara, OutCsvPath):

    ## Score Organize
    # title part
    TitleList = [['', 'PESQ', 'STOI', 'ESTOI', 'CSIG', 'CBAK', 'COVL', 'SSNR', 'fwSSNR', 'SRMRI']]
    AvegeList = [['AVG.']]
    for scr in ScorPara['aveg'].tolist():
        AvegeList[0].append(scr)

    # Empty part
    BlankLine = [[''] * len(TitleList[0])]

    # score part
    UttesList = []
    for filInd, fileNme in enumerate(ScorPara['name']):
        UttesList.append([fileNme])
        for scr in ScorPara['scor'][filInd].tolist():
            UttesList[filInd].append(scr)

    # making table
    LargeTable = copy.deepcopy(TitleList)
    LargeTable.extend(copy.deepcopy(AvegeList))
    LargeTable.extend(copy.deepcopy(BlankLine))
    LargeTable.extend(copy.deepcopy(BlankLine))
    LargeTable.extend(copy.deepcopy(TitleList))
    LargeTable.extend(copy.deepcopy(UttesList))

    with open(OutCsvPath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(LargeTable)

# 主程式

if __name__ == '__main__':

    # get parameter
    args = get_args()

    # Multiprocessing
    cpu_count = mp.cpu_count()
    MPPool = mp.Pool(processes=args.num_workers)

    # Making processing list
    EvPthCfig = making_list_config({'lstpth':args.listPath, 'enhpth': args.dgrdPath, 'outpth': args.output_csv})

    # Evaluation Start
    MP_ResScor = MPPool.starmap(MP_CalScore, EvPthCfig['mplstpth'])  
    '''
    MP_ResScor = []
    for (cnpth, dgpth, flnme)in EvPthCfig['mplstpth']:
        MP_ResScor.append(MP_CalScore(cnpth, dgpth, flnme))
    '''
    ScorDict = MpScor2Dict(MP_ResScor, EvPthCfig['size'])

    # Output the results
    SaveCsvFile(ScorDict, EvPthCfig['outcsvpath'])
