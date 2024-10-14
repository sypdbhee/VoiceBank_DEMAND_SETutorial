#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu. May 12, 2022

@author: sypdbhee
"""

import os
import ipdb
import argparse
import random

from os.path import join
from os.path import sep as filesep

class MakingList:
    def __init__(self, InpFold = '.'):

        self.TpList = []
        for dirpth, _, filname in os.walk(InpFold.strip()):
            pre_path = dirpth.split(InpFold)[1]
            for fnme in filname:
                extend = fnme.split('.')[-1]
                if extend.lower() == 'wav':
                    self.TpList.append(join(pre_path,fnme))

    def __getitem__(self,idx):

        return self.TpList[idx]

    def __len__(self):

        return len(self.TpList)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model setting
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Data located
    parser.add_argument('noisy_folder_path', type=str, default='noy_folder')
    parser.add_argument('clean_folder_path', type=str, default='cln_folder')
    parser.add_argument('save_to_list_fold', type=str, default='sav_folder')
    
    args = parser.parse_args()
    
    return args

def making_Tr_list(noy_pth, cln_pth):

    GetPth = MakingList(noy_pth)
    
    TrNyList = []
    TrCnList = []
    VaNyList = []
    VaCnList = []

    for pth in GetPth:
        speaker = pth.split(filesep)[-1].split('_')[0]
        if (speaker == 'p226') or (speaker == 'p287'):
            VaNyList.append(join(noy_pth,pth))
            VaCnList.append(join(cln_pth,pth))
        else:
            TrNyList.append(join(noy_pth,pth))
            TrCnList.append(join(cln_pth,pth))            

    return TrNyList, TrCnList, VaNyList, VaCnList

def making_Ts_list(noy_pth, cln_pth):

    GetPth = MakingList(noy_pth)
    
    NyList = []
    CnList = []
    FiName = []

    for pth in GetPth:
        NyList.append(join(noy_pth,pth))
        CnList.append(join(cln_pth,pth))            
        FiName.append(pth)

    return NyList, CnList, FiName

if __name__ == '__main__':
    # get parameter
    args = get_args()

    if args.mode == 'train':
        TrNyList, TrCnList, VaNyList, VaCnList = making_Tr_list(args.noisy_folder_path,args.clean_folder_path)

        with open(join(args.save_to_list_fold,'vctk_training.list'),'w') as fp:
            for noypth, clnpth in zip(TrNyList,TrCnList):
                fp.write(f'{noypth} {clnpth}\n')

        with open(join(args.save_to_list_fold,'vctk_valid.list'),'w') as fp:
            for noypth, clnpth in zip(VaNyList,VaCnList):
                fp.write(f'{noypth} {clnpth}\n')

    elif args.mode == 'test':
        NyList, CnList, DtaIfo = making_Ts_list(args.noisy_folder_path,args.clean_folder_path)

        with open(join(args.save_to_list_fold,'vctk_testing.list'),'w') as fp:
            for noypth, clnpth, info in zip(NyList,CnList, DtaIfo):
                fp.write(f'{noypth} {clnpth} {info}\n')
