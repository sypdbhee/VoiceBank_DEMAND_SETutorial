#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu. May 12, 2022

@author: sypdbhee
"""

import os
import ipdb
import argparse
import torch

from os.path import join
from os.path import sep as filesep

from torch.utils.data import DataLoader
from torchsummary import summary as sumary

from proc import fix_seed, Mdl_Config, check_path

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model setting
    parser.add_argument('--model_save_path', default=os.getcwd(), type=str)
    parser.add_argument('--model_name', choices=['LSTM', 'BLSTM', 'DNN', 'FCN','EnsDNN','EnsLSTM','EnsBLSTM','EnsFCN'], default='DNN', type=str)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--seed', default=999, type=int)   
    parser.add_argument('--mode', default='train', choices=['train','test'], type=str)
    parser.add_argument('--pretrain_mdl_pth', default='None', type=str)

    # training specifics
    parser.add_argument('--batch_size', '-bs', default=1, type=int)    
    parser.add_argument('--num_epochs', '-e', default=100, type=int)
    parser.add_argument('--start_epochs', '-se', default=1, type=int)
    parser.add_argument('--warm_up_epochs', default=10, type=int)     
    
    # Optimizer
    parser.add_argument('--loss_wei', default='1,1', type=str)
    parser.add_argument('--optim', type=str, choices=['adam', 'nadam', 'radam'], default='nadam')
    parser.add_argument('--scheduler', type=str, default="steplr", help='Learning rate scheduler')
    parser.add_argument('--learning_rate','-lr', default=1e-3, type=float)
    parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');
    parser.add_argument('--test_interval', type=int, default=10, help='Test and save every [test_interval] epochs');

    # Data loader
    parser.add_argument('--sampling_rate', '-sr', default=16000, type=int)
    parser.add_argument('--frame_size', default=400, type=int) # 25 ms
    parser.add_argument('--n_fft', default=512, type=int)
    parser.add_argument('--hop_length', default=160, type=int) # 10 ms
    
    # Others
    parser.add_argument('fir_list', type=str, default='firInp_list')
    parser.add_argument('sec_list', type=str, default='secInp_list')
    
    args = parser.parse_args()
    
    return args

def get_path(args):

    RotFolder = args.model_save_path
    
    MdlRotFdr = os.path.join(RotFolder, f'DL-{args.model_name}-NFFT{args.n_fft}-Hop{args.hop_length}')
    MdlRotFdr = MdlRotFdr + f'-LossWei{args.loss_wei}'
        
    CptFolder = os.path.join(MdlRotFdr, 'checkpoint')
    MdlFolder = os.path.join(MdlRotFdr, 'bst_mdl')
    if not os.path.exists(CptFolder):
        os.makedirs(CptFolder)
    if not os.path.exists(MdlFolder):
        os.makedirs(MdlFolder)
        
    model_path = os.path.join(MdlFolder, 'mdl_save.pkl')
    
    return CptFolder, model_path
    
def making_list(lst_pth_dict, mode='train'):

    fp = open(lst_pth_dict['lstpth'],'r')
    pth_lst = fp.readlines()

    if mode == 'train':
    
        NyList = []
        CnList = []
        EnList = None       
        
        for lin in pth_lst:        
            NyList.append(lin.strip().split(' ')[0])
            CnList.append(lin.strip().split(' ')[1])            
        
    elif mode == 'test':
    
        NyList = []
        CnList = []
        EnList = []
    
        for lin in pth_lst:
            NyList.append(lin.strip().split(' ')[0])
            CnList.append(lin.strip().split(' ')[1])
            EnList.append(join(lst_pth_dict['enhpth'], lin.strip().split(' ')[2]))
        
        for lin in EnList:
            check_path(lin)
    
    fp.close()
        
    return {'noisy_path':NyList, 'clean_path':CnList, 'outpt_path':EnList}

if __name__ == '__main__':
    # get parameter
    args = get_args()
    
    # seed
    if args.seed:
        fix_seed(args.seed)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')

    mdlNme = args.model_name
    
    exec (f"from models import {args.model_name} as Mdl")               #ex. from models import DNN as Mdl
    if any(mdlNme.lower() == mdlnme for mdlnme in ['lstm','enslstm','blstm','ensblstm']):
        from proc import AgentTrTserLSTM as AgenTrTs
        from util import Dataset_LSTM as DtSe
        from util import collate_fn_LSTM as Cofn
    elif any(mdlNme.lower() == mdlnme for mdlnme in ['dnn','ensdnn']): #ex. from util import Dataset_DNN as DtSe
        from proc import AgentTrTserDNN as AgenTrTs
        from util import Dataset_DNN as DtSe
        from util import collate_fn_DNN as Cofn
    elif any(mdlNme.lower() == mdlnme for mdlnme in ['fcn']): #ex. from util import collate_fn_DNN as Cofn
        from proc import AgentTrTserFCN as AgenTrTs
        from util import Dataset_FCN as DtSe
        from util import collate_fn_FCN as Cofn

    # declair path
    checkpoint_path, model_path = get_path(args)
        
    Tr_Tser = AgenTrTs(device, args)
    if args.mode == 'train':
    
        # model configure
        Mdl, mdl_cfg = Mdl_Config(args, Mdl(args), checkpoint_path, model_path, device)
        
        sumary(Mdl.to(device))
        
        TrPthDict = making_list({'lstpth':args.fir_list}, mode=args.mode)
        VaPthDict = making_list({'lstpth':args.sec_list}, mode=args.mode)
        
        TrDataLdr = { 
            'tr' : DataLoader(DtSe(TrPthDict, args, args.mode), batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn = Cofn),
            'va' : DataLoader(DtSe(VaPthDict, args, 'valid'), batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn = Cofn)
            }                
        
        Tr_Tser.train(Mdl, mdl_cfg, TrDataLdr)
                      
    if args.mode == 'test':
    
        TsPthDict = making_list({'lstpth':args.fir_list, 'enhpth': args.sec_list}, mode=args.mode)
        
        TsDtaLder = DataLoader(DtSe(TsPthDict, args, args.mode), batch_size=1, num_workers=0, collate_fn = Cofn)
        Tr_Tser.test(Mdl(args), {'mdlpth':model_path}, TsDtaLder)
    
