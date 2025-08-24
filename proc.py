#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun. Aug. 1 22:21 2021

@author: sypdbhee
"""
import librosa
import random
import os
import subprocess
import sys
import argparse
import ipdb
import torch
import h5py
import math
import tensorrt

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal as ssg
import multiprocessing as mp
#from torch.contrib.optim import SWA

from torch.utils.data import DataLoader
from torch.optim import Adam, NAdam, RAdam
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchsummary import summary as sumary

from os.path import join
#from kmeans_pytorch import kmeans_predict
from pesq import pesq, pesq_batch
from tqdm import tqdm

from os.path import sep as filesep
from shutil import copy as cp
from scipy.io.wavfile import write as audiowrite

from util import wav2spec, spec2wav, make_spec

epsilon = np.finfo(float).eps
np.random.seed(0)

# fix random
def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_path(inpPath):
    folder = ''.join([i + filesep for i in inpPath.split(filesep)[:-1]])
    if not os.path.exists(folder):
        os.makedirs(folder)

def Mdl_Config(args, model, checkpoint_path, model_path, device = 'cuda'):

    optimizers = {
        'adam' : Adam(model.parameters(), lr=args.learning_rate),
        'nadam': NAdam(model.parameters(), lr=args.learning_rate),
        'radam': RAdam(model.parameters(), lr=args.learning_rate)
        }

    optimizer = optimizers[args.optim]

    para = count_parameters(model)
    print(f'Num of model parameter : {para}\n')

    # tensorboard
    writer = SummaryWriter(checkpoint_path)
    writer.add_hparams(vars(args), dict())

    #model.apply(weights_init)
    if args.pretrain_mdl_pth != 'None':
        model = load_pre_train_mdl(model, join(args.pretrain_mdl_pth, 'mdl_save.pkl'), device)
        #checkpoint = torch.load(join(args.pretrain_mdl_pth, 'mdl_save.pkl'))
        #model.load_state_dict(checkpoint['model_state_dict'])
        ##for LayNme, Param in model.named_parameters():
        ##    if LayNme.split('.')[0] != 'last_layer':
        ##        Param.requires_grad = False

    #criterion = importlib.import_module('loss').__getattribute__(loss_fn)
    criterion = LossFun(args)
    
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * ( math.cos((epoch - args.warm_up_epochs) /(args.num_epochs - args.warm_up_epochs) * math.pi) + 1)

    normal_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.test_interval, gamma=args.lr_decay)    
    warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    return model, {'optmizr':optimizer, 'lr_schr':{'warm_up':warm_up_scheduler, 'normal':normal_scheduler}, 'critern':criterion, 'mdlpth':model_path, 'writer': writer}

def update_all_model(mdl):

    for LayNme, Param in dict(mdl.state_dict()).items():
        mdl.state_dict()[LayNme].requires_grad_(True)

    return mdl

def load_pre_train_mdl(model, pretrain_mdl_pth, device = 'cuda'):

    pre_train_mdl = torch.load(pretrain_mdl_pth,map_location=device)    
    Num_of_layers = [LayNme.split('.')[-2] for LayNme in pre_train_mdl['model_state_dict'].keys()][-1]
    for LayNme, Param in dict(pre_train_mdl['model_state_dict']).items():
        if LayNme.split('.')[-2] != Num_of_layers:
            model.state_dict()[LayNme].copy_(Param)
        else:
            tar_dim = model.state_dict()[LayNme].shape[0]
            sor_dim = Param.shape[0]
            
            m_dim = np.minimum(tar_dim, sor_dim)
            model.state_dict()[LayNme].copy_(Param[0:m_dim])
            
    return model

class LossFun(nn.Module):
    def __init__(self, args):
        super(LossFun, self).__init__()
        
        self.Dtype = torch.float

        self.weight = torch.from_numpy(np.array(args.loss_wei.split(',')).astype('float64')).type(self.Dtype)

        self.MSELoss = nn.MSELoss()        
        self.L1NLoss = nn.L1Loss()
        self.CEtLoss = nn.CrossEntropyLoss()
        self.L1SLoss = nn.SmoothL1Loss()        

        self.MdlNme = args.model_name

    def forward(self, DecPred, DecTar):
    
        if any(self.MdlNme.lower() == mdlnme for mdlnme in ['lstm','enslstm','blstm','ensblstm','ensdnn', 'dnn']):
            WavfOut = torch.real(DecPred['wavfout']).type(self.Dtype)
            ModlOut = torch.real(DecPred['mdloupt']).type(self.Dtype)
        
            WavfRef = torch.real(DecTar['wavfRef']).type(self.Dtype)
            ModlRef = torch.real(DecTar['modlRef']).type(self.Dtype)
        elif any(self.MdlNme.lower() == mdlnme for mdlnme in ['ensfcn', 'fcn']):
            WavfOut = torch.real(DecPred['mdloupt']).type(self.Dtype)
            WavfRef = torch.real(DecTar['modlRef']).type(self.Dtype)
            ModlOut = WavfOut
            ModlRef = WavfRef

        f_loss = self.L1NLoss(ModlOut, ModlRef)
        s_loss = self.MSELoss(WavfOut, WavfRef)
           
        t_loss = self.weight[0] * f_loss + self.weight[1] * s_loss
        
        return {'fullLoss': t_loss, 'wavfLoss': s_loss, 'moutLoss': f_loss}
        
class AgentTrTserFCN:
    def __init__(self, device, args):

        self.device = device
        self.Args = args
        
        self.Dtype = torch.float
        
    def train(self, InpMdl, Mdl_Cfg, DtaLoader):

        self._pre_setting_for_train(InpMdl, Mdl_Cfg, DtaLoader)

        Curr_epoch = self.Args.start_epochs
        Best_PESQ = 0.0

        self.CalPESQMod = self.Args.test_interval

        modSavInd = False

        while Curr_epoch <= self.Args.num_epochs:
            self._train_epoch(Curr_epoch)
            self._val_epoch(Curr_epoch)
            
            self.writer.add_scalar('TrFullLoss', self.TrFullLoss, Curr_epoch)
            self.writer.add_scalar('TrWavfLoss', self.TrWavfLoss, Curr_epoch)
            self.writer.add_scalar('TrMFeaLoss', self.TrMFeaLoss, Curr_epoch)
            
            self.writer.add_scalar('VaFullLoss', self.VaFullLoss, Curr_epoch)
            self.writer.add_scalar('VaWavfLoss', self.VaWavfLoss, Curr_epoch)
            self.writer.add_scalar('VaMFeaLoss', self.VaMFeaLoss, Curr_epoch)

            # checkpointing
            if self.CaPSQ:
                self.writer.add_scalar('VaPESQ', self.VaPESQ, Curr_epoch)
                if  self.VaPESQ > Best_PESQ:
                    self._save_Mdl(Curr_epoch)
                    Best_PESQ = self.VaPESQ
                    self.CaPSQ = False
                    modSavInd = True

            #pdb.set_trace()
            #self.scheduler['warm_up'].step()
            if Curr_epoch <= self.Args.warm_up_epochs:
                self.scheduler['warm_up'].step()
                if Curr_epoch == self.Args.warm_up_epochs:
                   self.DeepMdl = update_all_model(self.DeepMdl)
            else:
                self.scheduler['normal'].step()

            Curr_epoch += 1

        self.writer.flush()
        self.writer.close()

        return modSavInd

    def _train_epoch(self, Curr_epoch):

        self.TrFullLoss, self.TrWavfLoss, self.TrMFeaLoss = 0.0, 0.0, 0.0

        pbar = tqdm(self.dtaloader['tr'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Epoch {str(Curr_epoch)}/{str(self.Args.num_epochs)}')

        self.DeepMdl.train()
        for itrNum, FeaDict in enumerate(pbar):

            loss = self._step_tr(FeaDict)
            self.TrFullLoss += loss['fullLoss'].item()
            self.TrWavfLoss += loss['wavfLoss'].item()
            self.TrMFeaLoss += loss['moutLoss'].item()
            
            ShowFull = self.TrFullLoss/(itrNum + 1)
            ShowWavf = self.TrWavfLoss/(itrNum + 1)
            ShowMout = self.TrMFeaLoss/(itrNum + 1)

            pbar.set_postfix_str(f'(F_loss, S_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f})')

        pbar.close()
        self.TrFullLoss /= len(self.dtaloader['tr'])
        self.TrWavfLoss /= len(self.dtaloader['tr'])
        self.TrMFeaLoss /= len(self.dtaloader['tr'])

    def _step_tr(self, bthFea):
        
        mdlout = self.DeepMdl(bthFea['DegFeat']['inpfeat'].type(self.Dtype).to(self.device))
        tarFea = self._making_ground_truth(bthFea)

        loss = self._loss_calculate({'mdloupt':mdlout.to(self.device)}, tarFea)
            
        self._mdl_update(loss['fullLoss'])

        return loss

    def _val_epoch(self, Curr_epoch):
    
        self.VaFullLoss, self.VaWavfLoss, self.VaMFeaLoss = 0.0, 0.0, 0.0
        self.VaPESQ = 0.0
        self.CaPSQ  = False
        Num_Proc_Utt = 0
        
        pbar = tqdm(self.dtaloader['va'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Val')

        self.DeepMdl.eval()
        for itrNum, FeaDict in enumerate(pbar):
        
            with torch.no_grad():
                loss, enhwav = self._step_va(FeaDict)

            self.VaFullLoss += loss['fullLoss'].item()
            self.VaWavfLoss += loss['wavfLoss'].item()
            self.VaMFeaLoss += loss['moutLoss'].item()
            
            ShowFull = self.VaFullLoss/(itrNum + 1)
            ShowWavf = self.VaWavfLoss/(itrNum + 1)
            ShowMout = self.VaMFeaLoss/(itrNum + 1)
            
            batchsize = FeaDict['DegFeat']['inpfeat'].type(self.Dtype).detach().shape[0]
            Num_Proc_Utt += batchsize
            
            if Curr_epoch % self.CalPESQMod == 0:
                clnwav = np.array([FeaDict['TarWavf']['rcnwavf'][utIdx].type(self.Dtype).cpu() for utIdx in range(batchsize)])
                try:
                    self.VaPESQ += np.sum(self.cal_pesq(np.squeeze(clnwav,axis=1), np.squeeze(enhwav,axis=1), Fs = self.Args.sampling_rate, NCPU = self.Args.num_workers))
                except:
                    Num_Proc_Utt = Num_Proc_Utt - batchsize
                
                ShowPESQ = self.VaPESQ/Num_Proc_Utt
                pbar.set_postfix_str(f'(F_loss, S_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f})--PESQ={ShowPESQ:<5.3f};')
                self.CaPSQ = True
            else:
                pbar.set_postfix_str(f'(F_loss, S_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f});')

        if Curr_epoch % self.CalPESQMod == 0:
            self.VaFullLoss /= len(self.dtaloader['va'])
            self.VaWavfLoss /= len(self.dtaloader['va'])
            self.VaMFeaLoss /= len(self.dtaloader['va'])
            self.VaPESQ /= Num_Proc_Utt
            print('F_loss={:^7.3f};  PESQ={:^5.3f}'.format(self.VaFullLoss, self.VaPESQ))
            
        pbar.close()            

    def _step_va(self, bthFea):
    
        mdlout = self.DeepMdl(bthFea['DegFeat']['inpfeat'].type(self.Dtype).to(self.device))
        tarFea = self._making_ground_truth(bthFea)

        loss = self._loss_calculate({'mdloupt':mdlout.to(self.device)}, tarFea)
            
        batchsize = mdlout.detach().shape[0]
        
        return loss, np.array([mdlout[utIdx].type(self.Dtype).cpu() for utIdx in range(batchsize)])

    def test(self, InpMdl, Mdl_Cfg, DtaLoader):

        self.MdlPth = Mdl_Cfg['mdlpth']
        self.dtaloader = DtaLoader

        self._load_Mdl(InpMdl)
        self._test_epoch()

    def _test_epoch(self,):

        Ca_PSQ = True
        All_PESQ_Score = 0
        Num_Proc_Utt = 0

        pbar = tqdm(self.dtaloader, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Test')

        self.DeepMdl.eval()
        for itrNum, FeaDict in enumerate(pbar):

            batchsize = FeaDict['DegFeat']['inpfeat'].type(self.Dtype).detach().shape[0]
            enhwav = self._step_ts(FeaDict)
            if Ca_PSQ:
                clnwav = np.array([FeaDict['TarWavf']['rcnwavf'][utIdx].type(self.Dtype) for utIdx in range(batchsize)])
                try:
                    All_PESQ_Score += np.sum(self.cal_pesq(np.squeeze(clnwav,axis=1), np.squeeze(enhwav,axis=1), Fs = self.Args.sampling_rate, NCPU = self.Args.num_workers))
                except:
                    Num_Proc_Utt = Num_Proc_Utt - batchsize

                Num_Proc_Utt += batchsize
                pbar.set_postfix_str('PESQ={:^5.3f}'.format(All_PESQ_Score/Num_Proc_Utt))
            else:
                pbar.set_postfix_str('PESQ=4.5')

            for utIdx in range(batchsize):
                audiowrite(FeaDict['EhPthls'][utIdx], self.Args.sampling_rate, np.round(np.array(enhwav[utIdx] * (2**15))).squeeze().astype('int16'))

        pbar.close()

    def _step_ts(self, bthFea):
    
        mdlout = self.DeepMdl(bthFea['DegFeat']['inpfeat'].type(self.Dtype).to(self.device))

        batchsize = mdlout.detach().shape[0]
        
        return np.array([mdlout[utIdx].type(self.Dtype).detach().cpu() for utIdx in range(batchsize)])
        
    def _load_Mdl(self, InpMdl):
        print(f'Loading model from {self.MdlPth}')

        self.DeepMdl = InpMdl

        checkpoint = torch.load(self.MdlPth,map_location = self.device)
        self.DeepMdl.load_state_dict(checkpoint['model_state_dict'])
        self.DeepMdl.type(self.Dtype).to(self.device)

    def _save_Mdl(self,Curr_epoch):

        #save_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        print(f'Saving model to {self.MdlPth}')
        state_dict = {
            'epoch': Curr_epoch,
            'model_state_dict': self.DeepMdl.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.VaFullLoss,
            'pesq': self.VaPESQ
            }
        torch.save(state_dict, self.MdlPth)

    def _pre_setting_for_train(self, InpMdl, Mdl_Cfg, DtaLoader):

        self.DeepMdl = InpMdl.to(self.device)
        self.optimizer = Mdl_Cfg['optmizr']
        self.criterion = Mdl_Cfg['critern']
        self.scheduler = Mdl_Cfg['lr_schr']
        self.writer = Mdl_Cfg['writer']
        self.dtaloader = DtaLoader
        self.MdlPth = Mdl_Cfg['mdlpth']
        
        self.writer.add_graph(self.DeepMdl, torch.ones((1,1,150), dtype=self.Dtype).to(self.device))
        
    def _mdl_update(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _making_ground_truth(self, inpfeaDict):
    
        WavfFea = inpfeaDict['TarWavf']['rcnwavf'].type(self.Dtype).to(self.device)
    
        return {'modlRef': WavfFea}
    
    def _loss_calculate(self, degFea, target):
    
        return self.criterion(degFea, target)
            
    def cal_pesq(self, ref, deg, Fs = 16000, NCPU = 0):
        try:
            score = pesq_batch(Fs, ref, deg, 'wb', n_processor=NCPU)
            #score = pesq(Fs, ref.squeeze(0), deg.squeeze(0), 'nb')
        except:
            score = 0.

        return score

class AgentTrTserLSTM:
    def __init__(self, device, args):

        self.device = device
        self.Args = args

        self.Dtype = torch.float

    def train(self, InpMdl, Mdl_Cfg, DtaLoader):

        self._pre_setting_for_train(InpMdl, Mdl_Cfg, DtaLoader)

        Curr_epoch = self.Args.start_epochs
        Best_PESQ = 0.0

        self.CalPESQMod = self.Args.test_interval

        modSavInd = False

        while Curr_epoch <= self.Args.num_epochs:
            self._train_epoch(Curr_epoch)
            self._val_epoch(Curr_epoch)
            
            self.writer.add_scalar('TrFullLoss', self.TrFullLoss, Curr_epoch)
            self.writer.add_scalar('TrWavfLoss', self.TrWavfLoss, Curr_epoch)
            self.writer.add_scalar('TrMFeaLoss', self.TrMFeaLoss, Curr_epoch)
            
            self.writer.add_scalar('VaFullLoss', self.VaFullLoss, Curr_epoch)
            self.writer.add_scalar('VaWavfLoss', self.VaWavfLoss, Curr_epoch)
            self.writer.add_scalar('VaMFeaLoss', self.VaMFeaLoss, Curr_epoch)

            # checkpointing
            if self.CaPSQ:
                self.writer.add_scalar('VaPESQ', self.VaPESQ, Curr_epoch)
                if  self.VaPESQ > Best_PESQ:
                    self._save_Mdl(Curr_epoch)
                    Best_PESQ = self.VaPESQ
                    self.CaPSQ = False
                    modSavInd = True

            if Curr_epoch <= self.Args.warm_up_epochs:
                self.scheduler['warm_up'].step()
                if Curr_epoch == self.Args.warm_up_epochs:
                   self.DeepMdl = update_all_model(self.DeepMdl)
            else:
                self.scheduler['normal'].step()

            Curr_epoch += 1

        self.writer.flush()
        self.writer.close()

        return modSavInd

    def _train_epoch(self, Curr_epoch):

        self.TrFullLoss, self.TrWavfLoss, self.TrMFeaLoss = 0.0, 0.0, 0.0

        pbar = tqdm(self.dtaloader['tr'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Epoch {str(Curr_epoch)}/{str(self.Args.num_epochs)}')

        self.DeepMdl.train()
        for itrNum, FeaDict in enumerate(pbar):

            loss = self._step_tr(FeaDict)
            self.TrFullLoss += loss['fullLoss'].item()
            self.TrWavfLoss += loss['wavfLoss'].item()
            self.TrMFeaLoss += loss['moutLoss'].item()
            
            ShowFull = self.TrFullLoss/(itrNum + 1)
            ShowWavf = self.TrWavfLoss/(itrNum + 1)
            ShowMout = self.TrMFeaLoss/(itrNum + 1)

            pbar.set_postfix_str(f'(F_loss, W_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f})')

        pbar.close()
        self.TrFullLoss /= len(self.dtaloader['tr'])
        self.TrWavfLoss /= len(self.dtaloader['tr'])
        self.TrMFeaLoss /= len(self.dtaloader['tr'])

    def _step_tr(self, bthFea):
        
        mdlout = self.DeepMdl(bthFea['DegFeat']['inplps'].type(self.Dtype).to(self.device))

        EnhWav = self._spec_to_wav(mdlout, bthFea['DegFeat']['inppha'], bthFea['UtteLen'], bthFea['FFTInfo'])
        RefWav = self._spec_to_wav(bthFea['TarFeat']['tarlps'], bthFea['TarFeat']['tarpha'], bthFea['UtteLen'], bthFea['FFTInfo'])

        loss = self._loss_calculate({'wavfout': EnhWav, 'mdloupt': mdlout}, {'wavfRef': RefWav, 'modlRef': bthFea['TarFeat']['tarlps']})
            
        self._mdl_update(loss['fullLoss'])

        return loss

    def _val_epoch(self, Curr_epoch):
    
        self.VaFullLoss, self.VaWavfLoss, self.VaMFeaLoss = 0.0, 0.0, 0.0
        self.VaPESQ = 0.0
        self.CaPSQ  = False
        Num_Proc_Utt = 0
        
        pbar = tqdm(self.dtaloader['va'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Val')

        self.DeepMdl.eval()
        for itrNum, FeaDict in enumerate(pbar):
        
            with torch.no_grad():
                loss, enhwav, refwav = self._step_va(FeaDict)

            self.VaFullLoss += loss['fullLoss'].item()
            self.VaWavfLoss += loss['wavfLoss'].item()
            self.VaMFeaLoss += loss['moutLoss'].item()
            
            ShowFull = self.VaFullLoss/(itrNum + 1)
            ShowWavf = self.VaWavfLoss/(itrNum + 1)
            ShowMout = self.VaMFeaLoss/(itrNum + 1)
            
            batchsize = enhwav.type(self.Dtype).detach().shape[0]
            Num_Proc_Utt += batchsize
            
            if Curr_epoch % self.CalPESQMod == 0:
                clnwav = np.array([refwav[utIdx].type(self.Dtype).detach().cpu().numpy() for utIdx in range(batchsize)])
                ehwavf = np.array([enhwav[utIdx].type(self.Dtype).detach().cpu().numpy() for utIdx in range(batchsize)])
                try:
                    self.VaPESQ += np.sum(self.cal_pesq(clnwav, ehwavf, Fs = self.Args.sampling_rate, NCPU = self.Args.num_workers))
                except:
                    Num_Proc_Utt = Num_Proc_Utt - batchsize
                
                ShowPESQ = self.VaPESQ/Num_Proc_Utt
                pbar.set_postfix_str(f'(F_loss, W_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f})--PESQ={ShowPESQ:<5.3f};')
                self.CaPSQ = True
            else:
                pbar.set_postfix_str(f'(F_loss, W_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f});')

        if Curr_epoch % self.CalPESQMod == 0:
            self.VaFullLoss /= len(self.dtaloader['va'])
            self.VaWavfLoss /= len(self.dtaloader['va'])
            self.VaMFeaLoss /= len(self.dtaloader['va'])
            self.VaPESQ /= Num_Proc_Utt
            print('F_loss={:^7.3f};  PESQ={:^5.3f}'.format(self.VaFullLoss, self.VaPESQ))
            
        pbar.close()            

    def _step_va(self, bthFea):
    
        mdlout = self.DeepMdl(bthFea['DegFeat']['inplps'].type(self.Dtype).to(self.device))

        EnhWav = self._spec_to_wav(mdlout, bthFea['DegFeat']['inppha'], bthFea['UtteLen'], bthFea['FFTInfo'])
        RefWav = self._spec_to_wav(bthFea['TarFeat']['tarlps'], bthFea['TarFeat']['tarpha'], bthFea['UtteLen'], bthFea['FFTInfo'])

        loss = self._loss_calculate({'wavfout': EnhWav, 'mdloupt': mdlout}, {'wavfRef': RefWav, 'modlRef': bthFea['TarFeat']['tarlps']})
         
        return loss, EnhWav, RefWav

    def test(self, InpMdl, Mdl_Cfg, DtaLoader):

        self.MdlPth = Mdl_Cfg['mdlpth']
        self.dtaloader = DtaLoader

        self._load_Mdl(InpMdl)
        self._test_epoch()

    def _test_epoch(self,):

        Ca_PSQ = True
        All_PESQ_Score = 0
        Num_Proc_Utt = 0

        pbar = tqdm(self.dtaloader, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Test')

        self.DeepMdl.eval()
        for itrNum, FeaDict in enumerate(pbar):

            enhwav, refwav = self._step_ts(FeaDict)
            batchsize = enhwav.type(self.Dtype).detach().shape[0]

            if Ca_PSQ:
                clnwav = np.array([refwav[utIdx].type(self.Dtype).detach().cpu().numpy() for utIdx in range(batchsize)])
                ehwavf = np.array([enhwav[utIdx].type(self.Dtype).detach().cpu().numpy() for utIdx in range(batchsize)])
                try:
                    All_PESQ_Score += np.sum(self.cal_pesq(clnwav, ehwavf, Fs = self.Args.sampling_rate, NCPU = self.Args.num_workers))
                except:
                    Num_Proc_Utt = Num_Proc_Utt - batchsize

                Num_Proc_Utt += batchsize                
                pbar.set_postfix_str('PESQ={:^5.3f}'.format(All_PESQ_Score/Num_Proc_Utt))
            else:
                pbar.set_postfix_str('PESQ=4.5')

            for utIdx in range(batchsize):
                audiowrite(FeaDict['EhPthls'][utIdx], self.Args.sampling_rate, np.round(np.array(enhwav[utIdx].detach().cpu().numpy() * (2**15))).squeeze().astype('int16'))

        pbar.close()

    def _step_ts(self, bthFea):

        mdlout = self.DeepMdl(bthFea['DegFeat']['inplps'].type(self.Dtype).to(self.device))

        EnhWav = self._spec_to_wav(mdlout, bthFea['DegFeat']['inppha'], bthFea['UtteLen'], bthFea['FFTInfo'])
        RefWav = self._spec_to_wav(bthFea['TarFeat']['tarlps'], bthFea['TarFeat']['tarpha'], bthFea['UtteLen'], bthFea['FFTInfo'])

        return EnhWav, RefWav
        
    def _load_Mdl(self, InpMdl):
        print(f'Loading model from {self.MdlPth}')
        
        self.DeepMdl = InpMdl

        checkpoint = torch.load(self.MdlPth,map_location = self.device,weights_only=False)
        self.DeepMdl.load_state_dict(checkpoint['model_state_dict'])
        self.DeepMdl.type(self.Dtype).to(self.device)

    def _save_Mdl(self,Curr_epoch):

        #save_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        print(f'Saving model to {self.MdlPth}')
        state_dict = {
            'epoch': Curr_epoch,
            'model_state_dict': self.DeepMdl.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.VaFullLoss,
            'pesq': self.VaPESQ
            }
        torch.save(state_dict, self.MdlPth)

    def _pre_setting_for_train(self, InpMdl, Mdl_Cfg, DtaLoader):

        self.DeepMdl = InpMdl.to(self.device)
        self.optimizer = Mdl_Cfg['optmizr']
        self.criterion = Mdl_Cfg['critern']
        self.scheduler = Mdl_Cfg['lr_schr']
        self.writer = Mdl_Cfg['writer']
        self.dtaloader = DtaLoader
        self.MdlPth = Mdl_Cfg['mdlpth']
        
        InpStaticDim = int(self.Args.n_fft/2 + 1)
    
        self.writer.add_graph(self.DeepMdl, torch.ones((1,150,InpStaticDim), dtype=self.Dtype).to(self.device))
        
    def _mdl_update(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _loss_calculate(self, degFea, target):
    
        return self.criterion(degFea, target)
            
    def _spec_to_wav(self, InpLPS, InpPha, WavLen, FFTInfo):

        InpAmp = torch.permute(torch.sqrt(torch.pow(10, InpLPS)), (0,2,1))

        RecWav = spec2wav(InpAmp, InpPha, 
                          WinLen = FFTInfo['winsize'], 
                          NUM_FFT = FFTInfo['fftsize'], 
                          hop_length = FFTInfo['hopsize'], 
                          UttLen = WavLen, 
                          cmplxInp = False
                         )

        return RecWav

    def cal_pesq(self, ref, deg, Fs = 16000, NCPU = 0):
        try:
            score = pesq_batch(Fs, ref, deg, 'wb', n_processor=NCPU)
            #score = pesq(Fs, ref.squeeze(0), deg.squeeze(0), 'nb')
        except:
            score = 0.

        return score

class AgentTrTserDNN:
    def __init__(self, device, args):

        self.device = device
        self.Args = args

        self.Dtype = torch.float
        
    def train(self, InpMdl, Mdl_Cfg, DtaLoader):

        self._pre_setting_for_train(InpMdl, Mdl_Cfg, DtaLoader)

        Curr_epoch = self.Args.start_epochs
        Best_PESQ = 0.0

        self.CalPESQMod = self.Args.test_interval

        modSavInd = False

        while Curr_epoch <= self.Args.num_epochs:
            self._train_epoch(Curr_epoch)
            self._val_epoch(Curr_epoch)
            
            self.writer.add_scalar('TrFullLoss', self.TrFullLoss, Curr_epoch)
            self.writer.add_scalar('TrWavfLoss', self.TrWavfLoss, Curr_epoch)
            self.writer.add_scalar('TrMFeaLoss', self.TrMFeaLoss, Curr_epoch)
            
            self.writer.add_scalar('VaFullLoss', self.VaFullLoss, Curr_epoch)
            self.writer.add_scalar('VaWavfLoss', self.VaWavfLoss, Curr_epoch)
            self.writer.add_scalar('VaMFeaLoss', self.VaMFeaLoss, Curr_epoch)

            # checkpointing
            if self.CaPSQ:
                self.writer.add_scalar('VaPESQ', self.VaPESQ, Curr_epoch)
                if  self.VaPESQ > Best_PESQ:
                    self._save_Mdl(Curr_epoch)
                    Best_PESQ = self.VaPESQ
                    self.CaPSQ = False
                    modSavInd = True

            if Curr_epoch <= self.Args.warm_up_epochs:
                self.scheduler['warm_up'].step()
                if Curr_epoch == self.Args.warm_up_epochs:
                   self.DeepMdl = update_all_model(self.DeepMdl)
            else:
                self.scheduler['normal'].step()

            Curr_epoch += 1

        self.writer.flush()
        self.writer.close()

        return modSavInd

    def _train_epoch(self, Curr_epoch):

        self.TrFullLoss, self.TrWavfLoss, self.TrMFeaLoss = 0.0, 0.0, 0.0

        pbar = tqdm(self.dtaloader['tr'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Epoch {str(Curr_epoch)}/{str(self.Args.num_epochs)}')

        self.DeepMdl.train()
        for itrNum, FeaDict in enumerate(pbar):

            loss = self._step_tr(FeaDict)
            self.TrFullLoss += loss['fullLoss'].item()
            self.TrWavfLoss += loss['wavfLoss'].item()
            self.TrMFeaLoss += loss['moutLoss'].item()
            
            ShowFull = self.TrFullLoss/(itrNum + 1)
            ShowWavf = self.TrWavfLoss/(itrNum + 1)
            ShowMout = self.TrMFeaLoss/(itrNum + 1)

            pbar.set_postfix_str(f'(F_loss, W_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f})')

        pbar.close()
        self.TrFullLoss /= len(self.dtaloader['tr'])
        self.TrWavfLoss /= len(self.dtaloader['tr'])
        self.TrMFeaLoss /= len(self.dtaloader['tr'])

    def _step_tr(self, bthFea):
        
        mdlout = self.DeepMdl(bthFea['DegFeat']['inplps'].type(self.Dtype).to(self.device))

        EnhWav = self._spec_to_wav(mdlout, bthFea['DegFeat']['inppha'], bthFea['UtteLen'], bthFea['FFTInfo'])
        RefWav = self._spec_to_wav(bthFea['TarFeat']['tarlps'], bthFea['TarFeat']['tarpha'], bthFea['UtteLen'], bthFea['FFTInfo'])

        loss = self._loss_calculate({'wavfout': EnhWav, 'mdloupt': mdlout}, {'wavfRef': RefWav, 'modlRef': bthFea['TarFeat']['tarlps']})
            
        self._mdl_update(loss['fullLoss'])

        return loss

    def _val_epoch(self, Curr_epoch):
    
        self.VaFullLoss, self.VaWavfLoss, self.VaMFeaLoss = 0.0, 0.0, 0.0
        self.VaPESQ = 0.0
        self.CaPSQ  = False
        Num_Proc_Utt = 0
        
        pbar = tqdm(self.dtaloader['va'], bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Val')

        self.DeepMdl.eval()
        for itrNum, FeaDict in enumerate(pbar):
        
            with torch.no_grad():
                loss, enhwav, refwav = self._step_va(FeaDict)

            self.VaFullLoss += loss['fullLoss'].item()
            self.VaWavfLoss += loss['wavfLoss'].item()
            self.VaMFeaLoss += loss['moutLoss'].item()
            
            ShowFull = self.VaFullLoss/(itrNum + 1)
            ShowWavf = self.VaWavfLoss/(itrNum + 1)
            ShowMout = self.VaMFeaLoss/(itrNum + 1)
            
            batchsize = enhwav.type(self.Dtype).detach().shape[0]
            Num_Proc_Utt += batchsize
            
            if Curr_epoch % self.CalPESQMod == 0:
                clnwav = np.array([refwav[utIdx].type(self.Dtype).detach().cpu().numpy() for utIdx in range(batchsize)])
                ehwavf = np.array([enhwav[utIdx].type(self.Dtype).detach().cpu().numpy() for utIdx in range(batchsize)])
                try:
                    self.VaPESQ += np.sum(self.cal_pesq(clnwav, ehwavf, Fs = self.Args.sampling_rate, NCPU = self.Args.num_workers))
                except:
                    Num_Proc_Utt = Num_Proc_Utt - batchsize
                
                ShowPESQ = self.VaPESQ/Num_Proc_Utt
                pbar.set_postfix_str(f'(F_loss, W_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f})--PESQ={ShowPESQ:<5.3f};')
                self.CaPSQ = True
            else:
                pbar.set_postfix_str(f'(F_loss, W_loss, M_loss)=({ShowFull:>7.3f}, {ShowWavf:>7.3f}, {ShowMout:>7.3f});')

        if Curr_epoch % self.CalPESQMod == 0:
            self.VaFullLoss /= len(self.dtaloader['va'])
            self.VaWavfLoss /= len(self.dtaloader['va'])
            self.VaMFeaLoss /= len(self.dtaloader['va'])
            self.VaPESQ /= Num_Proc_Utt
            print('F_loss={:^7.3f};  PESQ={:^5.3f}'.format(self.VaFullLoss, self.VaPESQ))
            
        pbar.close()            

    def _step_va(self, bthFea):
    
        mdlout = self.DeepMdl(bthFea['DegFeat']['inplps'].type(self.Dtype).to(self.device))

        EnhWav = self._spec_to_wav(mdlout, bthFea['DegFeat']['inppha'], bthFea['UtteLen'], bthFea['FFTInfo'])
        RefWav = self._spec_to_wav(bthFea['TarFeat']['tarlps'], bthFea['TarFeat']['tarpha'], bthFea['UtteLen'], bthFea['FFTInfo'])

        loss = self._loss_calculate({'wavfout': EnhWav, 'mdloupt': mdlout}, {'wavfRef': RefWav, 'modlRef': bthFea['TarFeat']['tarlps']})
         
        return loss, EnhWav, RefWav

    def test(self, InpMdl, Mdl_Cfg, DtaLoader):

        self.MdlPth = Mdl_Cfg['mdlpth']
        self.dtaloader = DtaLoader

        self._load_Mdl(InpMdl)
        self._test_epoch()

    def _test_epoch(self,):

        Ca_PSQ = True
        All_PESQ_Score = 0
        Num_Proc_Utt = 0

        pbar = tqdm(self.dtaloader, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{rate_fmt}{postfix}|{elapsed}<{remaining}]')
        pbar.set_description(f'Test')

        self.DeepMdl.eval()
        for itrNum, FeaDict in enumerate(pbar):

            enhwav, refwav = self._step_ts(FeaDict)
            batchsize = enhwav.type(self.Dtype).detach().shape[0]

            if Ca_PSQ:
                clnwav = np.array([refwav[utIdx].type(self.Dtype).detach().cpu().numpy() for utIdx in range(batchsize)])
                ehwavf = np.array([enhwav[utIdx].type(self.Dtype).detach().cpu().numpy() for utIdx in range(batchsize)])
                try:
                    All_PESQ_Score += np.sum(self.cal_pesq(clnwav, ehwavf, Fs = self.Args.sampling_rate, NCPU = self.Args.num_workers))
                except:
                    Num_Proc_Utt = Num_Proc_Utt - batchsize

                Num_Proc_Utt += batchsize                
                pbar.set_postfix_str('PESQ={:^5.3f}'.format(All_PESQ_Score/Num_Proc_Utt))
            else:
                pbar.set_postfix_str('PESQ=4.5')

            for utIdx in range(batchsize):
                audiowrite(FeaDict['EhPthls'][utIdx], self.Args.sampling_rate, np.round(np.array(enhwav[utIdx].detach().cpu().numpy() * (2**15))).squeeze().astype('int16'))

        pbar.close()

    def _step_ts(self, bthFea):

        mdlout = self.DeepMdl(bthFea['DegFeat']['inplps'].type(self.Dtype).to(self.device))

        EnhWav = self._spec_to_wav(mdlout, bthFea['DegFeat']['inppha'], bthFea['UtteLen'], bthFea['FFTInfo'])
        RefWav = self._spec_to_wav(bthFea['TarFeat']['tarlps'], bthFea['TarFeat']['tarpha'], bthFea['UtteLen'], bthFea['FFTInfo'])

        return EnhWav, RefWav
        
    def _load_Mdl(self, InpMdl):
        print(f'Loading model from {self.MdlPth}')

        self.DeepMdl = InpMdl

        checkpoint = torch.load(self.MdlPth,map_location = self.device)
        self.DeepMdl.load_state_dict(checkpoint['model_state_dict'])
        self.DeepMdl.type(self.Dtype).to(self.device)

    def _save_Mdl(self,Curr_epoch):

        #save_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        print(f'Saving model to {self.MdlPth}')
        state_dict = {
            'epoch': Curr_epoch,
            'model_state_dict': self.DeepMdl.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.VaFullLoss,
            'pesq': self.VaPESQ
            }
        torch.save(state_dict, self.MdlPth)

    def _pre_setting_for_train(self, InpMdl, Mdl_Cfg, DtaLoader):

        self.DeepMdl = InpMdl.to(self.device)
        self.optimizer = Mdl_Cfg['optmizr']
        self.criterion = Mdl_Cfg['critern']
        self.scheduler = Mdl_Cfg['lr_schr']
        self.writer = Mdl_Cfg['writer']
        self.dtaloader = DtaLoader
        self.MdlPth = Mdl_Cfg['mdlpth']
        
        InpStaticDim = int(self.Args.n_fft/2 + 1)
    
        self.writer.add_graph(self.DeepMdl, torch.ones((1,150,InpStaticDim), dtype=self.Dtype).to(self.device))
        
    def _mdl_update(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _loss_calculate(self, degFea, target):
    
        return self.criterion(degFea, target)
            
    def _spec_to_wav(self, InpLPS, InpPha, WavLen, FFTInfo):

        InpAmp = torch.permute(torch.sqrt(torch.pow(10, InpLPS)), (0,2,1))

        RecWav = spec2wav(InpAmp, InpPha, 
                          WinLen = FFTInfo['winsize'], 
                          NUM_FFT = FFTInfo['fftsize'], 
                          hop_length = FFTInfo['hopsize'], 
                          UttLen = WavLen, 
                          cmplxInp = False
                         )

        return RecWav

    def cal_pesq(self, ref, deg, Fs = 16000, NCPU = 0):
        try:
            score = pesq_batch(Fs, ref, deg, 'wb', n_processor=NCPU)
            #score = pesq(Fs, ref.squeeze(0), deg.squeeze(0), 'nb')
        except:
            score = 0.

        return score

