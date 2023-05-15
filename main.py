import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from dataset import MyDataset
import numpy as np
import time
from model import LipNet
import torch.optim as optim
import re
import json
from torch.utils.tensorboard import SummaryWriter


if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu    
    writer = SummaryWriter()

def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()  

def ctc_decode(y):
    """
    Takes an input tensor of shape (BATCH, SEQUENCE_LENGTH, NUM_CLASSES)
    Then performs an "argmax" on the last dimension to select the class with highest probability
    Then, for each item in the batch (y.size(0)), it gets the text 
    """
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]
    
def test(model, net):

    with torch.no_grad():
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')
            
        print('num_test_data:{}'.format(len(dataset.data)))  
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):            
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            vid_spk = input.get('vid_spk')
            vid_name = input.get('vid_name')
            y = net(vid)
            
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
            cer.extend(MyDataset.cer(pred_txt, truth_txt))              
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0
                
                print(''.join(161*'-'))                
                print('{:<80}|{:>80}'.format('predict', 'truth'))
                print(''.join(161*'-'))                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<80}|{:>80}'.format(predict, truth))                
                print(''.join(161 *'-'))
                print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
                print(''.join(161 *'-'))
                
        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())
    
def train(model, net):
    
    dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
        
    loader = dataset2dataloader(dataset) 
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)
                
    print('num_train_data:{}'.format(len(dataset.data)))    
    crit = nn.CTCLoss(zero_infinity=True)
    tic = time.time()
    
    train_wer = []
    for epoch in range(opt.max_epoch):
        for (i_iter, input) in enumerate(loader):
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            vid_spk = input.get('vid_spk')
            vid_name = input.get('vid_name')
            
            optimizer.zero_grad()
            y = net(vid)
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()
            
            tot_iter = i_iter + epoch*len(loader)
            
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            
            if(tot_iter > 0 and tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0
                
                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)     
                writer.flush()         
                print(''.join(161*'-'))                
                print('{:<80}|{:>80}'.format('predict', 'truth'))                
                print(''.join(161*'-'))
                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<80}|{:>80}'.format(predict, truth))
                print(''.join(161*'-'))                
                print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                print(''.join(161*'-'))
                
            if(tot_iter > 0 and tot_iter % opt.test_step == 0):                
                (loss, wer, cer) = test(model, net)
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                writer.add_scalar('val loss', loss, tot_iter)                    
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                writer.flush()
                savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()
                
if(__name__ == '__main__'):
    print("Loading options...")

    isItaLIP = len(sys.argv) > 1 and sys.argv[1] == 'italip'

    if isItaLIP:
        print(f'Training ItaLIP')
        MyDataset.normalize = False
        opt.train_list = ''
        opt.val_list = ''
        opt.data_type = 'unseen'
        opt.video_path = 'italip_lip/'
        opt.train_list = f'italip_data/{opt.data_type}_train.txt'
        opt.val_list = f'italip_data/{opt.data_type}_val.txt'
        opt.anno_path = 'italip_GRID_align_txt'
        opt.vocab_file = f'italip_data/vocabulary.txt'
    else:
        MyDataset.normalize = False
    print(f'Normalizing images: {MyDataset.normalize}')

    if isItaLIP:
        assert opt.vocab_file is not None, 'Vocabulary file must be specified for ItaLIP'
        # Load the vocabulary and update the letters 
        with open(opt.vocab_file, 'r') as f:
            # Read all the non-whitespace lines
            chars = set([line.strip() for line in f.read().splitlines() if line.strip() != ''])
            # Make sure the space character is present
            chars.add(' ')
            if '\n' in chars:
                chars.remove('\n')
            if '\r' in chars:
                chars.remove('\r')
            
            # Verify that all chars are strings of length 1
            for c in chars:
                assert len(c) == 1, 'Invalid character in vocabulary: \'{}\''.format(c)

            # Sort all the characters
            chars = sorted(list(chars))
            MyDataset.letters = chars
    print(f'Loaded vocabulary with {len(MyDataset.letters)} characters')

    model = LipNet(vocab_size=len(MyDataset.letters))
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)
        
