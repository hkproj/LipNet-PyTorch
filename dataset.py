# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance

def normalize_images(images):
    # Compute mean and standard deviation per channel
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    
    # Normalize images per channel
    normalized_images = (images - mean) / std
    
    return normalized_images

class MyDataset(Dataset):
    normalize = True
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase, percentage=1.0):
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        
        with open(file_list, 'r') as f:
            self.videos = [os.path.join(video_path, line.strip()) for line in f.readlines()]

        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)            
            self.data.append((vid, items[-4], items[-1]))
        
                
    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        vid = self._load_vid(vid) # (T, H, W, C) = (VID_LEN, 64, 128, 3)
        if MyDataset.normalize:
            vid = normalize_images(vid)
        anno = self._load_anno(os.path.join(self.anno_path, spk, 'align', name + '.align')) # (ANNO_LEN,)

        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)
          
        vid = ColorNormalize(vid)                   
        
        vid_len = vid.shape[0]
        anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        return {
            'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), # (C, T, H, W)
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len,
            'vid_spk': spk[:],
            'vid_name': name[:],
        }
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
        array = np.stack(array, axis=0).astype(np.float32)
        return array # (T, H, W, C)
    
    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        return MyDataset.txt2arr(' '.join(txt).upper(), 1)
    
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def txt2arr(txt, start):
        """
        Converts an array of characters into an array of indices mapping to the vocabulary of letters.
        The space is mapped to 1, not zero (and subsequent letters are also INDEX + 1)
        """
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)
        
    @staticmethod
    def arr2txt(arr, start):
        """
        Converts an array of integers into an array of characters
        """
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        """
        Converts an array of integers mapping to characters into a string, by ignoring duplicates
        """
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])                
            pre = n
        return ''.join(txt).strip()
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer
