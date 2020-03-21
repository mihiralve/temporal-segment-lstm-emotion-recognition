import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader.split_train_test_video_2 import *      #Needed when running from motion_cnn
# from split_train_test_video_2 import *

class motion_dataset(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        #Generate a 16 Frame clip
        self.keys=list(dic.keys())
        self.values=list(dic.values())
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=320
        self.img_cols=320

    def stackopf(self):
        name = self.video
        # u = self.root_dir+ 'u/' + name
        # v = self.root_dir+ 'v/'+ name
        
        flow_dir = self.root_dir + name + "/inference/run.epoch-0-flow-vis/"
        
        flow = torch.FloatTensor(self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)


        for j in range(self.in_channel):
            idx = i + j
            idx = str(idx)
            frame_idx = idx.zfill(6) + "-vis.png"

            try:
                img = (Image.open(flow_dir + frame_idx))
                img = self.transform(img)
            except:
                idx = i + j - 1
                idx = str(idx)
                frame_idx = idx.zfill(6) + "-vis.png"
                img = (Image.open(flow_dir + frame_idx)) # In the event that one frame is corrupted stack the previous frame twice
                img = self.transform(img)

            flow[j-1,:,:] = img
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':
            self.video, nb_clips = list(self.keys)[idx].split(':')
            clips = []
            segments = 1
            for i in range(segments):
                clips.append(random.randint(int((int(nb_clips)/segments) * i), int((int(nb_clips)/segments) * (i+1) -1)))
        elif self.mode == 'val':
            self.video,self.clips_idx = list(self.keys)[idx].split(':')
        else:
            raise ValueError('There are only train and val mode')

        label = list(self.values)[idx]
        label = np.array(label).astype(np.float64)

        if self.mode == 'train':
            data = {}
            for i in range(len(clips)):
                key = 'opf'+str(i)
                self.clips_idx = clips[i]
                data[key] = self.stackopf()
            sample = (data,label)
        elif self.mode == 'val':
            data = self.stackopf()
            sample = (self.video,data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample


class Motion_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel,  path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count={}
        self.in_channel = in_channel
        self.data_path=path
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()
        
    def load_frame_count(self):
        # with open("../../bold_data/BOLD_ijcv/BOLD_public/annotations/framecount.pkl", "rb") as f:
        with open("../bold_data/BOLD_ijcv/BOLD_public/annotations/framecount.pkl", "rb") as f:
            self.frame_count = pickle.load(f)

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video
            
    def val_sample19(self):
        self.dic_test_idx = {}
        #print len(self.test_video)
        for video in self.test_video:

            sampling_interval = int((self.frame_count[video]-10+1)/19)
            for index in range(19):
                clip_idx = index*sampling_interval
                key = video + ':' + str(clip_idx+1)
                self.dic_test_idx[key] = self.test_video[video]
             
    def get_training_dic(self):
        self.dic_video_train={}
        for video in self.train_video:
            
            nb_clips = self.frame_count[video]-10+1
            key = video +':' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video] 
                            
    def train(self):
        training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train',
            transform = transforms.Compose([
            transforms.Scale([320,320]),
            transforms.Grayscale(),
            transforms.ToTensor(),
            ]))
        print('==> Training data :',len(training_set),' videos',training_set[0][0]['opf0'].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True)

        return train_loader

    def val(self):
        validation_set = motion_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Scale([320,320]),
            transforms.Grayscale(),
            transforms.ToTensor(),
            ]))
        print('==> Validation data :',len(validation_set),' frames',validation_set[1][1].size())
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True)

        return val_loader

if __name__ == '__main__':
    data_loader =Motion_DataLoader(BATCH_SIZE=1,num_workers=1,in_channel=10,
                                        path='../../bold_data/BOLD_ijcv/BOLD_public/optic_flow/',
                                        ucf_list = '../../bold_data/BOLD_ijcv/BOLD_public/annotations/',
                                        ucf_split = '04')
    train_loader,val_loader,test_video = data_loader.run()
    #print train_loader,val_loader