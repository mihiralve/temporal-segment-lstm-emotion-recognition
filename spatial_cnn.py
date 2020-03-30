import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from utils import *
from network import *
from network import LSTM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=25, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=1e-6, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    global arg
    arg = parser.parse_args()
    print(arg)

    #Prepare DataLoader
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=0,
                        path='../bold_data/BOLD_ijcv/BOLD_public/frames/',
                        ucf_list = '../bold_data/BOLD_ijcv/BOLD_public/annotations/',
                        ucf_split = '04'
                        )
    
    train_loader, test_loader, test_video = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video
    )
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        print('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=3)
        self.lstm = LSTM(26, 26, self.batch_size, output_dim=26, num_layers=2)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.lstm = nn.DataParallel(self.lstm)
        self.model.to(self.device)
        self.lstm.to(self.device)
        #Loss function and optimizer
        self.criterion = self.custom_cross_entropy_loss #nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.lstm_optimizer = torch.optim.SGD(self.lstm.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5,verbose=True)


    def custom_cross_entropy_loss(self, output, target):
        out_sm = nn.Softmax(dim=1)(output)
        loss = -(target * torch.log(out_sm))
        return loss.sum()

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                pickle.dump(self.dic_video_level_preds,f)
            f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict,label) in enumerate(progress):


            # measure data loading time
            data_time.update(time.time() - end)

            label = label.to(self.device)
            target_var = Variable(label).to(self.device)

            # compute output
            output = Variable(torch.zeros(len(data_dict['img0']),26).float()).to(self.device)
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).to(self.device)
                output += self.model(input_var)

                #output = self.lstm(output).view(-1, self.batch_size, 26)

            loss = self.criterion(output/len(data_dict), target_var)

            lambda_val = torch.tensor(1.).to(self.device)
            l2_reg = torch.tensor(0.).to(self.device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += lambda_val * l2_reg

            # record loss
            losses.update(loss.data, data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            #self.lstm_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.lstm_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg.item(),5)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        ap = mAPMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (video_name, data_dict,label) in enumerate(progress):
                label = label.to(self.device)
                target_var = Variable(label).to(self.device)

                # compute output
                output = Variable(torch.zeros(len(data_dict['img0']),26).float()).to(self.device)
                for i in range(len(data_dict)):
                    key = 'img'+str(i)
                    data = data_dict[key]
                    input_var = Variable(data).to(self.device)
                    output += self.model(input_var)
                output = output/len(data_dict)
                    #output = self.lstm(output).view(-1, self.batch_size, 26)

                loss = self.criterion(output/len(data_dict), target_var)

                lambda_val = torch.tensor(1.).to(self.device)
                l2_reg = torch.tensor(0.).to(self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += lambda_val * l2_reg

                # record loss
                losses.update(loss.data, data.size(0))

                output = nn.Softmax(dim=1)(output)
                for i in range(self.batch_size):
                    pred = output[i].cpu().numpy()
                    ap.add(pred, target_var[i].cpu().numpy(), 0.1)
                    self.dic_video_level_preds[video_name[i]] = pred

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        mean_average_precision = ap.results()

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[losses.avg.item()],
                'Prec@1':[0],
                'Prec@5':[0],
                'Mean Average Precision': mean_average_precision}
        record_info(info, 'record/spatial/rgb_test.csv','test')
        return mean_average_precision, losses.avg.item()

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),26))
        video_level_labels = np.zeros((len(self.dic_video_level_preds),26))
        ii=0
        
        top1 = 0
        top5 = 0
        
        for name in sorted(self.dic_video_level_preds.keys()):
        
            preds = self.dic_video_level_preds[name]
            preds = preds/preds.sum()
            label = list(self.test_video[name])
            label = np.array(label).astype(np.float64)
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii,:] = label
            ii+=1         
            
            
            if np.average(np.abs(preds-label)) < 0.1:
                top1 += 1
                
            if np.average(np.abs(preds-label)) < 0.5:
                top5 += 1
            

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).float()
        video_level_preds = torch.from_numpy(video_level_preds).float()
            
        loss = self.criterion(Variable(video_level_preds).to(self.device), Variable(video_level_labels).to(self.device))

        return top1,top5,loss.data.cpu().numpy()


if __name__=='__main__':
    main()