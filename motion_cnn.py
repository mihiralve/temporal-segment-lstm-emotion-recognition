import numpy as np
import pickle
from PIL import Image
import time
import tqdm
import shutil
from random import randint
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from network import *
from network import LSTM
import dataloader


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 motion stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-6, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    global arg
    arg = parser.parse_args()
    print(arg)

    #Prepare DataLoader
    data_loader = dataloader.Motion_DataLoader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=0,
                        path='../bold_data/BOLD_ijcv/BOLD_public/optic_flow/',
                        ucf_list = '../bold_data/BOLD_ijcv/BOLD_public/annotations/',
                        ucf_split = '04',
                        in_channel=10,
                        )
    
    train_loader,test_loader, test_video = data_loader.run()
    #Model 
    model = Motion_CNN(
                        # Data Loader
                        train_loader=train_loader,
                        test_loader=test_loader,
                        # Utility
                        start_epoch=arg.start_epoch,
                        resume=arg.resume,
                        evaluate=arg.evaluate,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        channel = 10,
                        test_video=test_video
                        )
    #Training
    model.run()

class Motion_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, channel,test_video):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.channel=channel
        self.test_video=test_video
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=10)
        self.lstm = LSTM(26, 26, self.batch_size, output_dim=26, num_layers=2)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.lstm = nn.DataParallel(self.lstm)
        self.model.to(self.device)
        self.lstm.to(self.device)
        #print self.model
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
            self.epoch=0
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
            with open('record/motion/motion_video_preds.pickle','wb') as f:
                pickle.dump(self.dic_video_level_preds,f)
            f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/motion/checkpoint.pth.tar','record/motion/model_best.pth.tar')

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
            output = Variable(torch.zeros(len(data_dict['opf0']),26).float()).to(self.device)
            for i in range(len(data_dict)):
                key = 'opf'+str(i)
                data = data_dict[key]
                input_var = Variable(data).to(self.device)
                output = self.model(input_var)

                output = self.lstm(output).view(-1, self.batch_size, 26)

            loss = self.criterion(output, target_var)

            # record loss
            losses.update(loss.data, data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            self.lstm_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lstm_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg.item(),5)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/motion/opf_train.csv','train')

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
                output = Variable(torch.zeros(len(data_dict['opf0']),26).float()).to(self.device)
                for i in range(len(data_dict)):
                    key = 'opf'+str(i)
                    data = data_dict[key]
                    input_var = Variable(data).to(self.device)
                    output = self.model(input_var)

                    output = self.lstm(output).view(-1, self.batch_size, 26)

                loss = self.criterion(output, target_var)

                # record loss
                losses.update(loss.data, data.size(0))

                output = nn.Softmax(dim=1)(output[0])
                for i in range(self.batch_size):
                    ap.add(output[i].cpu().numpy(), target_var[i].cpu().numpy(), 0.1)
                    self.dic_video_level_preds[video_name[i]] = output[i].cpu().numpy()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        #Frame to video level accuracy
        mean_average_precision = ap.results()

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[losses.avg.item()],
                'Prec@1':[0],
                'Prec@5':[0],
                'Mean Average Precision': mean_average_precision}
        record_info(info, 'record/motion/opf_test.csv','test')
        return mean_average_precision, losses.avg.item()

    def frame2_video_level_accuracy(self):
     
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),26))
        video_level_labels = np.zeros((len(self.dic_video_level_preds),26))
        top1 = 0
        top5 = 0
        ii=0
        for key in sorted(self.dic_video_level_preds.keys()):
            name = key.split('-',1)[0]

            preds = self.dic_video_level_preds[name]
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