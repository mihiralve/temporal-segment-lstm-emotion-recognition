import pickle,os
from PIL import Image
import scipy.io
from scipy import integrate
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# other util
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class mAPMeter():
    def __init__(self, data_len=26):
        self.count = 0
        self.data_len = data_len
        self.precision = []
        self.recall = []
        self.recall_sums = np.zeros(data_len)
        self.AP = np.zeros(data_len)
        self.mAP = 0


        self.precision.append(np.zeros(data_len))

    def add(self, output, target, threshold):

        diff = np.abs(output - target)

        diff = diff < threshold

        binary = []
        for x in diff:
            if x:
                binary.append(1)
            else:
                binary.append(0)

        self.track_precision(binary)
        self.track_recall(binary)

    def track_precision(self, binary):

        self.count += 1
        self.precision.append(np.zeros(len(binary)))

        self.precision[self.count] = (((self.count-1) * self.precision[self.count-1]) + binary)/self.count

    def track_recall(self, binary):

        self.recall.append(binary)
        self.recall_sums += binary

    def calc_precision_recall(self):
        self.recall = pd.DataFrame(self.recall)
        self.recall = self.recall.cumsum()/self.recall_sums
        self.recall = self.recall.fillna(0)

        self.precision = pd.DataFrame(self.precision[1:])

        # Smooth precision by setting each value to the max of it's rightmost values
        for i in range(self.data_len):
            self.precision[i] = self.precision.apply(lambda x: max(self.precision.iloc[x.name:, i]), axis=1)

    def calc_mean_ap(self):
        for i in range(self.data_len):
            self.AP[i] = scipy.integrate.trapz(self.precision[i], self.recall[i])

        self.mAP = np.average(self.AP)

    def results(self):
        self.calc_precision_recall()
        self.calc_mean_ap()

        return self.mAP


def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

def record_info(info,filename,mode):

    if mode =='train':

        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'LR {lr}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], lr=info['lr']))      
        print(result)

        df = pd.DataFrame.from_dict(info)
        # column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5','lr']
        column_names = ['Epoch','Batch Time','Data Time','Loss','lr']
        
    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Loss {loss} '#.format( batch_time=info['Batch Time'],
               'Prec@1 {top1} '
               'Prec@5 {top5} \n'
               'Mean Average Precision {mean_average_precision} \n'.format( batch_time=info['Batch Time'],
                loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'], mean_average_precision=info['Mean Average Precision']))
        print(result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Loss','Prec@1','Prec@5', 'Mean Average Precision']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)   


