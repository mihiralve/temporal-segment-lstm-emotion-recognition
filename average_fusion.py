from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader

if __name__ == '__main__':

    rgb_preds='record/spatial/spatial_video_preds.pickle'
    opf_preds = 'record/motion/motion_video_preds.pickle'

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=1,
                        num_workers=0,
                        path='../bold_data/BOLD_ijcv/BOLD_public/frames/',
                        ucf_list = '../bold_data/BOLD_ijcv/BOLD_public/annotations/',
                        ucf_split = '04'
                        )
    train_loader,val_loader,test_video = data_loader.run()

    video_level_preds = np.zeros((len(rgb.keys()),26))
    video_level_labels = np.zeros((len(rgb.keys()),26))
    correct=0
    top1=0
    top5=0
    ii=0
    for name in sorted(rgb.keys()):
        r = rgb[name]
        o = opf[name]

        #label = int(test_video[name])-1
        label = list(test_video[name])
        label = np.array(label).astype(np.float64)

        preds = (r+o)/sum((r+o))

        video_level_preds[ii,:] = preds
        video_level_labels[ii,:] = label
        ii+=1

        if np.average(np.abs(preds-label)) < 0.01:
                top1 += 1
        if np.average(np.abs(preds-label)) < 0.05:
                top5 += 1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()


    print(top1,top5)
