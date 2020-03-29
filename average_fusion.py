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

    ap = mAPMeter()
    ap_r = mAPMeter()
    ap_o = mAPMeter()
    video_level_preds = np.zeros((len(rgb.keys()),26))
    video_level_labels = np.zeros((len(rgb.keys()),26))
    correct=0
    top1=0
    top5=0
    ii=0
    for name in test_video:
        try:
            r = rgb[name]
            o = opf[name]

            #label = int(test_video[name])-1
            label = list(test_video[name])
            label = np.array(label).astype(np.float64)

            preds = (r+o)/sum((r+o))

            video_level_preds[ii,:] = preds
            video_level_labels[ii,:] = label
            ii+=1


            ap.add(preds, label, 0.05)
            ap_r.add(r, label, 0.05)
            ap_o.add(o, label, 0.05)
        except:
            pass

    print("rgb:" + str(ap_r.results()))
    print("opf:" + str(ap_o.results()))
    print("avg:" + str(ap.results()))

    cols = ['Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
       'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
       'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue',
       'Embarrasment', 'Yearning', 'Disapproval', 'Aversion', 'Annoyance',
       'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain',
       'Suffering']

    if not os.path.exists("record/graphs/"):
        os.mkdir("record/graphs/")

    data = [ap.AP, ap_r.AP, ap_o.AP]

    C = len(cols)
    ind = np.arange(C)
    ind = np.array([5*i for i in ind])
    width = 1
    indshift = np.linspace(-width/2,width/2,len(data))

    fig, ax = plt.subplots()
    fig.set_size_inches(15,9)
    rects1 = ax.bar(ind - width, ap.AP, width, color='#2c9c2d', align="center")
    rects2 = ax.bar(ind, ap_r.AP, width, color='#eb4034', align="center")
    rects3 = ax.bar(ind + width, ap_o.AP, width, color='#3486eb', align="center")
    ax.set_ylim(0, 1)
    ax.legend(('Average Fusion', 'Spatial Stream', 'Temporal Stream'), loc="upper right", bbox_to_anchor=(1, 1.12))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(cols, rotation=90)
    ax.set_ylabel("Average Precision")

    fig.tight_layout()
    plt.savefig("record/graphs/emotion_ap.png")
