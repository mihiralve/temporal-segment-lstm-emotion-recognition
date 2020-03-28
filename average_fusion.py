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

    plt.figure()
    plt.ylim(0, 1)
    plt.bar(cols, ap.AP)
    plt.xticks(rotation=90)
    plt.title("Average")
    plt.tight_layout()
    plt.savefig("record/graphs/categorical_ap.png")

    plt.figure()
    plt.ylim(0, 1)
    plt.bar(cols, ap_r.AP)
    plt.xticks(rotation=90)
    plt.title("Spatial Stream")
    plt.tight_layout()
    plt.savefig("record/graphs/categorical_ap_r.png")

    plt.figure()
    plt.ylim(0, 1)
    plt.bar(cols, ap_o.AP)
    plt.xticks(rotation=90)
    plt.title("Temporal Stream")
    plt.tight_layout()
    plt.savefig("record/graphs/categorical_ap_o.png")
