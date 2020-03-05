# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:45:12 2020

@author: Mihir
"""

import os, pickle


class UCF101_splitter():
    def __init__(self, path, split):
        self.path = path
        self.split = split
    
    def split_video(self):
        return self.get_dict("train"), self.get_dict("test")
    
    def get_dict(self, subset):
        self.vid_dict = {} 
        train_file = self.path + subset + "list" + self.split + ".txt"
        test_file = self.path + subset + "list" + self.split + ".txt"
        
        with open(train_file) as f:
            content = f.readlines()
        f.close()
        
        content = [x.strip('\r\n') for x in content]
        content = [x.split(" ") for x in content]
        
        for x in content:
            self.vid_dict[x[0]] = x[1:]
        
        return self.vid_dict

        
if __name__ == "__main__":
    path = '../../bold_data/BOLD_ijcv/BOLD_public/annotations/'
    split = '04'
    splitter = UCF101_splitter(path=path,split=split)
    train_video,test_video = splitter.split_video()
    print(len(train_video),len(test_video))