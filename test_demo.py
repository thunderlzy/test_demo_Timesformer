from pathlib import Path
import torch
from timesformer.models.vit import TimeSformer
import torch.nn as nn
import numpy as np
import os
import math
import cv2
from torchvision import transforms
import time 

def get_name(txt):
        with open ('./name_done.txt','r',encoding='utf-8') as f:
            n=0
            for x in f.readlines ():
                
                if n==txt:
                    return x
                else:
                    n+=1

def get_tensor_from_video(video_path):
    """
    :param video_path: 视频文件地址
    :return: pytorch tensor
    """
    if not os.access(video_path, os.F_OK):
        print('测试文件不存在')
        return

   
    cap = cv2.VideoCapture(video_path)

    frames_list = []
    while(cap.isOpened()):
        ret,frame = cap.read()

        if not ret:
            break
        else:
            # 注意，opencv默认读取的为BGR通道组成模式，需要转换为RGB通道模式 转换通道为C*H*W并进行归一化（0，1）
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
            #颜色标准化          
            frame=transform(frame)
            frames_list.append(frame)
    
    cap.release()
    frames_list1=[]
    n=len(frames_list)
    t=n//8
    for i in range(1,n):
        if i%t==0:
            frames_list1.append(frames_list[i])
        
    # 转换成tensor
    result_frames = torch.stack(frames_list1)
    # 注意：此时result_frames组成的维度为[视频帧数量，宽，高，通道数]
    return result_frames

def pre_process(images,min_size,max_size):
    size = int(round(np.random.uniform(min_size, max_size)))
    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images
    new_height = size
    new_width  = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))

    else:
        new_width = int(math.floor((float(width) / height) * size))

    return torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,)

def uniform_crop(images, size, spatial_idx):
    assert spatial_idx in [0, 1, 2]
    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]
    return cropped

    
if __name__=='__main__':
    start=time.time()
    model_file = '/Users/leizeyu/Desktop/TimeSformer/checkpoints/TimeSformer_divST_8x32_224_K600.pyth'
    video_path='./data/test11.mp4'
    model = TimeSformer(img_size=224, num_classes=600, num_frames=8, attention_type='divided_space_time',  pretrained_model=str(model_file))
    model.eval()
    video=get_tensor_from_video(video_path)
    video=video.permute(1,0,2,3)
    video=pre_process(video,256,320)#缩放
    video0=uniform_crop(video,224,0)#裁剪 012 左中右 上中下
    video1=uniform_crop(video,224,1)#裁剪 012 左中右 上中下
    video2=uniform_crop(video,224,2)#裁剪 012 左中右 上中下
    video0=video0.unsqueeze(0) 
    video1=video1.unsqueeze(0)
    video2=video2.unsqueeze(0)
    pred0= model(video0) # (600)
    pred1= model(video1) # (600)
    pred2= model(video2) # (600)
    pred=pred0+pred1+pred2
    pred=pred.squeeze()
    txt=int(torch.argmax(pred,0))
    name=get_name(txt)
    end=time.time()
    vtime=end-start
    print(name,vtime)

    