#!/usr/bin/env python
# coding: utf-8

# In[53]:


import logging
import os
import sys
import tempfile
from glob import glob
from PIL import Image
from glob import glob
import torch
import numpy as np
import pandas as pd

import monai
from monai.metrics import DiceMetric


# In[77]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, 
        default='/data/jiesi/COVID19/MONAI_model/2D/Unet/runs/May01_15-26-15_hitlab-SYS-7048GR-TR/output')
parser.add_argument('--task', type=str, 
        default='Task075_COVID19Challenge')
config = parser.parse_args()
print(config)


# In[78]:


from torchmetrics.functional import precision_recall
class get_precision_recall(object):
    def __init__(self):
        """
        nothing to do
        """
        self.flag = True
        self.reset()  
        
    def __call__(self, preds,target,average = None,num_classes=2):
        """
        This method should take torch tensor
        """
        if isinstance(preds, (list, tuple)):
            preds = torch.stack(preds)
        if isinstance(target, (list, tuple)):
            target = torch.stack(target)
        
        PR = precision_recall(preds, target,average = None,num_classes=2,mdmc_average = 'samplewise')
#         print(PR)
        PR_class1 = [float(PR[0][1].cpu().detach().numpy()),float(PR[1][1].cpu().detach().numpy())] # precision, recall
        self.result_list.append(PR_class1)
        return PR_class1 
    
    def reset(self):
        self.result_list = []
        self.mean_result = None
        
    def aggregate(self):
        self.result_list = np.array(self.result_list)
        self.mean_result = np.mean(self.result_list,axis = 0)
        return self.mean_result
def mask_(img3c_,mask_tmp):
    
    img3c = img3c_.copy().astype('float')
    mask_tmp = mask_tmp.astype('float')
    img3c[:,:,0] = img3c[:,:,0] + mask_tmp*40.0
    img3c[:,:,1] = img3c[:,:,1] - mask_tmp*20
    img3c[:,:,2] = img3c[:,:,2] - mask_tmp*20

    img3c[img3c>255] = 255
    img3c[img3c<0] = 0
    img3c = img3c.astype('uint8')
    return img3c


# In[79]:


if 'Task075_COVID19Challenge' in config.task:
    Image_path = '/data/jiesi/COVID19/nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task075_COVID19Challenge/data2D/val'
    GT = '/data/jiesi/COVID19/nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task075_COVID19Challenge/data2D/val_GT'
    print('Task',config.task)
elif 'Task074_COVID19sz' in config.task:
    Image_path = '/data/jiesi/COVID19/nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task074_COVID19sz/data2D/val'
    GT = '/data/jiesi/COVID19/nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task074_COVID19sz/data2D/val_GT'
    print('Task',config.task)
else:
    print('Task not implemented!')
prediction = config.output_path

# In[80]:


print('Length of GT',len(os.listdir(GT)))
print('Length of prediction',len(os.listdir(prediction)))


# In[57]:


# split the patient
# [[patient 1],[patient 2],...]
patients_ = sorted(set(['_'.join(i.split('_')[:2]) for i in sorted(os.listdir(GT))]))
patients_ordered = []
for i in patients_:
    temp = [j.split('/')[-1] for j in sorted(glob(os.path.join(GT,i+'_***.png')))]
    patients_ordered.append(temp)


# In[58]:


# import torchvision.transforms as transforms
# Transform = transforms.Resize((512, 512))
from PIL import Image
import PIL
from matplotlib import cm

def resize_array(pred_map):
    im = Image.fromarray(pred_map)
    im = im.resize((512,512),resample = PIL.Image.NEAREST)
    return np.array(im)


# In[86]:


# load patient and combined them
dice_dict = []
for thres in [50,75,100,125,150,175,200]:
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice2_list = []

    for patient_png in patients_ordered:
        patient_GT = []
        patient_pred = []
        for img_name in patient_png:
            GT_path = os.path.join(GT,img_name)
            patient_GT.append(np.asarray(Image.open(GT_path)))

            pred_path = os.path.join(prediction,img_name)
            pred_map = np.asarray(Image.open(pred_path)).T

            patient_pred.append(resize_array(pred_map))
        patient_GT = np.array([patient_GT])
        patient_pred = np.array([patient_pred])

        patient_pred[patient_pred<thres]=0
        patient_pred[patient_pred>=thres]=1
        dice_temp = dice_metric(y_pred=[torch.tensor(patient_pred)], y=[torch.tensor(patient_GT)])
        dice2_list.append(dice_temp)
#         print(patient_png[0],dice_temp)

    # compute dice index
    dice_tmp = dice_metric.aggregate().item()
    dice_dict.append([thres,dice_tmp])
    print("evaluation metric:", dice_tmp)
    print('Thres:',thres)
    # reset the status
    dice_metric.reset()


# In[90]:


# find the threshold with the largest metric
dice_dict = np.array(dice_dict)
dice_dict = dice_dict[dice_dict[:,1].argsort()]

# print(dice_dict)
# In[63]:


# load patient and combined them
thres = dice_dict[-1,0]
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
precision_recall_metric = get_precision_recall()

dice2_list = []
precision_recall_list = []
Value_list = {'Name':[],'Dice':[],'Precision':[],'Recall':[]}
for count,patient_png in enumerate(patients_ordered):
    patient_GT = []
    patient_pred = []
    for img_name in patient_png:
        GT_path = os.path.join(GT,img_name)
        patient_GT.append(np.asarray(Image.open(GT_path)))

        pred_path = os.path.join(prediction,img_name)
        pred_map = np.asarray(Image.open(pred_path)).T

        patient_pred.append(resize_array(pred_map))
    patient_GT = np.array([patient_GT])
    patient_pred = np.array([patient_pred])

    patient_pred[patient_pred<thres]=0
    patient_pred[patient_pred>=thres]=1
    
    dice_temp = dice_metric(y_pred=[torch.tensor(patient_pred)], y=[torch.tensor(patient_GT)])
    dice2_list.append(dice_temp)
    precision_recall_temp = precision_recall_metric([torch.tensor(patient_pred)],[torch.tensor(patient_GT)])
    
    Value_list['Name'].append(patient_png[0])
    Value_list['Dice'].append(dice_temp)
    Value_list['Precision'].append(precision_recall_temp[0])
    Value_list['Recall'].append(precision_recall_temp[1])
    


    
    
# compute dice index
precision_, recall_ = precision_recall_metric.aggregate()
dice3D = dice_metric.aggregate().item()
print("Dice metric:", dice3D)
print("PR metric:", precision_, recall_)
print('Thres:',thres)
# reset the status
dice_metric.reset()
precision_recall_metric.reset()


# In[72]:


# save result
import json
save_path = os.path.join('/'.join(prediction.split('/')[:-1]),'3D_output.json')
save_result = {'precision':precision_,'recall':recall_,'dice3D':dice3D,'Threshold':thres}
print(save_result)
with open(save_path, 'w') as fp:
    json.dump(save_result, fp)
    
# Save csv result
save_tmp = os.path.join('/'.join(prediction.split('/')[:-1]),'3D_output_val.csv')
Value_list = pd.DataFrame(Value_list)
Value_list.to_csv(save_tmp)




