#!/usr/bin/env python
# coding: utf-8

# In[1]:

print('Traing script V2.\nIt adding the computing of 3D dice and will save related data.')


import logging
import os
import sys
import tempfile
from glob import glob
import argparse
import numpy as np
import json 
import yaml

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.functional import precision_recall

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,)
from monai.visualize import plot_2d_or_3d_image


monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from dataset import ArrayDataset_name

from CTE_Net import net



def model_sum(model, torchsummary_ = False):
    param_size = 0
    param_num = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_num += param.nelement()
    buffer_size = 0
    buffer_num = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_num += buffer.nelement()

    print(f'param_size: {param_size}')
    print(f'param_num: {param_num}')
    print(f'buffer_size: {buffer_size}')
    print(f'buffer_num: {buffer_num}')
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    if torchsummary_:
        from torchsummary import summary
        summary(model, input_size=(1,400,400), batch_size=-1, device='cuda')


# In[3]:



parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./Unet/unetTEv3_1.yaml')
parser.add_argument('--torchsummary', type=bool, default=False)
config = parser.parse_args()
print(config)


# In[4]:


# load yaml
with open(config.config_path, 'r') as f:
    hyper = yaml.full_load(f)
print(hyper['MODE'])

# enter specific dir
ori_path = os.getcwd()
os.makedirs(hyper['MODEL']['NAME'], exist_ok = True)
os.chdir('./'+hyper['MODEL']['NAME'])

if hyper['MODEL']['NAME']=='Swin_Unet':
    config_ = get_config(config)
    


# In[5]:


# setting
if hyper['DATA_CONFIG']['use_sample']:
    image_path_train = hyper['DATA_CONFIG']['Sampled']['image_path_train']
    seg_path_train = hyper['DATA_CONFIG']['Sampled']['seg_path_train']
else:
    image_path_train = hyper['DATA_CONFIG']['Origin']['image_path_train']
    seg_path_train = hyper['DATA_CONFIG']['Origin']['seg_path_train']


image_path_val = hyper['DATA_CONFIG']['image_path_val']
seg_path_val = hyper['DATA_CONFIG']['seg_path_val']

# For dataloading
window_size = hyper['DATA_PREPROCESS']['window_size']

# For training
lr = hyper['OPTIMIZATION']['LR']
total_epoch = hyper['OPTIMIZATION']['NUM_EPOCHS']
val_interval = hyper['OPTIMIZATION']['VAL_INTERVAL']
batch_size = hyper['OPTIMIZATION']['BATCH_SIZE']
torch.cuda.set_device(hyper['OPTIMIZATION']['cuda_idx'])


# In[6]:



# define transforms for image and segmentation
from monai.transforms.intensity.array import RandShiftIntensity,RandGaussianNoise
from monai.transforms.spatial.array import RandFlip, Rand2DElastic,RandAxisFlip,RandAffine

train_imtrans = eval(hyper['DATA_PREPROCESS']['Train']['Img_trans'])
train_segtrans = eval(hyper['DATA_PREPROCESS']['Train']['Seg_trans'])
val_imtrans = eval(hyper['DATA_PREPROCESS']['Val']['Img_trans'])
val_segtrans = eval(hyper['DATA_PREPROCESS']['Val']['Seg_trans'])


# In[7]:



# load training data
images = sorted(glob(os.path.join(image_path_train, "Lung_***_***.png")))
segs = sorted(glob(os.path.join(seg_path_train, "Lung_***_***.png")))
train_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
# create a training data loader
train_ds = ArrayDataset_name(images, train_imtrans, segs, train_segtrans)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())


# load validation data
images = sorted(glob(os.path.join(image_path_val, "Lung_***_***.png")))
segs = sorted(glob(os.path.join(seg_path_val, "Lung_***_***.png")))
val_files = [{"img": img, "seg": seg} for img, seg in zip(images[:], segs[:])]
# create a validation data loader
val_ds = ArrayDataset_name(images, val_imtrans, segs, val_segtrans)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=torch.cuda.is_available())

print(f'Length of train {len(train_ds)}')
print(f'Length of val {len(val_ds)}')


# In[8]:


dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = eval(hyper['DATA_PREPROCESS']['post_trans'])


# In[9]:



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create UNet, DiceLoss and Adam optimizer
model = eval(hyper['MODEL']['NETWORK']).to(device)
print(hyper['MODEL']['NETWORK'])
try:
    if hyper['MODEL']['PRETRAIN']:
        model.load_from(config_)
        print('Load pretrained model')
except: print('Error loading pretrained model')

try:
    model_sum(model,config.torchsummary)
except Exception as e: print(e)


# In[10]:


loss_function = eval(hyper['OPTIMIZATION']['loss'])
optimizer = eval(hyper['OPTIMIZATION']['OPTIMIZER'])
scheduler = eval(hyper['OPTIMIZATION']['SCHEDULER'])


# In[11]:


# load model
if hyper['OPTIMIZATION']['continue_training']:
    model_path = os.path.join(hyper['OPTIMIZATION']['continue_path'],"latest_model_segmentation2d_array.pth")
    fed = model.load_state_dict(torch.load(model_path))
    print(fed,':',model_path)


# In[89]:


def get_3Ddice_(Dice_value):
    # compute the 3D dice with recorded dict
    # initialize data structure
    patients_name = sorted(list(set(['_'.join(i.split('_')[:2]) for i in Dice_value['name']])))
    patients_value = {}
    for i in patients_name:
        patients_value[i] = [0,0,0]
    # count for each patient
    for i in range(len(Dice_value['name'])):
        patient_name = '_'.join(Dice_value['name'][i].split('_')[:2])
        patients_value[patient_name][0]+=Dice_value['pred_size'][i]
        patients_value[patient_name][1]+=Dice_value['label_size'][i]
        patients_value[patient_name][2]+=Dice_value['Union_size'][i]
    # count the 3D dice
    dice3d_patient = []
    for key,item in patients_value.items():
        dice3d_tmp = 2*item[2]/(item[0]+item[1])
        dice3d_patient.append(dice3d_tmp)
    return np.mean(dice3d_patient)


# In[94]:


def run_validation(model,val_loader,mode = 'Val',tensorboard_writer = True,save_best = True):
    global best_metric,best_metric_epoch,epoch_loss_values,metric_values,writer,metric_values_train,best_metric_3d
    model.eval()
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        Dice_value = {'name':[],'pred_size':[],'label_size':[],'Union_size':[]}
        
        for ik,val_data in enumerate(val_loader):
            val_names = [i.split('/')[-1] for i in val_data[2]]
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            roi_size = (window_size, window_size)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model,mode="gaussian")
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            # count the value for 3D dice
            for i in range(len(val_names)):
                Dice_value['name'].append(val_names[i])
                Dice_value['pred_size'].append(val_outputs[i].sum().cpu().item())
                Dice_value['label_size'].append(val_labels[i].sum().cpu().item())
                Dice_value['Union_size'].append((val_outputs[i]*val_labels[i]).sum().cpu().item())

            # record image
            if tensorboard_writer and ik==400:
                #np.random.randint(0,3000)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # get the 3D dice
        dice3D = get_3Ddice_(Dice_value)
#         print(dice3D)
#         print(Dice_value)
        # reset the status for next validation round
        dice_metric.reset()
        if mode =='Val': metric_values.append(metric)
        elif mode == 'Train':metric_values_train.append(metric)
        
        if metric > best_metric and save_best:
            best_metric = metric
            best_metric_epoch = epoch + 1
            # save model
            torch.save(model.state_dict(), os.path.join('./'+writer.log_dir,"best_metric_model_segmentation2d_array.pth"))
            with open(os.path.join('./'+writer.log_dir,"training_progress.json"), "w") as outfile:
                json.dump({'epoch':epoch,'Lr':optimizer.param_groups[0]['lr'],'Dice':best_metric,'Dice3D':dice3D}, outfile)
            print("saved new best metric model")
        if dice3D > best_metric_3d and save_best:
            best_metric_3d = dice3D
            best_metric_epoch_3d = epoch + 1
            # save model
            torch.save(model.state_dict(), os.path.join('./'+writer.log_dir,"best_3Ddice_model_segmentation2d_array.pth"))
            with open(os.path.join('./'+writer.log_dir,"training_progress_3D.json"), "w") as outfile:
                json.dump({'epoch':epoch,'Lr':optimizer.param_groups[0]['lr'],'Dice':best_metric,'Dice3D':dice3D}, outfile)
            print("saved new best 3D dice model")
        # save latest
        torch.save(model.state_dict(), os.path.join('./'+writer.log_dir,"latest_model_segmentation2d_array.pth"))
        
        print(
            mode+': '+"current epoch: {} current mean dice: {:.4f} best mean 2D dice: {:.4f} 3D dice: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_3d, best_metric_epoch))
        
        writer.add_scalar(mode+"_mean_dice", metric, epoch + 1)
        writer.add_scalar(mode+"_mean_dice_3D", dice3D, epoch + 1)


# In[ ]:


# start a typical PyTorch training
best_metric = -1
best_metric_3d = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
metric_values_train = list()

if hyper['OPTIMIZATION']['continue_training']:
    writer = SummaryWriter(log_dir = hyper['OPTIMIZATION']['continue_path'])
    print(f'Set path as '+hyper['OPTIMIZATION']['continue_path'])
else:
    writer = SummaryWriter()
save_path = './'+writer.log_dir

# save hyper parameter
converted_dict = vars(config)
with open(os.path.join('./'+writer.log_dir,"argparse.json"), "w") as outfile:
    json.dump(converted_dict, outfile)
# os.system('cp '+config.config_path+' '+'./'+writer.log_dir)
with open(os.path.join('./'+writer.log_dir,"config.yaml"), 'w') as outfile:
    yaml.dump(hyper, outfile, default_flow_style=False)
print(f'Model folder: {writer.log_dir}')

try:
    dummy_input = torch.rand(1, 1, window_size, window_size).to(device)
    with SummaryWriter() as w:
        w.add_graph(model, (dummy_input, ))
except Exception as e:print(e)

for epoch in range(total_epoch):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{total_epoch}")
    model.train()
    epoch_loss = 0
    step = 0
    for iz,batch_data in enumerate(train_loader):
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyper['OPTIMIZATION']['GRAD_NORM_CLIP'])
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        if iz%100 ==0:
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'],epoch)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    if (epoch + 1) % val_interval == 0:
        run_validation(model,val_loader,mode = 'Val',tensorboard_writer = True,save_best = True)
        run_validation(model,train_loader,mode = 'Train',tensorboard_writer = False,save_best = False)
        
    scheduler.step(best_metric)
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

writer.close()


# In[ ]:





