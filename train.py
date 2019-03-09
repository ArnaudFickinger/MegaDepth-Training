import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
import itertools
from options.train_options import TrainOptions
import sys
from data.data_loader import CreateDataLoader
from data.data_loader import CreateDIWDataLoader
import math
from models.models import create_model
import h5py 
from scipy import misc


opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = "/"

train_list_dir_landscape = root + '/phoenix/S6/zl548/MegaDpeth_code/train_list/landscape/'
input_height = 240 
input_width = 320
data_loader_l = CreateDataLoader(root, train_list_dir_landscape, input_height, input_width)
dataset_l = data_loader_l.load_data()
dataset_size_l = len(data_loader_l)
print('========================= training landscape  images = %d' % dataset_size_l)


train_list_dir_portrait = root + '/phoenix/S6/zl548/MegaDpeth_code/train_list/portrait/'
input_height = 320 
input_width = 240
data_loader_p = CreateDataLoader(root, train_list_dir_portrait, input_height, input_width)
dataset_p = data_loader_p.load_data()
dataset_size_p = len(data_loader_p)
print('========================= training portrait  images = %d' % dataset_size_p)


_isTrain =  False
batch_size = 32
num_iterations_L = (dataset_size_l)/batch_size
num_iterations_P = (dataset_size_p)/batch_size
model = create_model(opt, _isTrain)
model.switch_to_train()

best_loss = 100

print("num_iterations ", num_iterations_L, num_iterations_P)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch =0
total_iteration = 0

print("=================================  BEGIN TRAINING =====================================")


# best_loss = validation(model, test_dataset, test_dataset_size)
# print("best_loss  ", best_loss)

start_diw_idx = -1
valiation_interval = 300


for epoch in range(0, 20):

    if epoch > 0:
        model.update_learning_rate()

    # landscape 
    for i, data in enumerate(dataset_l):
        total_iteration = total_iteration + 1
        print('L epoch %d, iteration %d, best_loss %f num_iterations %d best_epoch %d' % (epoch, i, best_loss, num_iterations_L, best_epoch) )
        stacked_img = data['img_1']
        targets = data['target_1']
        is_DIW = False
        model.set_input(stacked_img, targets, is_DIW)
        model.optimize_parameters(epoch)

    # portrait 
    for i, data in enumerate(dataset_p):
        total_iteration = total_iteration + 1
        print('P epoch %d, iteration %d, best_loss %f num_iterations %d best_epoch %d' % (epoch, i, best_loss, num_iterations_P, best_epoch) )
        stacked_img = data['img_1']
        targets = data['target_1']
        is_DIW = False
        model.set_input(stacked_img, targets, is_DIW)
        model.optimize_parameters(epoch)

print("We are done")
