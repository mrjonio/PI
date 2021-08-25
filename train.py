# -*- coding: utf-8 -*-

from __future__ import print_function
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader_origin import Tomographic_Dataset

from UNET import VGGNet, UNETmodel, UNETsota
from REDCNN2D import RED_CNN
from GOOGLENET2D import GOOGLENET2Dmodel

import numpy as np
import time

import os

view = 1
pre = False
filename = "checkpoint.pth.tar"

type = "VGGLoss"

#net = "UNET"
net = "GOOGLENET"
#net = "REDCNN"

batch_size = 15  # antes 10
epochs = 15
chkepoch = 0


momentum = 0.5
w_decay = 0  # antes 1e-5

# after each 'step_size' epochs, the 'lr' is reduced by 'gama'
lr = 0.0001  # antes le-4
step_size = 10
gamma = 0.5


configs = "{}-model".format(net)
n_class = 2
train_file = "training-final.csv"
val_file = "validation-final.csv"


path = "/home/cj/Documentos/Pesquisa"


input_dir = "/home/calves/Documents/Pesquisa/DATASET-256-LOW-DOSE/90_projections/"
target_dir = "/home/calves/Documents/Pesquisa/DATASET-256-LOW-DOSE/90_projs_target/"


validation_accuracy = np.zeros((epochs, 1))


# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, configs)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

model_src = "./models/{}-model-{}".format(net, type)


print("GPU Available: ", use_gpu, " number: ", len(num_gpu))

train_data = Tomographic_Dataset(csv_file=train_file, phase='train', train_csv=train_file, input_dir=input_dir,
                                 target_dir=target_dir)


train_loader =    DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)

val_data = Tomographic_Dataset(csv_file=val_file, phase='val', flip_rate=0, train_csv=train_file, input_dir=input_dir,
                               target_dir=target_dir)
val_loader = DataLoader(val_data, batch_size=10, num_workers=6)



#fcn_model = RED_CNN()
#fcn_model = UNETsota()
fcn_model = GOOGLENET2Dmodel()

if use_gpu:
    ts = time.time()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

#criterion = nn.MSELoss()

vgg = torch.load("/home/calves/Documents/Pesquisa/models/VGG-model-VGG16")
vgg.cuda()
vgg.eval()

criterion = nn.L1Loss()

optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)


def train(chkepoch):
    if pre:
        openChekpoint(epochs)
    for epoch in range(epochs):
        chkepoch += 1
        ts = time.time()
        fcn_model.train()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = Variable(batch['X']).cuda()
                labels = Variable(batch['Y']).cuda()
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)


            _, x = vgg(outputs)
            _, y = vgg(labels)


            loss = criterion(x, y)
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(chkepoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(chkepoch, time.time() - ts))

        torch.save(fcn_model, model_src)

        val(epoch)
        scheduler.step()


def val(epoch):
    fcn_model.eval()
    total_mse = []

    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output  = fcn_model(inputs)
        output  = output.data.cpu().numpy()

        N, _, h, w = output.shape
        #pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        pred = output[0, 0, :, :]

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_mse.append(mse_acc(p, t))

    mse_accs = np.mean(total_mse)
    validation_accuracy[epoch] = mse_accs

    print("epoch{}, mse_acc: {}".format(epoch, mse_accs))


def mse_acc(pred, target):
    return np.mean(np.square(pred - target))

def openChekpoint(epochs, chkepoch):
    checkpoint = torch.load(filename)
    fcn_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = epochs - checkpoint['epoch']
    chkepoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Checkpoint loss:" + str(loss))
    print("Epochs left:" + str(epochs))

if __name__ == "__main__":
    # val(0)  # show the accuracy before training
    start = time.time()
    train(chkepoch)
    end = time.time()
    duration = end - start

    d = datetime(1, 1, 1) + timedelta(seconds=int(duration))
    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))

