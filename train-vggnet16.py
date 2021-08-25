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
from data_loader_originVGG import Tomographic_Dataset

from UNET import VGGNet, UNETmodel, UNETsota
from REDCNN2D import RED_CNN
from GOOGLENET2D import GOOGLENET2Dmodel
from VGG import vgg16

import numpy as np
import time

import os

type = "VGG16"
net = "VGG"

batch_size = 25  # antes 10
epochs = 15
chkepoch = 0


momentum = 0.5
w_decay = 0  # antes 1e-5

# after each 'step_size' epochs, the 'lr' is reduced by 'gama'
lr = 0.00001  # antes le-4
step_size = 10
gamma = 0.5


configs = "{}-model".format(net)
n_class = 2
train_file = "training.csv"
val_file = "validation.csv"


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



fcn_model = vgg16()

if use_gpu:
    ts = time.time()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.CrossEntropyLoss()


optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)


def train(chkepoch):
    for epoch in range(epochs):
        chkepoch += 1
        ts = time.time()
        fcn_model.train()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = Variable(batch['X']).cuda()
                labels = Variable(batch['Y'].flatten()).cuda()
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs, _ = fcn_model(inputs)

            loss = criterion(outputs, labels)
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
    mse = 0.0
    tam = 0

    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output, _  = fcn_model(inputs)
        output = output.data.cpu().numpy()


        target = batch['l'].cpu().numpy()
        for p, t in zip(output, target):
            if p.argmax() == t:
                mse += 1
            tam += 1

    mse_accs = mse/tam
    validation_accuracy[epoch] = mse_accs

    print("epoch{}, mse_acc: {}".format(epoch, mse_accs))


def mse_acc(pred, target):
    return np.mean(np.square(pred - target))

if __name__ == "__main__":
    # val(0)  # show the accuracy before training
    start = time.time()
    train(chkepoch)
    end = time.time()
    duration = end - start

    d = datetime(1, 1, 1) + timedelta(seconds=int(duration))
    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))

