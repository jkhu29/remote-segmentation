import random
import warnings

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm

import utils
import config
from models import DANet
from models import DeepLabV3
from losses import SegmentationLoss


torch.autograd.set_detect_anomaly(True)

opt = config.get_options()

# deveice init
CUDA_ENABLE = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA_ENABLE and opt.cuda else 'cpu')
#device = 'cpu'
# seed init
manual_seed = opt.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# dataset init, train file need .tfrecord
description = {
    "image": "byte",
    "label": "byte",
    "size": "int",
}
train_dataset = TFRecordDataset("train.tfrecord", None, description)
# do not shuffle
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True
)
length = 35000

# models init
model = DANet(n_classes=9, aux=False).to(device)

# criterion init
criterion = SegmentationLoss(cuda=opt.cuda).build_loss(mode='ce')

# optim and scheduler init
model_optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1)
model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

# train model
print("-----------------train-----------------")
for epoch in range(opt.niter):
    model.train()
    epoch_losses = utils.AverageMeter()

    with tqdm(total=(length - length % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

        for record in train_dataloader:
            inputs = record["image"].reshape(
                opt.batch_size,
                3,
                record["size"][0],
                record["size"][0],
            ).float().to(device)
            labels = record["label"].reshape(
                opt.batch_size,
                record["size"][0],
                record["size"][0],
            ).long().to(device)

            model_optimizer.zero_grad()

            out = model(inputs)[0]
            loss = criterion(out, labels)

            loss.backward()

            model_optimizer.step()
            epoch_losses.update(loss.item(), opt.batch_size)

            t.set_postfix(
                loss='{:.6f}'.format(epoch_losses.avg),
            )
            t.update(opt.batch_size)

    model_scheduler.step()

    torch.save(model.state_dict(), "danet_epoch_{}.pth".format(epoch))
