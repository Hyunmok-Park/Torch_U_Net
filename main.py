from datetime import datetime
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np

from data_factory.data_loader import load_dataset
from model.UNet import UNet
from torchvision.utils import save_image

def train(config):
    EXP_NAME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(f"result/{EXP_NAME}", exist_ok=True)
    os.makedirs(f"result/{EXP_NAME}/test", exist_ok=True)

    train_loader = load_dataset(config['data_path'], 'train', config['batch_size'], config['img_size'])
    val_loader = load_dataset(config['data_path'], 'val', config['batch_size'], config['img_size'])
    test_loader = load_dataset(config['data_path'], 'test', 1, config['img_size'])

    model = UNet().to(config['device'])
    criterion = nn.BCEWithLogitsLoss().to(config['device'])
    optimizer = optim.Adam(params=model.parameters(), lr=config['learning_rate'])

    train_loss = []
    val_loss = []

    min_val_loss = np.inf

    for epoch in tqdm(range(config['epoch']), desc='EPOCH'):
        train_loss_batch = []
        val_loss_batch = []

        ###########
        #  TRAIN  #
        ###########
        model.train()
        for batch, (inputs, label) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(config['device']).type(torch.float32)
            label = label.to(config['device']).type(torch.float32)
            output = model.forward(inputs)
            loss = criterion(label, output)
            loss.backward()
            optimizer.step()
            train_loss_batch.append(loss.item())

        torch.save(model.state_dict(), f'result/{EXP_NAME}/checkpoint.pth')
        train_loss.append(np.mean(train_loss_batch))

        ##########
        #  EVAL  #
        ##########
        model.eval()
        for batch, (inputs, label) in enumerate(val_loader):
            inputs = inputs.to(config['device']).type(torch.float32)
            label = label.to(config['device']).type(torch.float32)
            output = model.forward(inputs)
            loss = criterion(label, output)
            val_loss_batch.append(loss.item())

        if min_val_loss > np.mean(val_loss_batch):
            min_val_loss = np.mean(val_loss_batch)
            torch.save(model.state_dict(), f'result/{EXP_NAME}/checkpoint_best_val.pth')

        val_loss.append(np.mean(val_loss_batch))

    ##########
    #  TEST  #
    ##########
    to_class = lambda x: 1.0 * (x > 0.5)
    model.load_state_dict(torch.load(f'result/{EXP_NAME}/checkpoint.pth'))
    model.eval()
    for idx, (inputs, label) in enumerate(test_loader):
        inputs = inputs.to(config['device']).type(torch.float32)
        label = label.to(config['device']).type(torch.float32)
        output = model.forward(inputs)

        inputs = inputs.detach().cpu().numpy().reshape(config['img_size'], config['img_size'])
        label = label.detach().cpu().numpy().reshape(config['img_size'], config['img_size'])
        output = output.detach().cpu().numpy().reshape(config['img_size'], config['img_size'])
        output = to_class(output)

        f, ax = plt.subplots(1, 3, figsize=(10, 4))
        ax[0].imshow(inputs, cmap='gray')
        ax[1].imshow(label, cmap='gray')
        ax[2].imshow(output, cmap='gray')

        f.savefig(f"result/{EXP_NAME}/test/{idx}.png")
        plt.close()

    model.load_state_dict(torch.load(f'result/{EXP_NAME}/checkpoint_best_val.pth'))
    model.eval()
    for idx, (inputs, label) in enumerate(test_loader):
        inputs = inputs.to(config['device']).type(torch.float32)
        label = label.to(config['device']).type(torch.float32)
        output = model.forward(inputs)

        inputs = inputs.detach().cpu().numpy().reshape(config['img_size'], config['img_size'])
        label = label.detach().cpu().numpy().reshape(config['img_size'], config['img_size'])
        output = output.detach().cpu().numpy().reshape(config['img_size'], config['img_size'])
        output = to_class(output)

        f, ax = plt.subplots(1, 3, figsize=(10, 4))
        ax[0].imshow(inputs, cmap='gray')
        ax[1].imshow(label, cmap='gray')
        ax[2].imshow(output, cmap='gray')

        f.savefig(f"result/{EXP_NAME}/test/{idx}_val.png")
        plt.close()

    f, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.plot(train_loss, color='blue')
    ax.plot(val_loss, color='red')

    f.savefig(f"result/{EXP_NAME}/train_loss.png")
    plt.close()


if __name__ == '__main__':
    config = {}
    config['num_channel'] = 3
    config['img_size'] = 512

    config['data_path'] = 'dataset'

    config['batch_size'] = 4
    config['learning_rate'] = 0.0001
    config['epoch'] = 200

    config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train(config)