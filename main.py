from datetime import datetime
import os

import torch
from tqdm import tqdm
import numpy as np

from data_factory.data_loader import load_dataset
from model.gan import PePe_GAN
from torchvision.utils import save_image

def train(config, verbose):
    EXP_NAME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    data_loader = load_dataset(config['data_path'], config['dataset'], config['batch_size'], config['img_size'])
    model = PePe_GAN(config).to(config['device'])

    for epoch in tqdm(range(config['epoch']), desc='EPOCH'):
        model.train()
        for batch, (real_image, label) in enumerate(data_loader):
            inputs = real_image.view(-1, config['img_size'] * config['img_size'] * config['num_channel']).to(config['device'])
            g_loss = model.update_generator(config['batch_size'])
            d_loss = model.update_discriminator(inputs, config['batch_size'])
            torch.save(model.state_dict(), f'result/{config["dataset"]}/{EXP_NAME}/checkpoint.pth')
        model.eval()
    torch.save(model.state_dict(), f'result/{config["dataset"]}/{EXP_NAME}/last_checkpoint.pth')


if __name__ == '__main__':
    config = {}
    config['latent_dim'] = 32
    config['hidden_dim1'] = 64
    config['hidden_dim2'] = 128
    config['hidden_dim3'] = 256

    config['img_size'] = 128
    config['num_channel'] = 3

    config['data_path'] = 'dataset'
    config['dataset'] = 'pepe'
    config['num_sample'] = 1

    config['batch_size'] = 16
    config['learning_rate'] = 0.0002
    config['epoch'] = 500

    config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train(config, True)