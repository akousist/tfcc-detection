import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchtools.optim import RangerLars

import os
import configparser
import numpy as np
import pandas as pd
from pathlib import PurePath, Path
import warnings
warnings.filterwarnings("ignore")

from args import get_setup_args
from dataloader import tfcc_dataloader
from model import TFCCNet
from lr_scheduler import CosineAnnealingWithRestartsLR
from train import train_model
from utils import balance_weights, cm_result

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

def main(args):
    # Setup
    TB_DIR = Path(args.root, args.tb_name, args.plane)
    # The folder where the logs are stored.
    LOG_DIR = Path(TB_DIR, f'{args.opt_name}/train/acc')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_writer = SummaryWriter(log_dir = LOG_DIR)
    LOG_DIR = Path(TB_DIR, f'{args.opt_name}/train/loss')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_loss_writer = SummaryWriter(log_dir = LOG_DIR)
    LOG_DIR = Path(TB_DIR, f'{args.opt_name}/valid/acc')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    valid_writer = SummaryWriter(log_dir = LOG_DIR)
    LOG_DIR = Path(TB_DIR, f'{args.opt_name}/valid/loss')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    valid_loss_writer = SummaryWriter(log_dir = LOG_DIR)
    LOG_IMG_DIR = Path(TB_DIR, f'{args.opt_name}/cm')
    LOG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    image_writer = SummaryWriter(LOG_IMG_DIR)
    # The folder where the model are stored.
    MODEL_DIR = Path(args.root, f'model/{args.plane}/{args.opt_name}')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = Path(MODEL_DIR, CONFIG['data']['model_1'])
    MODEL_PATH2 = Path(MODEL_DIR, CONFIG['data']['model_2'])
    # The folder where the confunsion matrix(images) are stored
    IMAGE_DIR = Path(args.root, f'cm/{args.plane}/{args.opt_name}')
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    VALID_IMAGE = Path(IMAGE_DIR, CONFIG['data']['confunsion_matrix_1'])
    VALID_LOSS_IMAGE = Path(IMAGE_DIR, CONFIG['data']['confunsion_matrix_2'])
    # data
    TRAIN_CSV = PurePath(args.root, CONFIG['data']['train_excel'])
    VALID_CSV = PurePath(args.root, CONFIG['data']['valid_excel'])
    TRAIN_PATH = Path(args.root, f'data/{args.plane}_t1')
    VALID_PATH = Path(args.root, f'data/{args.plane}_t1')
    train_data = pd.read_csv(TRAIN_CSV)
    train_data_0 = train_data[train_data['LABEL']==0].sample(frac=5,
                                                            replace=True,
                                                            random_state=1)
    train_data_1 = train_data[train_data['LABEL']==1].sample(frac=6.5,
                                                            replace=True,
                                                            random_state=1)
    train_data = pd.concat([train_data_0, train_data_1])
    valid_data = pd.read_csv(VALID_CSV)

    # TFCC dataloader
    dataloaders_dict = tfcc_dataloader(train_data, valid_data, TRAIN_PATH,  \
                                       VALID_PATH, args.batch_size)

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Send the model to GPU
    pnet = TFCCNet(args.num_classes).to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = RangerLars(pnet.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWithRestartsLR(optimizer_ft, T_max=args.t_max,   \
                                              T_mult=1, eta_min=args.eta_min)
    
    # Run
    # Setup the loss fxn
    weights = balance_weights(train_data, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # Train and evaluate
    train_model(pnet, dataloaders_dict, device, criterion, optimizer_ft,    \
                train_writer, train_loss_writer, valid_writer,  \
                valid_loss_writer,  scheduler, args.num_epochs, \
                args.n_epochs_stop, MODEL_PATH, MODEL_PATH2)
    
    # Confusion Matrix & Classification Report
    cm_result(args.num_classes, MODEL_PATH, dataloaders_dict, device,   \
              valid_data, VALID_IMAGE, image_writer, 1)
    cm_result(args.num_classes, MODEL_PATH2, dataloaders_dict, device,  \
              valid_data, VALID_LOSS_IMAGE, image_writer, 2)


if __name__ == '__main__':
    main(get_setup_args())