"""Test a model and calcuate accuracy"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import utils

from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from args import get_test_args
from models import ResNet18_compress1
from utils import TFCCDataset

def main(args):
    # Set up logging
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=False)
    log = utils.get_logger(args.save_dir, args.name)
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    device, args.gpu_ids = utils.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))

    # Get model
    log.info('Building model...')
    model = ResNet18_compress1(args.num_labels, args.pretrained)
    model = nn.DataParallel(model, args.gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = utils.load_model(model, args.load_path, args.gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    transform_composed = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5]*8,
                                                                  std=[0.5]*8)])
    dataset = TFCCDataset(record_file,
                          transform=transform_composed,
                          image_size=args.image_size,
                          plane=args.plane,
                          mri_type=args.mri_type)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)
    
    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = utils.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    eval_file = vars(args)[f'{args.split}_record_file']
    gold_dict = np.load(eval_file)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for images, ys, idxs in data_loader:
            # Set up for forward
            images = images.to(device)
            batch_size = images.size(0)

            # Forward
            log_p = model(images)
            ys = ys.to(device)
            loss = F.nll_loss(log_p, ys)
            nll_meter.update(loss.item(), batch_size)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            # Get accuracy
            probs = log_p.exp()
            
            preds = utils.predict(idxs.tolist(), probs.tolist())
            pred_dict.update(preds)
        
        results = utils.eval_dicts(gold_dict, pred_dict)
        results_list = [('NLL', nll_meter.avg),
                        ('Acc', results['Acc']),
                        ('AUC', results['AUC']),
                        ('Sensitivity', results['Sensitivity']),
                        ('Specificity', results['Specificity'])]
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        utils.visualize(tbx,
                        pred_dict=pred_dict,
                        eval_path=eval_file,
                        step=0,
                        split=args.split,
                        num_visuals=args.num_visuals,
                        plane=args.plane,
                        mri_type=args.mri_type)

if __name__ == '__main__':
    main(get_test_args())