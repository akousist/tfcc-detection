import random
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.transforms as transforms
import torch.utils.data as data
import utils

from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from args import get_train_args
from models import ResNet18, ResNet18_compress1
from utils import TFCCDataset

def main(args):
    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=True)
    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = utils.get_available_devices()
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get data loader
    log.info('Building dataset...')
    transform_composed = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5]*8,
                                                                  std=[0.5]*8)])
    train_dataset = TFCCDataset(args.train_record_file,
                                transform=transform_composed,
                                image_size=args.image_size,
                                plane=args.plane,
                                mri_type=args.mri_type)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    dev_dataset = TFCCDataset(args.dev_record_file,
                              transform=transform_composed,
                              image_size=args.image_size,
                              plane=args.plane,
                              mri_type=args.mri_type)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

    # Get model
    log.info('Building model...')
    model = ResNet18_compress1(num_labels=args.num_labels,
                               pretrained=args.pretrained)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = utils.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = utils.EMA(model, args.ema_decay)

    # Get saver
    saver = utils.CheckpointSaver(args.save_dir,
                                  max_checkpoints=args.max_checkpoints,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)
    
    # Get optimizer and scheduler
    TOTAL_TRAINING_STEPS = args.num_epochs * (len(train_dataset) // args.batch_size + 1)
    warmup_steps = TOTAL_TRAINING_STEPS * args.warmup_ratio
    def lr_lambda(step, warmup_steps=warmup_steps, num_cycles=args.num_cycles):
        if step < warmup_steps:
            return step / warmup_steps
        progress = float(step - warmup_steps) / float(TOTAL_TRAINING_STEPS - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_dataset)) as progress_bar:
            for images, ys, idxs in train_loader:
                # Set up for forward
                images = images.to(device)
                batch_size = images.size(0)
                optimizer.zero_grad()

                # Forward
                log_p = model(images)
                ys = ys.to(device)
                loss = F.nll_loss(log_p, ys)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader,
                                                  device, args.dev_record_file)

                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    utils.visualize(tbx,
                                    pred_dict=pred_dict,
                                    eval_path=args.dev_record_file,
                                    step=step,
                                    split='dev',
                                    num_visuals=args.num_visuals,
                                    plane=args.plane,
                                    mri_type=args.mri_type)

def evaluate(model, data_loader, device, eval_file):
    nll_meter = utils.AverageMeter()

    model.eval()
    pred_dict = {}

    # Load eval info
    gold_dict = np.load(eval_file)

    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
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
    
    model.train()

    results = utils.eval_dicts(gold_dict, pred_dict)
    results_list = [('NLL', nll_meter.avg),
                    ('Acc', results['Acc'])]
    results = OrderedDict(results_list)

    return results, pred_dict

if __name__ == '__main__':
    main(get_train_args())