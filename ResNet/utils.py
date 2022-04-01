import argparse
import logging
import os
import queue
import shutil
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pydicom

from functools import wraps
from pathlib import PurePath
from sklearn import metrics
from tqdm import tqdm

class TFCCDataset(data.Dataset):
    """TFCC dataset.

    Args:
        data_path (str): Path to .npz file containing image paths and labels.
        transform (obj): Transform applied on images.
        image_size (int): Size of output image
    """
    def __init__(self, data_path, transform, image_size, plane='cor', mri_type='t1'):
        super(TFCCDataset, self).__init__()
        data = np.load(data_path, allow_pickle=True)

        self.image_size = image_size
        self.plane = plane
        self.mri_type = mri_type
        self.root = PurePath(data_path).parent
        
        self.files = data['files']
        self.labels = torch.from_numpy(data['labels']).long()
        self.idxs = torch.from_numpy(data['idxs']).long()
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        image_file = self.files[idx] + '.npy'
        image_path = self.root.joinpath(f'{self.plane}_{self.mri_type}', image_file)
        image = np.load(image_path)
        label = self.labels[idx]
        idx = self.idxs[idx]

        image = self.transform(image)
        image = F.interpolate(image.unsqueeze(0), size=[self.image_size, self.image_size], mode='bilinear', align_corners=False).squeeze(0)

        example = (image, label, idx)

        return example

def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir
        
    raise RuntimeError('Too many save directories crewated with the same name. \
                       Delete old save directories or use another name.')

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.
    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.
    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars."""
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    return device, gpu_ids

def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.
    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.
    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)
    
    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model

class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avarage = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_avarage.clone()
    
    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True
        
        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))
        
    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir, f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')
        
        # Add checkpoint path to priority queue (lowesr priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val
        
        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

class AverageMeter:
    """Keep track of average values over time."""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()
    
    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

def predict(idxs, probs):
    """Predict the label according the probability.
    Args:
        ids (list): List of image IDs.
        p (list): List of predicted probability.
    Returns:
        pred_dict (dict): Dictionary index IDs -> predicted label.
    """
    pred_dict = {}
    for idx, prob in zip(idxs, probs):
        label = prob.index(max(prob))
        pred_dict[str(idx)] = (label, prob[1])
    
    return pred_dict

def eval_dicts(gold_data, pred_dict):
    """Compute metrics
    Args:
        gold_data (dict): Dictionary with eval info for the dataset.
        pred_dict (dict): Dictionary of predicted labels
    
    Returns:
        (dict): Dictionary of metrics
    """
    gold_dict = dict(zip(gold_data['idxs'].astype(str), gold_data['labels']))
    acc = total = 0
    ys, probs = [], []
    for key, pred in pred_dict.items():
        total += 1
        (label, prob) = pred
        ground_truth = gold_dict[key]
        acc += compute_acc(label, ground_truth)
        ys.append(ground_truth)
        probs.append(prob)
    
    ys = np.array(ys)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(np.int)
    fpr, tpr, thresholds = metrics.roc_curve(ys, probs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    cm = metrics.confusion_matrix(ys, preds)
    sensitivity = cm[1][1] / sum(cm[1])
    specificity = cm[0][0] / sum(cm[0])
    
    return {'Acc': 100. * acc / total,
            'AUC': 100. * auc,
            'Sensitivity': 100. * sensitivity,
            'Specificity': 100. * specificity}

def visualize(tbx, pred_dict, eval_path, step, split, num_visuals, plane, mri_type):
    """Visualize text examples to TensorBoard

    Args:
        tbx (tensorboard.SummaryWriter): Summary writer.
        pred_dict (dict): Dict of predictions of the form id -> pred.
        eval_path (str): Path to npz file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals < 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)
    
    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)

    eval_data = np.load(eval_path)
    eval_dict = dict(zip(eval_data['idxs'].astype(str), zip(eval_data['files'], eval_data['labels'])))

    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_]
        
        ground_truth = eval_dict[id_][1]

        root = PurePath(eval_path).parent
        image_file = eval_dict[id_][0] + '.npy'
        image_path = root.joinpath(f'{plane}_{mri_type}', image_file)
        image = np.load(image_path)
        image = image.transpose((2, 0, 1))[:, :, :, np.newaxis]

        tag = (f'{split}/{i+1}_of_{num_visuals}'
               + f'- Ground Truth: {ground_truth}'
               + f'- Prediction: {pred}')
        tbx.add_images(tag, image, 0, dataformats='NHWC')

def compute_acc(prediction, ground_truth):
    return int(prediction == ground_truth)

def write_csv(data, csv_path, column_name):
    with open(csv_path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(column_name)
        for row in data:
            writer.writerow(row)

def extractDICOM(input_dir):
    ATTRS = ['Modality', 'SeriesDescription', 'InStackPositionNumber', 'SliceLocation', 'SliceThickness']
    dicom_list = sorted(os.listdir(input_dir))
    dicom_infos = []
    for dicom in tqdm(dicom_list):
        dicom_info = [dicom]
        ds = pydicom.dcmread(PurePath(input_dir, dicom))
        for attr in ATTRS:
            try:
                info = getattr(ds, attr)
            except AttributeError:
                info = None
            dicom_info.append(info)
        dicom_infos.append(dicom_info)

    columns = ['Files'] + ATTRS
    csv_path = PurePath(*input_dir.split('/')[:-1], f"{input_dir.split('/')[-2]}.csv")
    write_csv(dicom_infos, csv_path, columns)
    
    return csv_file

def get_args():
    parser = argparse.ArgumentParser('Utility functions to deal with DICOM file')

    parser.add_argument('--convert_img',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether convert DICOM to image file.')
    parser.add_argument('--extract_info',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether extract information from DICOM.')
    parser.add_argument('--input_dir',
                        type=str,
                        required=True,
                        help='Input DICOM directory')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Output png directory')
    
    args = parser.parse_args()

    if args.convert_img:
        if args.output_dir is None:
            raise argparse.ArgumentError('Missing required argument --output_dir')

    return args

def handle_path(func):
    """Decorator that handle the absolute path."""
    @wraps(func)
    def decorate(*args, **kwargs):
        parent_path = PurePath(__file__).parent
        results = func()
        for key, value in vars(results).copy().items():
            if isinstance(value, str) and '/' in value:
                setattr(results, key, str(PurePath(parent_path, value)))
        return results
    return decorate

if __name__ == '__main__':
    args = get_args()
    if args.convert_img:
        DICOM2PNG(args)
    if args.extract_info:
        extractDICOM(input_dir=args.input_dir)
