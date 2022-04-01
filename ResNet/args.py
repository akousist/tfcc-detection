"""Command-line arguments for setup.py"""
import argparse

from utils import handle_path

@handle_path
def get_setup_args():
    """Get arguments needed in setup.py"""
    parser = argparse.ArgumentParser("Pre-process TFCC DICOM files")

    add_common_args(parser)

    parser.add_argument('--convert_dicom',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to convert dicom to npy file.')
    parser.add_argument('--input_dir',
                        type=str,
                        default='./raw_dicom',
                        help='Input directory of list of DICOM directory.')
    parser.add_argument('--stack_number',
                        type=int,
                        default=0,
                        help='Number of DICOM stacked for a image. 0 for stack all DICOM together')
    parser.add_argument('--data_file',
                        type=str,
                        default='./tfcc_342_random.csv')
    parser.add_argument('--train_split',
                        type=float,
                        default=0.82,
                        help='Training data split ratio.')
    parser.add_argument('--test_split',
                        type=float,
                        default=0.04,
                        help='Testing data split ratio.')
    parser.add_argument('--train_meta_file',
                        type=str,
                        default='./data/train_meta.json')
    parser.add_argument('--dev_meta_file',
                        type=str,
                        default='./data/dev_meta.json')
    parser.add_argument('--test_meta_file',
                        type=str,
                        default='./data/test_meta.json')

    args = parser.parse_args()

    return args

@handle_path
def get_train_args():
    """Get arguments needed in train.py"""
    parser = argparse.ArgumentParser("Train a model on TFCC")

    add_common_args(parser)
    add_train_test_args(parser)
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=3,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='Acc',
                        choices=('NLL', 'Acc'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help='Default learning rate for fine-tuning.')
    parser.add_argument('--lr_c',
                        type=float,
                        default=1e-3,
                        help='Default learning rate for the classifier layer.')
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1,
                        help='Steps of a lr warm-up scheme.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0.01,
                        help='L2 weight decay.')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=1000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=4,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--num_cycles',
                        type=float,
                        default=0.5,
                        help='The number of waves in the cosine schedule')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')

    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name == 'Acc':
        # Best checkpoint is the one that maximizes accuracy
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args

@handle_path
def get_test_args():
    """Get arguments need in test.py"""
    parser = argparse.ArgumentParser('Test a trained model on TFCC')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=('train', 'dev', 'test'))

    args = parser.parse_args()

    if args.load_path is None:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args

def add_common_args(parser):
    """Add arguments common to setup.py"""
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')
    parser.add_argument('--seed',
                        type=int,
                        default=995,
                        help='Random seed')

def add_train_test_args(parser):
    """Add arguments to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=5,
                        help='Batch size per GPU. Scales automatically \
                             when multiple GPUs are available.')
    parser.add_argument('--num_labels',
                        type=int,
                        default=2,
                        help='Number of sentiment labels.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=512,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=3,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--plane',
                        type=str,
                        default='cor',
                        choices=('ax', 'cor', 'sag'),
                        help='Planes of MRI images.')
    parser.add_argument('--mri_type',
                        type=str,
                        default='t1',
                        choices=('t1', 't2'),
                        help='Types of MRI images.')
    parser.add_argument('--image_size',
                        type=int,
                        default=256,
                        help='Size of scaled images.')
    parser.add_argument('--pretrained',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use pretrained model.')