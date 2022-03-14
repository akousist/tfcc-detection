import argparse


def get_setup_args():
    """Get arguments needed in main.py"""
    parser = argparse.ArgumentParser('Set up on TFCC MRNet')

    add_train_args(parser)

    parser.add_argument('--root',
                        type=str,
                        default='./', # /home/tedwu430/tfcc
                        help='The root path.')
    parser.add_argument('--tb_name',
                        type=str,
                        default='tb_tfcc',
                        help='The name of the tensorboardX folder.')
    parser.add_argument('--plane',
                        type=str,
                        default='cor',
                        choices=('ax', 'sag', 'cor'),
                        help='Planes of MRI images.')
    parser.add_argument('--opt_name',
                        type=str,
                        default='RangerLarsR',
                        help='What optimizer to use.')
    
    args = parser.parse_args()

    return args

def add_train_args(parser):
    """Get arguments needed in get_setup_args"""
    # Train a model on Knee Symptom
    parser.add_argument('--num_classes',
                        type=int,
                        default=2,
                        help='There are several categories of output.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size per GPU, it can only be 1.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--t_max',
                        type=int,
                        default=10,
                        help='The T_max of the cosine-annealing-restarts.')
    parser.add_argument('--eta_min',
                        type=float,
                        default=1e-6,
                        help='The eta_min of the cosine-annealing-restarts.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20, # 50
                        help='Number of epochs for which to train.')
    parser.add_argument('--n_epochs_stop',
                        type=int,
                        default=15, # 25
                        help='Number of epochs to stop train.')