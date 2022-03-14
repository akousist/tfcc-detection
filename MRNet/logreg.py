import warnings
warnings.filterwarnings("ignore")
import joblib
import argparse
import configparser
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

from dataloader import tfcc_dataloader
from model import TFCCNet

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

parser = argparse.ArgumentParser('Set up on TFCC MRNet for LogisticRegression')
parser.add_argument('--root',
                    type=str,
                    default='./', # /home/tedwu430/tfcc/
                    help='The root path.')
parser.add_argument('--num_classes',
                    type=int,
                    default=2,
                    help='There are several categories of output.')
parser.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help='Batch size per GPU, it can only be 1.')


def extract_predictions(root, mode, plane, num_classes, batch_size):

    assert mode in ['train', 'val'] , 'mode should be train or val.'
    assert plane in ['ax', 'cor', 'sag'] , 'plane should be axial, coronal, sagittal.'

    # Set up
    if mode == 'train':
        tfcc_data = pd.read_csv(Path(root, f'excel/{mode}.csv'))
        tfcc_data_0 = tfcc_data[tfcc_data['LABEL']==0].sample(frac=5,
                                                              replace=True,
                                                              random_state=1)
        tfcc_data_1 = tfcc_data[tfcc_data['LABEL']==1].sample(frac=6.5,
                                                              replace=True,
                                                              random_state=1)
        tfcc_data = pd.concat([tfcc_data_0, tfcc_data_1])
        tfcc_path = Path(root, f'data/{plane}_t1')
    elif mode == 'val':
        tfcc_data = pd.read_csv(Path(root, f'excel/{mode}id.csv'))
        tfcc_path = Path(root, f'data/{plane}_t1')

    model_path = Path(root, f'model/{plane}/RangerLarsR/tfcc_prediction.pt')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    model = TFCCNet(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    
    model.eval()
    # Load data and augment
    dataloaders_dict = tfcc_dataloader(tfcc_data, tfcc_data, tfcc_path, tfcc_path, batch_size)
    
    # Run
    predictions = []
    labels = []
    with torch.no_grad():
        for image, label in dataloaders_dict[f'{mode}']:
            logit = model(image.to(device))
            prediction = torch.softmax(logit, dim = 1)
            predictions.append(prediction.cpu().numpy())
            labels.append(label.numpy())      
    print('How many to predict:', len(predictions))
    
    return predictions, labels

def train_valid_logreg(mode):
    results = {}
    logreg_model = ''

    for plane in ['cor', 'sag', 'ax']:
        predictions, labels = extract_predictions(args.root, mode, plane, args.num_classes, args.batch_size)
        results['labels'] = labels
        results[plane] = predictions
        
    X = np.zeros((len(predictions), 3))
    X[:, 0] = [i[0][1] for i in results['cor']]
    X[:, 1] = [i[0][1] for i in results['sag']]
    X[:, 2] = [i[0][1] for i in results['ax']]

    if mode == 'train':
        y = np.array(labels)
        logreg = LogisticRegression(solver='lbfgs')
        logreg_model = logreg.fit(X, y)

    return logreg_model, X

def show_save(valid_csv, logreg_image, model):

    logreg_model, _ = train_valid_logreg('train')
    _, x_val = train_valid_logreg('val')

    y_val = pd.read_csv(valid_csv)['LABEL']
    y_pred = logreg_model.predict_proba(x_val)[:, 1]
    print('AUC: ', roc_auc_score(y_val, y_pred))

    y_pred_class = logreg_model.predict(x_val)
    print('BACC: ', balanced_accuracy_score(y_val, y_pred_class))

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_val, y_pred_class))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='g')
    plt.savefig(logreg_image)
    plt.close('all')

    image_PIL = Image.open(logreg_image)
    img = np.array(image_PIL)
    image_writer.add_image('Valid Confusion Matrix', img, 3, dataformats='HWC')
    image_writer.close()

    joblib.dump(logreg_model, model)

if __name__ == '__main__':

    args = parser.parse_args()

    VALID_CSV = Path(args.root, CONFIG['data']['valid_excel'])

    IMAGE_DIR = Path(args.root, 'cm')
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LOGREG_IMAGE = Path(IMAGE_DIR, CONFIG['data']['confunsion_matrix_3'])

    LOG_IMG_DIR = Path(args.root, 'tb_tfcc/logreg')
    LOG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    image_writer = SummaryWriter(LOG_IMG_DIR)

    MODEL = Path(args.root, 'model/', CONFIG['data']['model_3'])

    show_save(VALID_CSV, LOGREG_IMAGE, MODEL)
