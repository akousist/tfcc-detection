import joblib
import argparse
import configparser
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

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


def extract_predictions(root, plane, num_classes, batch_size):

    assert plane in ['ax', 'cor', 'sag'] , 'plane should be axial, coronal, sagittal.'

    # Set up
    tfcc_data = pd.read_csv(Path(root, f'excel/valid.csv'))
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
        for image, label in dataloaders_dict['val']:
            logit = model(image.to(device))
            prediction = torch.softmax(logit, dim = 1)
            predictions.append(prediction.cpu().numpy())
            labels.append(label.numpy())      
    print('How many to predict:', len(predictions))
    
    return predictions, labels

def train_valid_logreg():
    results = {}

    for plane in ['cor', 'sag', 'ax']:
        predictions, labels = extract_predictions(args.root, plane, args.num_classes, args.batch_size)
        results['labels'] = labels
        results[plane] = predictions
        
    X = np.zeros((len(predictions), 3))
    X[:, 0] = [i[0][1] for i in results['cor']]
    X[:, 1] = [i[0][1] for i in results['sag']]
    X[:, 2] = [i[0][1] for i in results['ax']]

    return X

def show_save(valid_csv):

    logreg_model = joblib.load('./model/tfcc_logreg.dat')
    x_val = train_valid_logreg()

    y_val = pd.read_csv(valid_csv)['LABEL']
    y_pred = logreg_model.predict_proba(x_val)[:, 1]
    y_pred_label = logreg_model.predict(x_val)
    y_true = y_val.values.tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
    
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print('Accuracy:', accuracy)
    print('Precision:', precision) # 陽性的樣本中有幾個是預測正確的
    print('Recall:', recall) # 事實為真的樣本中有幾個是預測正確的
    print('F1:', 2 / ((1 / precision) + (1 / recall)))
    print('AUC:', roc_auc_score(y_val, y_pred))
    print('BACC:', balanced_accuracy_score(y_val, y_pred_label))
    
    np.savez('./data/MRNet_result.npz', y_pred=y_pred, y_true=y_true)



if __name__ == '__main__':

    args = parser.parse_args()

    VALID_CSV = Path(args.root, CONFIG['data']['valid_excel'])

    show_save(VALID_CSV)
