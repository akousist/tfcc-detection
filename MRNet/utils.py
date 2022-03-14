import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import TFCCNet

# Weight decay
def balance_weights(train_data, device):
    pos = np.sum(train_data['LABEL'])
    neg = len(train_data['LABEL'])-pos
    weights = [1, neg / pos]
    weights = torch.FloatTensor(weights).to(device)
    return weights
    
# Confusion Matrix and Classification Report
def cm_result(num_classes, model_path, dataloaders_dict, device, valid_data, valid_image, image_writer, page):
    
    y_pred_list = []
    y_pred_softmax_list = []
    model = TFCCNet(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        model.eval()
        for inputs, _ in dataloaders_dict['val']:
            inputs = inputs.to(device)
            y_test_pred = model(inputs)
            y_pred_softmax = torch.softmax(y_test_pred, dim = 1)
            y_pred_softmax_list.extend(y_pred_softmax)

            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            y_pred_tags = y_pred_tags.cpu().numpy()
            y_pred_list.extend(y_pred_tags)

    y_pred_softmax_list = [i.cpu().numpy()[1] for i in y_pred_softmax_list]
    y_val = valid_data['LABEL'].tolist()
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_val, y_pred_list))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='g')
    plt.savefig(valid_image)
    plt.close('all')

    image_PIL = Image.open(valid_image)
    img = np.array(image_PIL)
    image_writer.add_image("Valid Confusion Matrix", img, page, dataformats='HWC')
    image_writer.close()

    print(classification_report(y_val, y_pred_list))