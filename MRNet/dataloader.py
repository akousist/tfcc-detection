import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data_processing import TFCCDataset
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip


def tfcc_dataloader(train_data, valid_data, train_path, valid_path, batch_size):
    
    # Load data and augment
    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            RandomRotate(25),
            RandomTranslate([0.11, 0.11]),
            RandomFlip(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
        ]),
        'val': transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {
        'train': TFCCDataset(
            train_data, train_path, data_transforms['train']),
        'val': TFCCDataset(
            valid_data, valid_path, data_transforms['val'])
    }
    # Dataloader iterators
    dataloaders_dict = {
        'train': DataLoader(
            dataset = image_datasets['train'], batch_size = batch_size,
            shuffle=True, num_workers=4),
        'val': DataLoader(
            dataset = image_datasets['val'], batch_size = batch_size,
            shuffle=False, num_workers=4)
    }

    return dataloaders_dict