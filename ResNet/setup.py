import json
import random
import shutil
import dicom2nifti
import pydicom
import numpy as np
import pandas as pd

from pathlib import Path
from utils import extractDICOM, write_csv
from tqdm import tqdm
from utils import get_logger
from args import get_setup_args

def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
    with open(filename, "w") as fh:
        json.dump(obj, fh)

def save_npz(file_path, data, message=None):
    if message is not None:
        print(f'Saving {message} npz...')
    if data.size == 0:
        return
    elif len(data.shape) == 1:
        data = data[np.newaxis, :]
    np.savez(file_path,
             files=data[:, 0],
             labels=data[:, 1].astype(np.int64),
             idxs=np.arange(1, len(data)+1))

def create_directories(output_directory, planes, mri_types):
    for plane in planes:
        for mri_type in mri_types:
            output_subdir = output_directory / f'{plane}_{mri_type}'
            output_subdir.mkdir(exist_ok=True)

def get_stacked_dicom(df, plane, mri_type, stack_number):
    """Get DICOM file list in each of SeriesDescription"""
    mask = df['SeriesDescription'].apply(lambda x: plane in x.lower() and mri_type in x.lower())
    df_sd = df[mask]
    df_sd.drop_duplicates(subset='SliceLocation', inplace=True)
    if stack_number:
        position_num_center = (len(df_sd) - 1) // 2
        position_num_min = int(position_num_center - stack_number / 2)
        position_num_max = int(position_num_center + stack_number / 2)
        df_sd = df_sd.iloc[position_num_min:position_num_max]
    
    return df_sd['Files'].tolist()

def stack_image(input_dir, dicom_list):
    image_stacked = []
    for dicom in dicom_list:
        ds = pydicom.dcmread(input_dir / dicom)
        image = ds.pixel_array
        image_scaled = np.uint8(image / image.max() * 255)
        image_stacked.append(image_scaled)
    image_stacked = np.stack(image_stacked, axis=-1)
    return image_stacked

def convert_dicom(input_dir, output_directory, stack_number, log):
    planes = ('ax', 'cor', 'sag')
    mri_types = ('t1', 't2')

    input_directory = Path(input_dir)
    total_dicom = len(list(input_directory.iterdir()))
    create_directories(output_directory, planes, mri_types)
    
    valid_data = []
    for directory in tqdm(input_directory.iterdir(), total=total_dicom):
        csv_file = directory / (directory.name + '.csv')
        df = pd.read_csv(csv_file)

        try:
            for plane in planes:
                for mri_type in mri_types:
                    dicom_list = get_stacked_dicom(df, plane, mri_type, stack_number)
                    image_stacked = stack_image(directory, dicom_list)
                    np.save(output_directory / f"{plane}_{mri_type}" / f"{directory.name}.npy", image_stacked)
            
            valid_data.append(directory.name)
            log.info(f"Conversion: {directory.name} finished")
        except:
            log.info(f"Conversion: {directory.name} failed")
    np.save('valid_data.npy', np.array(valid_data))

def pre_process(args):
    if args.convert_dicom:
        output_directory = Path('./data')
        output_directory.mkdir(exist_ok=True)
        log = get_logger(output_directory, 'setup')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        convert_dicom(args.input_dir, output_directory, args.stack_number, log)

    df = pd.read_csv(args.data_file)
    files, labels = df['serial_number'].tolist(), df['label'].tolist()
    valid_data = np.load('valid_data.npy')

    # Set random seed
    random.seed(args.seed)
    train_data, dev_data, test_data = [], [], []
    for data in zip(files, labels):
        if data[0] not in valid_data:
            continue

        rand_num = random.random()
        if rand_num <= args.train_split:
            train_data.append(data)
        elif args.train_split < rand_num <= args.train_split + args.test_split:
            test_data.append(data)
        else:
            dev_data.append(data)
    
    train_data = np.array(train_data)
    dev_data = np.array(dev_data)
    test_data = np.array(test_data)

    save_npz(args.train_record_file, train_data, message='train')
    save_npz(args.dev_record_file, dev_data, message='dev')
    save_npz(args.test_record_file, test_data, message='test')

    save(args.train_meta_file, {'total': len(train_data)}, message='train meta')
    save(args.dev_meta_file, {'total': len(dev_data)}, message='dev meta')
    save(args.test_meta_file, {'total': len(test_data)}, message='test meta')

if __name__ == '__main__':
    pre_process(get_setup_args())
    