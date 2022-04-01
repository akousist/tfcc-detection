import csv
import argparse
import configparser
import shutil
import pandas as pd
import pydicom

from pathlib import Path

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

def get_procedure_args():
    parser = argparse.ArgumentParser(description='Data process procedure')
    parser.add_argument('--remove_dir',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to remove redundant directories.')
    parser.add_argument('--extract_info',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to extract information in dicom and save as csv file.')
    args = parser.parse_args()
    return args

def remove_redundant_dir():
    target_csv = pd.read_csv(CONFIG['data']['target_csv'])
    target_directories = target_csv['serial_number'].tolist()
    data_directory = Path(CONFIG['data']['data_directory'])
    for directory in data_directory.iterdir():
        directory_name = directory.name
        if directory_name not in target_directories:
            shutil.rmtree(directory, ignore_errors=True)

def write_csv(data, csv_path, column_name):
    with open(csv_path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(column_name)
        for row in data:
            writer.writerow(row)

def extract_info(target_dir):
    def collect_dicom_info(ds, attrs=None):
        dicom_info = []
        for attr in attrs:
            item = ds.get(attr)
            dicom_info.append(item)
        return dicom_info
    
    attrs = ['Modality', 'SeriesDescription', 'InStackPositionNumber', 'SliceLocation', 'SliceThickness']
    dicom_infos = []
    for target_file in target_dir.iterdir():
        if str(target_file).endswith('.dcm'):
            ds = pydicom.dcmread(target_file)
            dicom_info = collect_dicom_info(ds, attrs)
            dicom_infos.append([target_file.name] + dicom_info)
    
    column_name = ['Files'] + attrs
    csv_path = target_dir.joinpath(target_dir.name + '.csv')
    write_csv(dicom_infos, csv_path, column_name)

def get_csv():
    data_directory = Path(CONFIG['data']['data_directory'])
    for directory in data_directory.iterdir():
        if directory.is_dir():
            extract_info(directory)

if __name__ == '__main__':
    args = get_procedure_args()

    if args.remove_dir:
        remove_redundant_dir()
    if args.extract_info:
        get_csv()