# some code from nnU-Net: https://github.com/MIC-DKFZ/nnUNet
from time import time, sleep
from datetime import datetime
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Logger(): 
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        timestamp = datetime.now()
        self.log_file = os.path.join(
            self.output_folder, 
            "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
            (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
            timestamp.second)
        )
        with open(self.log_file, 'w') as f:
            f.write("Starting... \n")
        
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)
        
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)
            
            
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def split_prostatedataset(data_root, patientid_path, center=1, seed=0, random_validation=False):
    all_centers = ['A-ISBI', 'B-ISBI_1.5', 'C-I2CVB', 'D-UCL', 'E-BIDMC', 'F-HK']
    train_centers = [all_centers[center-1]]
    patientid = pd.read_csv(patientid_path)
    all_patientid = []
    train_patientid = []
    val_patientid = []
    for c in train_centers:
        all_patientid.extend(patientid[patientid.center==c]['patientid'].values.tolist()) 

    if random_validation:
        np.random.seed(seed)
        np.random.shuffle(all_patientid)
        train_patientid = sorted(all_patientid[:int(len(all_patientid)*0.9)])
        val_patientid = sorted(all_patientid[int(len(all_patientid)*0.9):])
    else:
        train_patientid = all_patientid[:int(len(all_patientid)*0.9)]
        val_patientid = all_patientid[int(len(all_patientid)*0.9):]

    train_patientid = [x for x in os.listdir(data_root) if int(x[9:12]) in train_patientid]
    val_patientid = [x for x in os.listdir(data_root) if int(x[9:12]) in val_patientid]

    # exclude centers
    ood_centers = list(set(all_centers) - set(train_centers))
    test_patientid = []
    for c in ood_centers:
        test_patientid.extend(patientid[patientid.center==c]['patientid'].values.tolist())
        
    return train_patientid, val_patientid, test_patientid


def plot_loss(train_losses, val_losses, model_save_path):
    # plot loss
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Dice CE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(model_save_path, 'train_loss.png'))
    plt.close()

