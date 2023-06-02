import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from medpy import metric


def dice_calculate(gt_root, mask_proposals_root, test_patientid, edge_set_zero=0):
    dice_list = []
    for patientid in tqdm(test_patientid):
        patientid_path_list = [x for x in os.listdir(gt_root) if '_{}_'.format(str(patientid).zfill(3)) in x]
        patientid_path_list = sorted(patientid_path_list)        
        gt_numpy = []
        mask_numpy = []
        for patientid_path in patientid_path_list:
            
            gt_data = np.load(
                os.path.join(gt_root, patientid_path), 
                allow_pickle=True
            )['gts']
                        
            if not os.path.exists(os.path.join(mask_proposals_root, patientid_path)):
                if os.path.exists(os.path.join(mask_proposals_root, patientid_path+'.npz')):
                    mask_proposals = np.load(
                        os.path.join(mask_proposals_root, patientid_path+'.npz'), 
                        allow_pickle=True
                    )['arr_0'][0]['segmentation']
                else:
                    mask_proposals = np.zeros_like(gt_data)
            else:
                try:
                    mask_proposals = np.load(
                        os.path.join(mask_proposals_root, patientid_path), 
                        allow_pickle=True
                    )['arr_0'][0]['segmentation']
                except:
                    mask_proposals = np.load(
                        os.path.join(mask_proposals_root, patientid_path), 
                        allow_pickle=True
                    )['data']
                    
            mask_proposals[:edge_set_zero, ...] = 0
            mask_proposals[mask_proposals.shape[0]-edge_set_zero:, ...] = 0
            mask_proposals[..., :edge_set_zero] = 0
            mask_proposals[..., mask_proposals.shape[0]-edge_set_zero:] = 0
                            
            gt_numpy.append(gt_data)
            mask_numpy.append(mask_proposals.astype(np.uint8))
            
        gt_numpy = np.array(gt_numpy)
        mask_numpy = np.array(mask_numpy)
        
        dice = metric.dc(mask_numpy, gt_numpy)
        dice_list.append(dice)
        
    return np.mean(dice_list)