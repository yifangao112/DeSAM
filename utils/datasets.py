import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ProstateDataset(Dataset): 
    def __init__(self, data_root, patient_id, neg_points):
        self.data_root = data_root
        self.npz_files = sorted(patient_id)
        self.neg_points = neg_points
        self.image_scale = 4 # 1024 / 256
    
    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index):
        npz_data = np.load(os.path.join(self.data_root, self.npz_files[index]), allow_pickle=True)
        img_embed = npz_data['img_embeddings']
        img_embed = [torch.tensor(x).float() for x in img_embed]
        gt2D = npz_data['gts']        
        
        is_background = np.random.randint(0, self.neg_points+1)
        if is_background:
            # background point
            y_indices, x_indices = np.where(gt2D == 0)
            random_idx = np.random.randint(0, len(y_indices))
            prompt_points = np.array((
                x_indices[random_idx] * self.image_scale, 
                y_indices[random_idx] * self.image_scale
            ))
            iou_label = torch.tensor([0]).float()
            
        else:
            # foreground point
            y_indices, x_indices = np.where(gt2D > 0)
            random_idx = np.random.randint(0, len(y_indices))
            prompt_points = np.array((
                x_indices[random_idx] * self.image_scale, 
                y_indices[random_idx] * self.image_scale
            ))
            iou_label = torch.tensor([1]).float()
        
        in_points = torch.as_tensor(prompt_points)
        in_labels = 1
        # convert img embedding, mask, bounding box to torch tensor
        return img_embed, torch.tensor(gt2D[None, :,:]).float(), in_points, in_labels, iou_label