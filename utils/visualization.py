import matplotlib.pyplot as plt
import numpy as np


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        color = np.array([30, 144, 255, 0.4*255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax = plt.gca()
    ax.imshow(mask_image.astype(np.uint8))
    
    
def show_anns(anns, save_mask_path, colors=0.35):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    np.savez_compressed(save_mask_path, anns)
    if len(anns) == 1:
        mask = anns[0]['segmentation']
        h, w = mask.shape[-2:]
        color = np.array([255, 165, 0, 0.6*255])
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax = plt.gca()
        ax.imshow(mask_image.astype(np.uint8))
    else:
        for ann in anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, m*colors)))
            # plt.imshow(np.dstack((img, m*colors)))
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 