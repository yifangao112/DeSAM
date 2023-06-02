"""
code is from https://github.com/bowang-lab/MedSAM/blob/main/pre_MR.py.
"""
import numpy as np
import SimpleITK as sitk
import os
join = os.path.join 
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from desam import sam_model_registry
from desam.utils.transforms import ResizeLongestSide
import argparse

# set up the parser
parser = argparse.ArgumentParser(description='preprocess non-CT images')
parser.add_argument('-w', '--work_dir', type=str, default='E:/DeSAMData', help='path to the work dir')
parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--img_name_suffix', type=str, default='_0000.nii.gz', help='image name suffix')
parser.add_argument('--label_id', type=int, default=1, help='label id')
parser.add_argument('--model_type', type=str, default='vit_h', help='model type')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
# seed
parser.add_argument('--seed', type=int, default=2023, help='random seed')
args = parser.parse_args()


work_dir = args.work_dir
nii_path = join(work_dir, 'raw_data', 'imagesTr')
gt_path = join(work_dir, 'raw_data', 'labelsTr')
npz_path = join(work_dir, 'image_embeddings', 'npz_files_{}'.format(args.model_type))
png_path = join(work_dir, 'image_embeddings', 'png_files_{}'.format(args.model_type))

os.makedirs(npz_path, exist_ok=True)
os.makedirs(png_path, exist_ok=True)

model_zoo = {
    'vit_h': os.path.join(work_dir, 'checkpoint/sam_vit_h_4b8939.pth'),
    'vit_l': os.path.join(work_dir, 'checkpoint/sam_vit_l_0b3195.pth'),
    'vit_b': os.path.join(work_dir, 'checkpoint/sam_vit_b_01ec64.pth'),
}
checkpoint = model_zoo[args.model_type]

names = sorted(os.listdir(gt_path))
names = [name for name in names if os.path.exists(join(nii_path, name.split('.nii.gz')[0] + args.img_name_suffix))]

# split names into training and testing
np.random.seed(args.seed)
np.random.shuffle(names)
train_names = sorted(names)

# def preprocessing function
def preprocess_nonct(gt_path, nii_path, gt_name, image_name, label_id, image_size, sam_model, device):
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    # gt_data = np.uint8(gt_data==label_id)
    gt_data = np.uint8(gt_data==label_id)

    if np.sum(gt_data)>0:
        imgs = []
        gts =  []
        img_embeddings = []
        assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = np.uint8(image_data_pre)
        
        z_index, _, _ = np.where(gt_data>0)
        z_min, z_max = np.min(z_index), np.max(z_index)
        
        for i in range(z_min, z_max):
            gt_slice_i = gt_data[i,:,:]
            gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
            if np.sum(gt_slice_i)>0:
                # resize img_slice_i to 256x256
                img_slice_i = transform.resize(image_data_pre[i,:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
                # convert to three channels
                img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
                assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
                assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
                imgs.append(img_slice_i)
                gts.append(gt_slice_i)
                if sam_model is not None:
                    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                    resize_img = sam_transform.apply_image(img_slice_i)
                    # resized_shapes.append(resize_img.shape[:2])
                    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                    # model input: (1, 3, 1024, 1024)
                    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
                    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                    # input_imgs.append(input_image.cpu().numpy()[0])
                    with torch.no_grad():
                        embedding = sam_model.image_encoder(input_image)
                        img_embeddings.append([x.cpu().numpy()[0] for x in embedding])
                        # img_embeddings.append(embedding[-1][0].cpu().numpy())

    if sam_model is not None:
        return imgs, gts, img_embeddings
    else:
        return imgs, gts


# prepare the save path
save_path_tr = npz_path
os.makedirs(save_path_tr, exist_ok=True)

# set up the model
sam_model = sam_model_registry[args.model_type](checkpoint=checkpoint).to(args.device)

for name in tqdm(train_names):
    image_name = name.split('.nii.gz')[0] + args.img_name_suffix
    gt_name = name 
    imgs, gts, img_embeddings = preprocess_nonct(gt_path, nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model, args.device)
    # save to npz file
    # stack the list to array
    for idx in range(len(imgs)):
        # imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
        # gts = np.stack(gts, axis=0) # (n, 256, 256)
        # img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
        np.savez_compressed(
            join(save_path_tr, gt_name.split('.nii.gz')[0]+'_'+str(idx).zfill(2)+'.npz'), 
            imgs=imgs[idx], 
            gts=gts[idx], 
            img_embeddings=img_embeddings[idx]
        )
        # save an example image for sanity check
        img_idx = imgs[idx]
        gt_idx = gts[idx]
        bd = segmentation.find_boundaries(gt_idx, mode='inner')
        img_idx[bd, :] = [255, 0, 0]
        io.imsave(join(png_path, gt_name.split('.nii.gz')[0]+'_'+str(idx).zfill(2)+'.png'), img_idx, check_contrast=False)