# DeSAM 
This is the official repository for DeSAM: Decoupling Segment Anything Model for Generalizable Medical Image Segmentation.

![Teaser image](./DeSAM.png)

> **[DeSAM: Decoupling Segment Anything Model for Generalizable Medical Image Segmentation](https://arxiv.org/abs/2306.00499)**<br>
> Yifan Gao, Wei Xia, Dingdu Hu and Xin Gao<br>
> *arxiv preprint, 2023*

## Abstract
Deep learning based automatic medical image segmentation models often suffer from domain shift, where the models trained on a source domain do not generalize well to other unseen domains. As a vision foundation model with powerful generalization capabilities, Segment Anything Model (SAM) shows potential for improving the cross-domain robustness of medical image segmentation. However, SAM and its finetuned models performed significantly worse in fully automatic mode compared to when given manual prompts. Upon further investigation, we discovered that the degradation in performance was related to the coupling effect of poor prompts and mask segmentation. In fully automatic mode, the presence of inevitable poor prompts (such as points outside the mask or boxes significantly larger than the mask) can significantly mislead mask generation. To address the coupling effect, we propose the decoupling SAM (DeSAM). DeSAM modifies SAM’s mask decoder to decouple mask generation and prompt embeddings while leveraging pretrained weights. We conducted experiments on publicly available prostate cross-site datasets. The results show that DeSAM improves dice score by an average of 8.96% (from 70.06% to 79.02%) compared to previous state-of-the-art domain generalization method. Moreover, DeSAM can be trained on personal devices with entry-level GPU since our approach does not rely on tuning the heavyweight image encoder.

## Training DeSAM on cross-site prostate dataset

### Installation 
1. Create a virtual environment `conda create -n desam python=3.10 -y` and activate it `conda activate desam`
2. Install [Pytorch](https://pytorch.org/get-started/locally/)
3.  `git clone https://github.com/yifangao112/DeSAM`
4. Enter the DeSAM folder `cd DeSAM` and run `pip install -r requirements.txt`

### Data preparation and preprocessing

Our files are organized as follows, similar to nnU-Net:
- work_dir
    - raw_data
    - checkpoint
    - image_embeddings
    - results_folder

1. Download the cross-site prostate dataset [Google Drive](https://drive.google.com/file/d/1purk1f96GUGjAuVg_qCg_lWonIRUsdhw/view?usp=sharing), unzip it and put files under the `work_dir/raw_data` dir. The data also host on [Baidu Netdisk](https://pan.baidu.com/s/1Lq3uz3T2bq44-kYodsGjzg), password: dsam. The original pre-processing data was downloaded from [MaxStyle](https://github.com/cherise215/MaxStyle), many thanks!

2. Download [SAM ViT-H checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place it at `work_dir/checkpoint/sam_vit_h_4b8939.pth`.

3. Precompute image embeddings (~90G, Make sure your work_dir is on SSD):

```bash
python precompute_embeddings.py --work_dir your_work_dir
```

### Train the model

#### whole box mode:

```bash
python desam_train_wholebox.py --work_dir your_work_dir --center=1 --pred_embedding=True --mixprecision=True
```

#### grid Points mode (segment-anything mode):

```bash
python desam_train_gridpoints.py --work_dir your_work_dir --center=1 --pred_embedding=True --mixprecision=True
```

## Acknowledgements
This repository is based on [MedSAM](https://github.com/bowang-lab/MedSAM). We thank Jun Ma for making the source code of MedSAM publicly available. Part of codes are reused from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

## Citation
If this code is helpful for your study, please cite our [paper]():

```
@article{gao2023desam,
  title={DeSAM: Decoupling Segment Anything Model for Generalizable Medical Image Segmentation},
  author={Yifan Gao and Wei Xia and Dingdu Hu and Xin Gao},
  journal={arXiv preprint arXiv:2306.00499},
  year={2023}
}
```

