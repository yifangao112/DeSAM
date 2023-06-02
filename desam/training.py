import torch
from medpy import metric
import numpy as np
import os
from utils.visualization import show_anns, show_mask
from utils.dice_calculate import dice_calculate
import gc
from medpy import metric
from desam import SamAutomaticMaskGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from utils.utils import poly_lr


def run_training(
        model, max_num_epochs, logger, model_save_path, optimizer, initial_lr,
        train_dataloader, val_dataloader, scaler, dicece_loss, iou_loss, 
        device='cuda', is_mixprecision=False, 
    ):
    train_losses = []
    val_losses = []
    best_loss = 1e10

    for epoch in range(max_num_epochs):
        # update lr
        model.train()
        logger.print_to_log_file("\nepoch: ", epoch)
        epoch_start_time = time()
        optimizer.param_groups[0]['lr'] = poly_lr(epoch, max_num_epochs, initial_lr, 0.9)
        
        epoch_diceceloss = 0
        epoch_iouloss = 0
        # train
        for step, (image_embedding, gt, in_points, in_labels, iou_label) in enumerate(train_dataloader):
            # do not compute gradients for prompt encoder
            with torch.no_grad():
                points_torch = (in_points[:, None, :].to(device), in_labels[:, None].to(device))
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points_torch,
                    boxes=None,
                    masks=None,
                )

            # predicted masks
            with torch.cuda.amp.autocast(enabled=is_mixprecision):
                image_embedding = [x.to(device) for x in image_embedding]
                mask_pred, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embedding, # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                    multimask_output=False,
                )
                    
                diceceloss = dicece_loss(mask_pred, gt.to(device))                
                iouloss = iou_loss(iou_predictions, iou_label.to(device))
                loss = diceceloss + iouloss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_diceceloss += diceceloss.item()
            epoch_iouloss += iouloss.item()
        
        epoch_diceceloss /= step
        epoch_iouloss /= step
        train_epoch_loss = epoch_diceceloss
        train_losses.append(epoch_diceceloss)
        
        val_epoch_loss, val_epoch_dice = run_validation(
            model, val_dataloader, device, dicece_loss
        )
        val_losses.append(val_epoch_loss)

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'desam_model_best.pth'))
        
        logger.print_to_log_file("train loss : %.4f" % train_epoch_loss)
        logger.print_to_log_file("validation loss: %.4f" % val_epoch_loss)
        logger.print_to_log_file("Average global foreground Dice: %.4f" % val_epoch_dice)
        logger.print_to_log_file(
            "(interpret this as an estimate dice of the iid validation set. This is not exact.)"
        )
        logger.print_to_log_file("lr:", np.round(optimizer.param_groups[0]['lr'], decimals=8))
        epoch_end_time = time()
        logger.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

    return train_losses, val_losses


def run_training_wholebox(
        model, max_num_epochs, logger, model_save_path, optimizer, initial_lr,
        train_dataloader, val_dataloader, scaler, dicece_loss, iou_loss, 
        device='cuda', is_mixprecision=False, 
    ):
    train_losses = []
    val_losses = []
    best_loss = 1e10

    for epoch in range(max_num_epochs):
        # update lr
        model.train()
        logger.print_to_log_file("\nepoch: ", epoch)
        epoch_start_time = time()
        optimizer.param_groups[0]['lr'] = poly_lr(epoch, max_num_epochs, initial_lr, 0.9)
        
        epoch_diceceloss = 0
        # train
        for step, (image_embedding, gt, in_points, in_labels, iou_label) in enumerate(train_dataloader):
            
            with torch.no_grad():
                B,_, H, W = gt.shape
                boxes_torch = torch.from_numpy(np.array([[0, 0, 1024, 1024]]*B)).float().to(device)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=boxes_torch,
                    masks=None,
                )

            # predicted masks
            with torch.cuda.amp.autocast(enabled=is_mixprecision):
                image_embedding = [x.to(device) for x in image_embedding]
                mask_pred, _ = model.mask_decoder(
                    image_embeddings=image_embedding, # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                    multimask_output=False,
                )
                    
                diceceloss = dicece_loss(mask_pred, gt.to(device))                
                loss = diceceloss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_diceceloss += diceceloss.item()
        
        epoch_diceceloss /= step
        train_epoch_loss = epoch_diceceloss
        train_losses.append(epoch_diceceloss)
        
        val_epoch_loss, val_epoch_dice = run_validation_wholebox(
            model, val_dataloader, device, dicece_loss
        )
        val_losses.append(val_epoch_loss)

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'desam_model_best.pth'))
        
        logger.print_to_log_file("train loss : %.4f" % train_epoch_loss)
        logger.print_to_log_file("validation loss: %.4f" % val_epoch_loss)
        logger.print_to_log_file("Average global foreground Dice: %.4f" % val_epoch_dice)
        logger.print_to_log_file(
            "(interpret this as an estimate dice of the iid validation set. This is not exact.)"
        )
        logger.print_to_log_file("lr:", np.round(optimizer.param_groups[0]['lr'], decimals=8))
        epoch_end_time = time()
        logger.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

    return train_losses, val_losses


def run_validation(model, val_dataloader, device, dicece_loss):
    model.eval()
    val_epoch_diceceloss = 0
    dice_list = []
    for step, (image_embedding, gt, in_points, in_labels, _) in enumerate(val_dataloader):
        with torch.no_grad():
            # do not compute gradients
            points_torch = (in_points[:, None, :].to(device), in_labels[:, None].to(device))
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points_torch,
                boxes=None,
                masks=None,
            )
            # predicted masks
            image_embedding = [x.to(device) for x in image_embedding]
            mask_pred, _ = model.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
                
            diceceloss = dicece_loss(mask_pred, gt.to(device))
            val_epoch_diceceloss += diceceloss.item()
            
            mask_pred = torch.sigmoid(mask_pred)
            mask_pred = mask_pred.cpu().numpy().squeeze()
            mask_pred = (mask_pred > 0.5).astype(np.uint8)
                        
            dice_list.append(metric.dc(mask_pred, gt.cpu().numpy().squeeze()))
    
    val_epoch_diceceloss /= step
    val_epoch_loss = val_epoch_diceceloss
        
    return val_epoch_loss, np.mean(dice_list)


def run_validation_wholebox(model, val_dataloader, device, dicece_loss):
    model.eval()
    val_epoch_diceceloss = 0
    dice_list = []
    for step, (image_embedding, gt, in_points, in_labels, _) in enumerate(val_dataloader):
        with torch.no_grad():
            # do not compute gradients
            box_torch = torch.from_numpy(np.array([[0, 0, 1024, 1024]])).float().to(device)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            # predicted masks
            image_embedding = [x.to(device) for x in image_embedding]
            mask_pred, _ = model.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
                
            diceceloss = dicece_loss(mask_pred, gt.to(device))
            val_epoch_diceceloss += diceceloss.item()
            
            mask_pred = torch.sigmoid(mask_pred)
            mask_pred = mask_pred.cpu().numpy().squeeze()
            mask_pred = (mask_pred > 0.5).astype(np.uint8)
                        
            dice_list.append(metric.dc(mask_pred, gt.cpu().numpy().squeeze()))
    
    val_epoch_diceceloss /= step
    val_epoch_loss = val_epoch_diceceloss
        
    return val_epoch_loss, np.mean(dice_list)


def run_testing(
        model, data_root, model_save_path, iou_thresh, ood_patientid,
        grid, logger, pred_embedding=True, device='cuda'
    ):
    
    model.eval()
            
    save_fig = os.path.join(model_save_path, 'savefig_iou{}'.format(iou_thresh))
    save_mask = os.path.join(model_save_path, 'save_mask')
    os.makedirs(save_fig, exist_ok=True)
    os.makedirs(save_mask, exist_ok=True)
    logger.print_to_log_file('now pred_iou_thresh: {}'.format(iou_thresh))
                    
    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=grid,
        points_per_batch=grid,
        pred_iou_thresh=iou_thresh,
        stability_score_thresh=0.6,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=1000,
    )

    plt.figure(figsize=(10, 10))
    ood_patientid_list = [x for x in os.listdir(data_root) if int(x[9:12]) in ood_patientid]
    for patientid in tqdm(ood_patientid_list):                        
        npz_data = np.load(os.path.join(data_root, patientid), allow_pickle=True)
        image = npz_data['imgs']
        mask_gt = npz_data['gts']
        
        if pred_embedding:
            image_embedding = npz_data['img_embeddings']
            image_embedding = [torch.tensor(x)[None].float().to(device) for x in image_embedding]
        else:
            image_embedding = None
            
        masks, _ = mask_generator.generate(image, image_embedding)
        
        plt.imshow(image)
        show_mask(mask_gt)
        show_anns(masks, os.path.join(save_mask, patientid), colors=1)
        # show_points(points_output)
        plt.axis('off')
        plt.savefig(os.path.join(save_fig, patientid + '.jpg'))
        plt.clf()
        gc.collect()
                
    dice = dice_calculate(data_root, save_mask, ood_patientid, edge_set_zero=50)
    logger.print_to_log_file('pred_iou_thresh: {}, mean_dice: {}'.format(iou_thresh, dice))
    
    
def run_testing_wholebox(
        model, data_root, model_save_path, iou_thresh, ood_patientid,
        grid, logger, pred_embedding=True, device='cuda'
    ):
    
    model.eval()
            
    save_fig = os.path.join(model_save_path, 'savefig_iou{}'.format(iou_thresh))
    save_mask = os.path.join(model_save_path, 'save_mask')
    os.makedirs(save_fig, exist_ok=True)
    os.makedirs(save_mask, exist_ok=True)
    logger.print_to_log_file('now pred_iou_thresh: {}'.format(iou_thresh))
                    
    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=grid,
        points_per_batch=grid,
        pred_iou_thresh=iou_thresh,
        stability_score_thresh=0.6,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=1000,
    )

    plt.figure(figsize=(10, 10))
    ood_patientid_list = [x for x in os.listdir(data_root) if int(x[9:12]) in ood_patientid]
    for patientid in tqdm(ood_patientid_list):                        
        npz_data = np.load(os.path.join(data_root, patientid), allow_pickle=True)
        image = npz_data['imgs']
        mask_gt = npz_data['gts']
        
        if pred_embedding:
            image_embedding = npz_data['img_embeddings']
            image_embedding = [torch.tensor(x)[None].float().to(device) for x in image_embedding]
        else:
            image_embedding = None
            
        masks, _ = mask_generator.generate_box(image, image_embedding)
        
        plt.imshow(image)
        show_mask(mask_gt)
        show_anns(masks, os.path.join(save_mask, patientid), colors=1)
        # show_points(points_output)
        plt.axis('off')
        plt.savefig(os.path.join(save_fig, patientid + '.jpg'))
        plt.clf()
        gc.collect()
                
    dice = dice_calculate(data_root, save_mask, ood_patientid, edge_set_zero=50)
    logger.print_to_log_file('pred_iou_thresh: {}, mean_dice: {}'.format(iou_thresh, dice))