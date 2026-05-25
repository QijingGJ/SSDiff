"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.append("../")
sys.path.append("./")
import random
import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from C2L_dataloader import Cine2LGEDataset, PublicDataset, ZSDataset
import torchvision.utils as vutils
import torch.utils.data as data

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError as MAE
from torchmetrics.regression import MeanSquaredError as MSE


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating test dataloader...")

    # =============================================================================================================
    test_dataset = Cine2LGEDataset(
        '/mnt/data_2/qijingothers2/AMI_Multitask_lynx/data_excel/data_PLA_Lynx_test.csv',
        '/mnt/data_2/qijingothers2/AMI_Multitask/data_image/', False, img_size=args.image_size)

    print('The number of test images = %d' % len(test_dataset))
    datal = th.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=8, pin_memory=True)
    data = iter(datal)
    # =============================================================================================================

    caiyang_tag = 'dpm'
    data_name = 'AMI_PLA_dpm_final'
    folder = 'sample_{}'.format(data_name)

    if not os.path.exists('/mnt/data_2/qijingothers2/AMI_Multitask_lynx/result_sample/' + folder):
        os.makedirs('/mnt/data_2/qijingothers2/AMI_Multitask_lynx/result_sample/' + folder)
    # =============================================================================================================

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("sampling...")

    ms_ssim_metric = MS_SSIM(data_range=1.0, kernel_size=7, betas=(0.0448, 0.2856, 0.3001)).to(dist_util.dev())
    ssim_metric = SSIM(data_range=1.0).to(dist_util.dev())
    psnr_metric = PSNR(data_range=1.0).to(dist_util.dev())
    mse_metric = MSE().to(dist_util.dev())
    mae_metric = MAE().to(dist_util.dev())

    result = []
    for i in range(len(data)):
        logger.log("sampling for {}".format(i + 1))

        pack = next(data)
        cine_right = pack['cine_right']
        lge = pack['lge']
        t2 = pack['t2']
        seg_mask = pack['seg_mask']

        label = pack['label']
        slice_ID = pack['npz_name'][0].split('.')[0]
        print("{} Index: ".format(slice_ID))

        origin_cine = cine_right
        origin_lge = lge
        origin_t2 = t2
        model_kwargs = {'y': label.to(dist_util.dev())}

        if caiyang_tag == 'dpm':
            torch.manual_seed(2025)
        c1 = th.randn_like(lge)
        c2 = th.randn_like(seg_mask)
        img = th.cat((cine_right, t2, c1, c2), dim=1).float().to(dist_util.dev())

        sample_fn = None
        if caiyang_tag == 'ddpm':
            sample_fn = diffusion.p_sample_loop
        elif caiyang_tag == 'ddim':
            sample_fn = diffusion.ddim_sample_loop
        elif caiyang_tag == 'dpm':
            sample_fn = diffusion.dpm_solver_sampling
        else:
            assert 'no method!'
        sample = sample_fn(model, (args.batch_size, 1, args.image_size, args.image_size), img, pack,  # slice_ID,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs,
                           )  # (1,1,128,128)

        sample_lge = sample[:, -2:-1, ...]
        sample_msk = sample[:, -1:, ...]

        sample_lge = torch.clip(sample_lge, -1, 1)
        sample_msk = torch.clip(sample_msk, -1, 1)

        # -------------------------------
        tensor_xx = (sample_lge + 1) / 2
        tensor_yy = (origin_lge + 1) / 2
        # -------------------------------

        sample_msk = (sample_msk + 1) / 2
        pred_lge = tensor_xx.repeat(1, 3, 1, 1)
        pred_msk = sample_msk.repeat(1, 3, 1, 1)

        origin_lge = ((origin_lge + 1) / 2).repeat(1, 3, 1, 1)
        origin_cine = ((origin_cine + 1.0) / 2).repeat(1, 3, 1, 1)
        origin_t2 = ((origin_t2 + 1.0) / 2).repeat(1, 3, 1, 1)
        origin_mask = ((seg_mask + 1.0) / 2).repeat(1, 3, 1, 1)

        # 指标计算--------------------------------------------------------
        batch_mae = mae_metric(tensor_xx, tensor_yy.to(dist_util.dev()))
        batch_mse = mse_metric(tensor_xx, tensor_yy.to(dist_util.dev()))
        batch_ms_ssim = ms_ssim_metric(tensor_xx, tensor_yy.to(dist_util.dev()))
        batch_ssim = ssim_metric(tensor_xx, tensor_yy.to(dist_util.dev()))
        batch_psnr = psnr_metric(tensor_xx, tensor_yy.to(dist_util.dev()))

        index_mae = batch_mae.cpu().numpy()
        index_mse = batch_mse.cpu().numpy()
        index_msssim = batch_ms_ssim.cpu().numpy()
        index_ssim = batch_ssim.cpu().numpy()
        index_psnr = batch_psnr.cpu().numpy()
        # ---------------------------------------------------------------

        result.append([slice_ID, index_mae, index_mse, index_msssim, index_ssim, index_psnr])

        print('MAE: {:.3f}\t MSE: {:.3f}\t PSNR: {:.3f}\t SSIM: {:.3f}\t MS_SSIM: {:.3f}\t'.format(
            index_mae, index_mse, index_psnr, index_ssim, index_msssim
        ))
        # ---------------------------------------------------------------

        vutils.save_image(pred_lge, fp='./result_sample/{}/{}_6_generation.png'.format(folder, slice_ID))
        vutils.save_image(origin_lge, fp='./result_sample/{}/{}_5_original.png'.format(folder, slice_ID))
        vutils.save_image(origin_cine, fp='./result_sample/{}/{}_1_cinePmax.png'.format(folder, slice_ID))
        vutils.save_image(origin_t2, fp='./result_sample/{}/{}_2_t2.png'.format(folder, slice_ID))
        vutils.save_image(origin_mask, fp='./result_sample/{}/{}_7_ori_waimo.png'.format(folder, slice_ID))
        vutils.save_image(pred_msk.float(), fp='./result_sample/{}/{}_8_gen_waimo.png'.format(folder, slice_ID))
        # -------------------------------------------------------------------------------------------

        data_result = pd.DataFrame(result, columns=['image_name', 'MAE', 'MSE', 'MS_SSIM', 'SSIM', 'PSNR'])
        data_result.to_csv('./result_index_excel/result_index_{}.csv'.format(data_name), index=False)
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        out_dir='./result_sample/',
        multi_gpu=None,
        gpu_dev="1",
        use_fp16=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
