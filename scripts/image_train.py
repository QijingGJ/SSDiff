"""
Train a diffusion model on images.
"""
import sys
sys.path.append("../")
sys.path.append("./")
import argparse
import torch
import torch.utils.data as data
from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from C2L_dataloader import Cine2LGEDataset, PublicDataset


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    train_dataset = Cine2LGEDataset(
        '/mnt/data_2/qijingothers2/AMI_Multitask_lynx/data_excel/data_PLA_Lynx_train.csv',
        '/mnt/data_2/qijingothers2/AMI_Multitask/data_image/', True, img_size=args.image_size)
    print('The number of training images = %d' % len(train_dataset))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_dataset,
        val_data=None,
        dataloader=train_dataloader,
        val_dataloader=None,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        args=args,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        out_dir='result_model/20250909',
        schedule_sampler="loss-second-moment",  # loss-second-moment/uniform
        lr=1e-4,
        weight_decay=0.1,
        lr_anneal_steps=50000,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        multi_gpu=None,
        gpu_dev="0",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
