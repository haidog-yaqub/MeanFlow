import argparse
import importlib.util
import math
import os
import shutil
import time
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data import build_dataset, cycle
from meanflow import MeanFlow
from models.dit import MFDiT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run_suffix', type=str, default=None)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location('train_config', args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.config

    accelerator = Accelerator(
        mixed_precision=cfg.get('mixed_precision', 'bf16'),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    train_dataloader = torch.utils.data.DataLoader(
        build_dataset(cfg),
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=cfg['num_workers'],
    )

    use_latent = cfg.get('latent', False)
    vae = None
    if use_latent:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(cfg['vae_id']).to(accelerator.device)

    model = MFDiT(**cfg['model']).to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    scheduler = None
    sched_cfg = cfg.get('scheduler')
    if sched_cfg is not None:
        warmup_steps = sched_cfg.get('warmup_steps', 0)
        min_lr_ratio = sched_cfg.get('min_lr_ratio', 0.1)
        total_steps = cfg['n_steps']

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(step, 1) / warmup_steps
            if step >= total_steps:
                return min_lr_ratio
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    mf_cfg = dict(cfg['meanflow'])
    mf_cfg['channels'] = cfg['model']['in_channels']
    mf_cfg['image_size'] = cfg['model']['input_size']
    mf_cfg['num_classes'] = cfg['model']['num_classes']
    meanflow = MeanFlow(**mf_cfg)

    if vae is not None:
        model, vae, optimizer, train_dataloader = accelerator.prepare(
            model, vae, optimizer, train_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        run_name = (
            f"{cfg['name']}_{args.run_suffix}"
            if args.run_suffix else
            f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        cfg.update({
            'run_name': run_name,
            'log_dir': os.path.join('logs', run_name),
            'image_dir': os.path.join('logs', run_name, 'images'),
            'ckpt_dir': os.path.join('logs', run_name, 'ckpts'),
        })
        os.makedirs(cfg['log_dir'], exist_ok=True)
        os.makedirs(cfg['image_dir'], exist_ok=True)
        os.makedirs(cfg['ckpt_dir'], exist_ok=True)
        cfg['log_file'] = os.path.join(cfg['log_dir'], 'train.log')
        shutil.copy2(args.config, os.path.join(cfg['log_dir'], 'config.py'))
        with open(cfg['log_file'], 'w') as f:
            f.write(f"run_name: {run_name}\nconfig: {args.config}\n\n")
    accelerator.wait_for_everyone()

    model_module = model.module if hasattr(model, 'module') else model
    train_dataloader = cycle(train_dataloader)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model_module.parameters())
        print(f"MF mode: {meanflow.mode}")
        print(f"Model size: {n_params / 1e6:.2f}M")
        with open(cfg['log_file'], 'a') as f:
            f.write(f"MF mode: {meanflow.mode}\n")
            f.write(f"Model size: {n_params / 1e6:.2f}M\n\n")

    global_step = 0
    grad_clip = cfg['grad_clip']
    losses = log_fm_loss = log_mf_loss = log_mf_v_mse = log_grad_norm = 0.0

    model.train()
    pbar = tqdm(
        range(cfg['n_steps']),
        desc=cfg['name'],
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )
    for _ in pbar:
        x, c = next(train_dataloader)
        x = x.to(accelerator.device)
        c = c.to(accelerator.device)
        if use_latent:
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample()

        loss, mse_val = meanflow.loss(model, x, c)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        losses += loss.item()
        log_fm_loss += mse_val['fm_loss'].item()
        log_mf_loss += mse_val['mf_loss'].item()
        log_mf_v_mse += mse_val['mf_v_mse'].item()
        log_grad_norm += grad_norm.item()

        if accelerator.is_main_process:
            pbar.set_postfix(
                Loss=f"{loss.item():.4f}",
                MF_V_MSE=f"{mse_val['mf_v_mse'].item():.4f}",
                refresh=False,
            )

        if accelerator.is_main_process and global_step % cfg['log_step'] == 0:
            s = cfg['log_step']
            with open(cfg['log_file'], 'a') as f:
                lr = optimizer.param_groups[0]['lr']
                f.write(
                    f"{time.asctime(time.localtime(time.time()))}\n"
                    f"Step: {global_step}    LR: {lr:.6f}    Grad_Norm: {log_grad_norm / s:.6f}\n"
                    f"Loss: {losses / s:.6f}    FM_Loss: {log_fm_loss / s:.6f}    "
                    f"MF_Loss: {log_mf_loss / s:.6f}    MF_V_MSE: {log_mf_v_mse / s:.6f}\n"
                )
            losses = log_fm_loss = log_mf_loss = log_mf_v_mse = log_grad_norm = 0.0

        if global_step % cfg['sample_step'] == 0:
            if accelerator.is_main_process:
                z = meanflow.sample_each_class(
                    model_module, n_per_class=1, classes=cfg['sample_classes']
                )
                if use_latent:
                    with torch.no_grad():
                        z = vae.decode(z).sample
                    z = z * 0.5 + 0.5
                save_image(
                    make_grid(z, nrow=cfg['sample_nrow'] or int(z.shape[0] ** 0.5)),
                    os.path.join(cfg['image_dir'], f'step_{global_step:07d}.png'),
                )
            accelerator.wait_for_everyone()
            model.train()

    if accelerator.is_main_process:
        accelerator.save(
            model_module.state_dict(),
            os.path.join(cfg['ckpt_dir'], f'step_{global_step:07d}.pt'),
        )
