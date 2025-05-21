from dit import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import time
import os


if __name__ == '__main__':
    n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 48
    os.makedirs('images', exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')

    dataset = torchvision.datasets.CIFAR10(
        root="cifar",
        train=True,
        download=True,
        transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
    )
    # dataset = torchvision.datasets.MNIST(
    #     root="mnist",
    #     train=True,
    #     download=True,
    #     transform=T.Compose([T.Resize((32, 32)), T.ToTensor(),]),
    # )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8
    )
    train_dataloader = cycle(train_dataloader)

    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    meanflow = MeanFlow(channels=3,
                        image_size=32,
                        num_classes=10,
                        flow_ratio=0.50,
                        cfg_ratio=None,
                        cfg_scale=None,)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0.0
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 500

    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            x = data[0].to(accelerator.device)
            c = data[1].to(accelerator.device)

            loss, mse_val = meanflow.loss(model, x, c)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            mse_losses += mse_val.item()

            if accelerator.is_main_process:
                if global_step % log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    batch_info = f'Global Step: {global_step}'
                    loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'

                    # Extract the learning rate from the optimizer
                    lr = optimizer.param_groups[0]['lr']
                    lr_info = f'Learning Rate: {lr:.6f}'

                    log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'

                    with open('log.txt', mode='a') as n:
                        n.write(log_message)

                    losses = 0.0
                    mse_losses = 0.0

            if global_step % sample_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, 'module') else model
                    z = meanflow.sample_each_class(model_module, 1)
                    log_img = make_grid(z, nrow=10)
                    img_save_path = f"images/step_{global_step}.png"
                    save_image(log_img, img_save_path)
                accelerator.wait_for_everyone()
                model.train()