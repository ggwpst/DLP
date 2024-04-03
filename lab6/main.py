import os
import random
import numpy as np
import argparse
import json
import einops
import imageio
from tqdm.auto import tqdm

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
from model import MyConditionedUNet
from dataloader import ICLEVRDataset

from evaluator import evaluation_model

def plot_result(losses, test_acc, new_test_acc, args):
    epoch_full = np.arange(0, args.num_epochs)
    epoch_sub = np.arange(0, args.num_epochs, 5)
    epoch_sub = np.append(epoch_sub, epoch_full[-1])
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    plt.title('Training loss / Accuracy curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epoch_full, losses, c='silver', label='mse')
    ax1.legend(loc='lower right')

    ax2.set_ylabel('Accuracy')
    ax2.plot(epoch_sub, test_acc, label='Test')
    ax2.plot(epoch_sub, new_test_acc, label='New Test')
    ax2.legend(loc='center right')

    fig.tight_layout()
    plt.savefig('{}/trainingCurve.png'.format(args.figure_dir))
    print("-- Save training figure")


def get_test_label(args, test_file):
    label_dict = json.load(open("objects.json"))
    labels = json.load(open(test_file + ".json"))

    newLabels = []
    for i in range(len(labels)):
        onehot_label = torch.zeros(24, dtype=torch.float32)
        for j in range(len(labels[i])):
            onehot_label[label_dict[labels[i][j]]] = 1 
        newLabels.append(onehot_label)

    return newLabels

def evaluate(model, scheduler, epoch, args, device, test_file):
    test_label = torch.stack(get_test_label(args, test_file)).to(device)
    num_samples = len(test_label)

    x = torch.randn(num_samples, 3, args.sample_size, args.sample_size).to(device)
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            noise_residual = model(x, t, test_label).sample

        x = scheduler.step(noise_residual, t, x).prev_sample

    image = (x / 2 + 0.5).clamp(0, 1)

    save_image(make_grid(image, nrow=8), "{}/{}_{}.png".format(args.figure_dir, test_file, epoch))

    return x, test_label

def make_gif(model, scheduler, args, device, test_file):
    label = torch.stack(get_test_label(args, test_file)).to(device)
    num_samples = len(label)
    model.eval()

    frames = []
    x = torch.randn(num_samples, 3, args.sample_size, args.sample_size).to(device)
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_residual = model(x, t, label).sample
        
        x = scheduler.step(noise_residual, t, x).prev_sample

        image = (x / 2 + 0.5).clamp(0, 1)
        grid = make_grid(image, nrow=8).cpu().permute(1, 2, 0).numpy()
        grid = (grid * 255).round().astype("uint8")
        frames.append(grid)

    gif_path = "{}/{}.gif".format(args.gif_dir, test_file)
    with imageio.get_writer(gif_path, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(len(scheduler.timesteps) // 3):
                    writer.append_data(frames[-1])

    print("> Save {}".format(gif_path))

def train(losses, global_step):
    progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")

    total_loss = 0
    for i, (x, class_label) in enumerate(train_loader):
        x, class_label = x.to(device), class_label.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 1000, (x.shape[0],)).long().to(device)
        noisy_image = noise_scheduler.add_noise(x, noise, timesteps)

        with accelerator.accumulate(model):
            noise_pred = model(noisy_image, timesteps, class_label).sample

            loss = loss_fn(noise_pred, noise)
            total_loss += loss.item()
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        logs = {"loss": total_loss / (i+1), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.update(1)
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1

    losses.append(total_loss / len(train_loader))
    return losses

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default="./dataset", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--figure_dir', default="figures", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_epochs', default=50, type=int) 
    parser.add_argument('--sample_size', default=64, type=int)
    parser.add_argument('--beta_schedule', default="linear", type=str)
    parser.add_argument('--predict_type', default="epsilon", type=str)
    parser.add_argument('--block_dim', default=128, type=int)
    parser.add_argument('--layers_per_block', default=2, type=int)
    parser.add_argument('--lr_warmup_steps', default=500, type=int)
    parser.add_argument('--mixed_precision', default="fp16", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device: {}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    name = 'lr={:.5f}-lr_warmup={}-block_dim={}-layers={}-schedule={}-predict_type={}'.format(args.lr, args.lr_warmup_steps, args.block_dim, args.layers_per_block, args.beta_schedule, args.predict_type)
    args.log_dir = './%s/%s' % (args.log_dir, name)
    args.figure_dir = '%s/%s' % (args.log_dir, args.figure_dir)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    with open('{}/train_record.txt'.format(args.log_dir), 'w') as train_record:
        train_record.write('args: {}\n'.format(args))

    sample_size = args.sample_size
    block_dim = args.block_dim
    layers = args.layers_per_block
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size

    model = MyConditionedUNet(
        sample_size=sample_size,       
        in_channels=3,                     
        out_channels=3,
        layers_per_block=layers,
        block_out_channels=(block_dim, block_dim, block_dim*2, block_dim*2, block_dim*4, block_dim*4),
        down_block_types=(
            "DownBlock2D",          
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",      
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",        
            "UpBlock2D",            
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type=args.predict_type, beta_schedule=args.beta_schedule) 

    transform = transforms.Compose([
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = ICLEVRDataset(args, mode='train', transforms=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=len(train_loader) * num_epochs,
    )
    evaluation = evaluation_model()

    state = {
        "model": model.state_dict(),
        "best_epoch": 0,
        "test_acc": 0,
        "new_test_acc": 0,
    }

    losses = []
    acc_list = []
    new_acc_list = []
    best_acc = 0
    best_new_acc = 0
    global_step = 0

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.log_dir, "logging"),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    for epoch in range(num_epochs):
        losses = train(losses, global_step)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            test_image, test_label = evaluate(model, noise_scheduler, epoch, args, device, "test")
            new_test_image, new_test_label = evaluate(model, noise_scheduler, epoch, args, device, "new_test")
            test_acc = evaluation.eval(test_image, test_label)
            new_test_acc = evaluation.eval(new_test_image, new_test_label)
            acc_list.append(test_acc)
            new_acc_list.append(new_test_acc)
            print("> Accuracy: [Test]: {:.4f}, [New Test]: {:.4f}".format(test_acc, new_test_acc))

            with open('{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('[Epoch: %02d] loss: %.5f | test acc: %.5f | new_test acc: %.5f\n' % (epoch, losses[-1], test_acc, new_test_acc)))

            if test_acc >= best_acc and new_test_acc >= best_new_acc:
                state["model"] = model.state_dict()
                state["best_epoch"] = epoch
                state["test_acc"] = test_acc
                state["new_test_acc"] = new_test_acc
                best_acc = test_acc
                best_new_acc = new_test_acc
                torch.save(state, os.path.join(args.log_dir, "model.pth"))
                print("checkpoint")

    plot_result(losses, acc_list, new_acc_list, args)