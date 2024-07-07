import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from utils import AvgMeter, find_piece_size

from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet

from simple_diffusion.ema import EMA
from PIL import Image

from dataset import PatternDataset

IMG_SIZE = 32 # 训练用图像大小
IMG_SIZE2 = 600 # 推理用图像大小
BATCH_SIZE = 32
TIME_STEPS = 5000000
DEVICE = 'cuda:3'
pattern_file = 'pattern.jpg'
generate_rate = 0.0001 # 加噪率

img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*2-1)
])

full_image = Image.open(pattern_file).convert("RGB")
croped_image = transforms.RandomCrop(IMG_SIZE2)(full_image)
croped_image_tensor = img_trans(croped_image)
piece_size = find_piece_size(croped_image_tensor, begin=100, end = IMG_SIZE2 // 2)
image_part = transforms.RandomCrop(piece_size)(croped_image)
image_part_tensor: torch.Tensor = img_trans(image_part).to(DEVICE)


dl = DataLoader(PatternDataset(image_part, image_size=IMG_SIZE, sample_cnt=BATCH_SIZE * 1200), batch_size=BATCH_SIZE, shuffle=True, num_workers=32)
model = UNet(3, image_size=IMG_SIZE, hidden_dims=[32, 64],
                use_flash_attn=False).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
noise_scheduler = DDIMScheduler(num_train_timesteps=TIME_STEPS,
                                    beta_schedule="cosine")
avg = AvgMeter('loss', 100)

for imgs in dl:
    with torch.no_grad():
        imgs = imgs.to(DEVICE)
        batch_size = imgs.shape[0]
        noise = torch.randn(imgs.shape, device=DEVICE)
        time_steps = torch.randint(0, int(TIME_STEPS*generate_rate), [batch_size], device=DEVICE).long()
        noisy_imgs = noise_scheduler.add_noise(imgs, noise, time_steps)
    noise_pred = model(noisy_imgs, time_steps)['sample']
    opt.zero_grad()
    loss = F.l1_loss(noise_pred, noise) + F.mse_loss(noise_pred, noise)
    loss.backward()
    opt.step()
    avg.add(loss)

IMG_SIZE = IMG_SIZE2
model.sample_size = IMG_SIZE


images = image_part_tensor.unsqueeze(0)

print(images.shape)

# images = transforms.RandomCrop(piece_size)(images).unsqueeze(0)

images: torch.Tensor = noise_scheduler.generate_with_image(model, images, steps_to_do=int(TIME_STEPS*generate_rate),
                                            device=DEVICE, num_inference_steps=TIME_STEPS)['sample_pt']

images = images.repeat(1, 1, 10, 10)
images = images.cpu().permute(0, 2, 3, 1).numpy()

images_processed = (images * 255).round().astype("uint8")

for i, img in enumerate(images_processed):
    img_save = Image.fromarray(img)
    img_save.save(f'output/{i}.png')


