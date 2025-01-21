#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

working_directory = os.getcwd()
#MODIFY LINE BELOW IF NECESSARY
sys.path.append(working_directory + 'weak_lensing_denoising-diffusion-pytorch/')

import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from astropy.io import fits
import requests
import tarfile
import os
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import numpy as np
from torchvision import transforms
from PIL import Image
import psutil


# In[ ]:


import requests
import tarfile
import os


output_directory = "raw"

url = "http://astronomy.nmsu.edu/aklypin/SUsimulations/MassiveNuS/convergence_maps/convergence_gal_mnv0.00000_om0.30000_As2.1000.tar"

os.makedirs(output_directory, exist_ok=True)

filename = url.split("/")[-1]
file_path = os.path.join(output_directory, filename)

try:
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0

    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk: 
                file.write(chunk)
                downloaded_size += len(chunk)
                progress = (downloaded_size / total_size) * 100
                print(f"\rProgress: {progress:.2f}%", end="")
    print(f"\nDownloaded {filename} to {file_path}.")
    
    print(f"Extracting {filename}...")
    with tarfile.open(file_path, "r:") as tar:
        tar.extractall(path=output_directory)
    print(f"Extracted files to {output_directory}.")

except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")
except tarfile.TarError as e:
    print(f"Error extracting the file: {e}")

# In[ ]:


import numpy as np
from astropy.io import fits
from tqdm import trange

def rebin(a, shape):
  sh = (shape[0],a.shape[0]//shape[0],
     shape[1],a.shape[1]//shape[1])

  b = a.reshape(sh)

  return a.reshape(sh).mean(-1).mean(-2)

def get_kappa(i):
  formatted_number = f"{i:04d}"
  filenames = []
  filenames.append(f"raw/Maps05/WLconv_z0.50_{formatted_number}r.fits")
  filenames.append(f"raw/Maps10/WLconv_z1.00_{formatted_number}r.fits")
  filenames.append(f"raw/Maps15/WLconv_z1.50_{formatted_number}r.fits")
  filenames.append(f"raw/Maps20/WLconv_z2.00_{formatted_number}r.fits")
  kappa_tomo = []
  for filename in filenames:
    f = fits.open(filename)
    kappa_full = f[0].data
    kappa_128 = rebin(kappa_full, (128,128))
    kappa_tomo.append(kappa_128)
  return np.array(kappa_tomo)


kappa = get_kappa(526)

os.makedirs('kappa_128_4bins/', exist_ok=True)
for i in trange(1,10000):
  kappa = get_kappa(i)
  np.save('kappa_128_4bins/%d.npy'%(i), kappa)

# In[ ]:

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

print(torch.cuda.is_available())

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False, 
    channels = 4
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 999,
    ddim_sampling_eta = 1 
)

working_directory = os.getcwd()
path = working_directory + '/kappa_128_4bins'
trainer = Trainer(
    diffusion,
    path, 
    train_batch_size = 16,
    train_lr = 8e-5,
    save_and_sample_every = 1000,
    train_num_steps = 80000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.train()




