#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import os as os
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from matplotlib.animation import FuncAnimation
import torchvision.transforms as transforms
import imageio
from FieldAnalysis import PowerSpectrumCalculator, FieldCorrelations
from MapTools import TorchMapTools
import math


def torch_shear_to_kappa(shear, N_grid = 256, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    kappa = torch_map_tool.do_KS_inversion(shear)
    return kappa

def torch_kappa_to_shear(kappa, N_grid = 256, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    y_1, y_2   = torch_map_tool.do_fwd_KS1(kappa)
    shear_map = torch.stack((y_1, y_2))
    return shear_map

def add_noise_to_shear(shear_map, std_map):
    """
    Adds Gaussian noise to a shear map while ensuring gradients are tracked.

    Parameters:
    std_map (torch.Tensor, optional): A  tensor specifying the standard deviation at each pixel or a single scalar
    """
    if not shear_map.requires_grad:
        shear_map.requires_grad = True

    noise = torch.randn_like(shear_map) * std_map
    noisy_shear_map = shear_map + noise

    if not noisy_shear_map.requires_grad:
        noisy_shear_map.requires_grad = True

    return noisy_shear_map

def neff2noise(neff, pix_area):
    """
    :neff: Effective number density of galaxies per arcmin^2
    :pix_area: pixel area in arcmin^2
    """
    N = neff * pix_area    # avg. number of galaxies per pixel
    sigma_e = 0.26      # avg. shape noise per galaxy
    total_noise = sigma_e / math.sqrt(N)
    return total_noise

#Kappa_min and Kappa_max are dependent on the simulations used, and need to be changed accordingly
def unnorm_kappa(kappa, kappa_min = -0.08201675, kappa_max = 0.7101586):
    kappa_unnorm = (kappa * (kappa_max - kappa_min)) + kappa_min
    return kappa_unnorm

#Hyperparameters
Delta_theta = 3.5 / 256 * 60.      # Pixel side in arcmin
pix_area    = Delta_theta**2
ddim_sampling_eta = 1
batch_size = 1 #Number of samples for Unconditioned sampling, only one sample produced for conditioned at a time
neff = 10

sigma_noise = neff2noise(neff, pix_area)
print('Noise', sigma_noise)

#Prep Noisy Data Measurement
#data_single_norm is the data normed to have min 0 and max 1, data is the data normed with a global kappa min and max
x_path = "/home2/mgjacob/Diffusion/data/data_images_grey/WLconv_z2.00_0002r.png" 


working_directory = os.getcwd()
x_path = working_directory + "/kappa_128_4_bins/1.npy"

x_map = np.load(x_path)
transform = transforms.Compose([transforms.ToTensor()])
x_map = transform(torch.tensor(x_map).float()).squeeze(0)
kappa_map = unnorm_kappa(x_map)

kappa_map = kappa_map.to('cuda:0')
kappa_map.requires_grad = False

shear_map = torch_kappa_to_shear(kappa_map)
noisy_shear_map = add_noise_to_shear(shear_map, sigma_noise)
noisy_shear_map = noisy_shear_map.detach()

KS_inverse = torch_shear_to_kappa(noisy_shear_map)


#Model architecture
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False, 
    channels = 1
).cuda()

noisy_shear_map = noisy_shear_map.unsqueeze(0)
diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 1000, 
    noisy_image = noisy_shear_map, #Map to condition to
    sigma_noise = sigma_noise, #Noise specification
    ddim_sampling_eta = ddim_sampling_eta
).cuda()

trainer = Trainer(
    diffusion,
    '/home2/mgjacob/Diffusion/data/single_channel/data_images_grey',
    train_batch_size = 16,
    train_lr = 8e-5,
    save_and_sample_every = 20000,
    num_samples = 100, 
    train_num_steps = 200000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

#Specify 3 or 3-NORM, 3 being the original Diffusion Model, 3-NORM being the Diffusion Model trained on maps normed witha  global kappa max and min
trainer.load('3')

#Posterior sampling
sampled_images_posterior = diffusion.sample_posterior(batch_size = 1, return_all_timesteps = False)
x, y, *_ = sampled_images_posterior.shape


sampled_images = diffusion.sample(batch_size = batch_size, return_all_timesteps = False)
sampled_images_detached = []
for i in sampled_images: 
    sampled_images_detached.append(i.detach().cpu().squeeze())

name = "DPS"  # Replace 'NAME' with the desired folder name
samples_root = os.path.join("./samples", name)
os.makedirs(samples_root, exist_ok=True)
len_samples = len(os.listdir(samples_root))

#This type of output produces a side by side between the original x map and diffusion model sample as well as summary statistics
#Enable comparison = False in order to also view the PowerSpectrum Ratio between the x map and the diffusion model
for i in range(sampled_images_posterior.size(0)):
    index = i
    current_image_tensor = sampled_images_posterior[i].detach().cpu().squeeze(0)
    current_image_tensor = (current_image_tensor - current_image_tensor.min())/(current_image_tensor.max() - current_image_tensor.min())
    diffusion_output = unnorm_kappa(current_image_tensor)
    file_name = f"comparison_plot_{index + len_samples}"
    file_name = os.path.join(samples_root, file_name)
    FieldCorrelations(current_image_tensor, x_map, 256, file_name, comp_fields = sampled_images_detached, comparison = True, KS_inverse = KS_inverse)#, comparison = False)

print("All samples are saved in folder")
# In[ ]:





