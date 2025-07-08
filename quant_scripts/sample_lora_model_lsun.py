import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.set_grad_enabled(False)
torch.cuda.manual_seed(3407)
import torch.nn as nn
from omegaconf import OmegaConf
import argparse

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import random
import sys

from ldm.modules.ema import LitEma
from quant_scripts.quant_model import QuantModel_lora

n_bits_w = 2
n_bits_a = 8

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # model.cuda()
    # model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml")  
    model = load_model_from_config(config, "models/ldm/lsun_beds256/model.ckpt")
    return model

def get_train_samples(train_loader, num_samples):
    image_data, t_data = [], []
    for (image, t) in train_loader:
        image_data.append(image)
        t_data.append(t)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_bits_a",type=int, default=6,
        help="activation bit"
    )
    parser.add_argument(
        "--n_bits_w",type=int, default=3,
        help="weight bit"
    )
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    n_bits_w = args.n_bits_w
    n_bits_a = args.n_bits_a


    model = get_model()
    dmodel = model.model.diffusion_model

    classes = [387, 88, 979, 417]
    n_samples_per_class = 4
    ddim_steps = 100
    ddim_eta = 1.0

    from quant_scripts.quant_dataset import DiffusionInputDataset, lsunInputDataset
    from torch.utils.data import DataLoader

    dataset = lsunInputDataset('DiffusionInput_lsun_100steps.pth')
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True) ## each sample is (16,4,32,32)
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': True}
    qnn = QuantModel_lora(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params, num_steps=ddim_steps)

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    print('First run to init model...') ## need run to init act quantizer (delta_list)
    with torch.no_grad():
        _ = qnn(cali_images[:4].to(device),cali_t[:4].to(device))

    setattr(model.model, 'diffusion_model', qnn)
    ckpt = torch.load('checkpoints/lsun_bedroom/quantw{}a{}_{}steps_mpqdm.pth'.format(n_bits_w, n_bits_a, ddim_steps), map_location='cpu')
    model.load_state_dict(ckpt)
            
    model.cuda()
    model.eval()

    sampler = DDIMSampler(model)

    all_samples = list()


    with torch.no_grad():
        # with model.ema_scope():
        for times in range(3):
            print('now sample batch', times)
            t0 = time.time()
            
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            batch_size=n_samples_per_class,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                        min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)

            t1 = time.time()
            print('throughput : {}'.format(x_samples_ddim.shape[0] / (t1 - t0)))

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image_to_save = Image.fromarray(grid.astype(np.uint8))
    image_to_save.save('sample_w{}a{}.jpg'.format(n_bits_w, n_bits_a))
