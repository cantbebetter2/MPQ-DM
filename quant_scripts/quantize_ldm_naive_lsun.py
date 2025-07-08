import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os, time
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn
from omegaconf import OmegaConf
import argparse

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from quant_scripts.quant_model import QuantModel
from quant_scripts.quant_layer import QuantModule


from tqdm import tqdm

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # model.cuda()
    model.eval()
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
        "--n_bits_a",type=int, default=8,
        help="activation bit"
    )
    parser.add_argument(
        "--n_bits_w",type=int, default=4,
        help="weight bit"
    )
    return parser


if __name__ == '__main__':
    model = get_model()
    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)
    model = model.model.diffusion_model
    
    model.cuda()
    model.eval()

    parser = get_parser()
    args = parser.parse_args()
    n_bits_w = args.n_bits_w
    n_bits_a = args.n_bits_a

    from quant_scripts.quant_dataset import DiffusionInputDataset, lsunInputDataset
    from torch.utils.data import DataLoader

    dataset = lsunInputDataset('DiffusionInput_lsun_100steps.pth')
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    
    # wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'max'}
    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params, need_init=True)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    start = time.time()
    print('First run to init model...')
    with torch.no_grad():
        _ = qnn(cali_images[:256].to(device),cali_t[:256].to(device))

    torch.save(qnn.state_dict(), 'checkpoints/lsun_bedroom/quantw{}a{}_naiveQ.pth'.format(n_bits_w, n_bits_a))
    # torch.save(qnn.state_dict(), 'checkpoints/lsun_bedroom/quantw{}a{}_naiveQ.pth'.format(n_bits_w, n_bits_a))
    pass