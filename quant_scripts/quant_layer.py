import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import math
import numpy as np

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, weight_tensor = None, need_init=True, mask=None):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.mask = mask

        if weight_tensor is not None:
            self.inited = True
            if len(weight_tensor.shape) == 4:
                self.delta = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1, 1, 1))) ## removed requires_grad=False here
                self.zero_point = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1, 1, 1)))
            elif len(weight_tensor.shape) == 2:
                self.delta = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1)))
                self.zero_point = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1)))           
            else:
                print(weight_tensor.shape)
                raise ValueError('shape not implemented')
        else:
            self.inited = not need_init # use this when quantizing models
            self.delta = nn.Parameter(torch.tensor(0.005)) ## removed requires_grad=False here
            self.zero_point = nn.Parameter(torch.tensor(0.005))
        
    def clipping(self, x, lower, upper):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x
    
    def forward(self, x: torch.Tensor):
        if self.inited is False:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            if not isinstance(zero_point, torch.Tensor):
                zero_point = torch.tensor(float(zero_point))
            self.delta = torch.nn.Parameter(delta)
            self.zero_point = torch.nn.Parameter(zero_point)
            self.inited = True

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = self.clipping(x_int, 0, self.n_levels - 1) ## modified here to replace torch.clamp for gradient prop
        # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()

            ## comment below for faster initialization in inference
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if self.mask is not None:
                    if len(self.mask.shape) == 4:
                        if not self.mask[c, 0, 0, 0]:
                            delta[c] = torch.tensor(0.005)
                            zero_point[c] = torch.tensor(0.005)
                            continue
                    else:
                        if not self.mask[c, 0]:
                            delta[c] = torch.tensor(0.005)
                            zero_point[c] = torch.tensor(0.005)
                            continue
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                best_score4low = 1e10
                best_scroe4high = 1e10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
                    if not self.channel_wise:
                        n_bits = self.n_bits
                        self.n_bits = n_bits-1
                        self.n_levels == 2 ** self.n_bits
                        x_q = self.quantize(x, new_max, new_min)
                        score4low = lp_loss(x, x_q, p=2.4, reduction='all')
                        if score4low < best_score4low:
                            best_score4low = score4low
                            self.register_buffer('val_min_low', new_min.clone().detach())
                            self.register_buffer('val_max_low', new_max.clone().detach())
                        self.n_bits = n_bits+1
                        self.n_levels == 2 ** self.n_bits
                        x_q = self.quantize(x, new_max, new_min)
                        score4high = lp_loss(x, x_q, p=2.4, reduction='all')
                        if score4high < best_scroe4high:
                            best_scroe4high = score4high
                            self.register_buffer('val_min_high', new_min.clone().detach())
                            self.register_buffer('val_max_high', new_max.clone().detach())
                        self.n_bits = n_bits
            else:
                raise NotImplementedError
        if delta is None:
            print(x)
        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)
    

    
class TemporalActivationQuantizer(UniformAffineQuantizer):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, num_steps = 100):
        super(TemporalActivationQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.n_levels_small = 2 ** (self.n_bits-1)
        self.n_levels_big = 2 ** (self.n_bits+1)

        self.total_steps = num_steps
        self.current_step = self.total_steps - 1

        self.delta = nn.Parameter(torch.tensor(0.005), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0.005), requires_grad=False)
        self.inited = False ## modified here
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.register_buffer('val_min_low', torch.tensor(0.))
        self.register_buffer('val_max_low', torch.tensor(0.))
        self.register_buffer('val_min_high', torch.tensor(0.))
        self.register_buffer('val_max_high', torch.tensor(0.))
        self.split = 5
        # self.low_indice = [18, 17, 16, 15, 14, 13, 12]
        # self.high_indice = [0, 1, 2, 3, 4, 5, 6]
        self.low_indice = []
        self.high_indice = []
        
    def clipping(self, x, lower, upper):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x
    
    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.inited = True
            delta_data = [self.delta.data for _ in range(self.total_steps)]
            zp_data = [self.zero_point.data for _ in range(self.total_steps)]
            self.delta_list = nn.Parameter(torch.tensor(delta_data, device=self.delta.device), requires_grad=True)
            self.zp_list = nn.Parameter(torch.tensor(zp_data, device=self.delta.device), requires_grad=True)

            # start quantization
            x_int = round_ste(x / self.delta) + self.zero_point
            x_quant = self.clipping(x_int, 0, self.n_levels - 1) ## modified here to replace torch.clamp for gradient prop
            # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - self.zero_point) * self.delta
            return x_dequant
        else:
            # print(self.current_step)
            # start quantization
            x_int = round_ste(x / self.delta_list[self.current_step]) + self.zp_list[self.current_step]
            if self.current_step in self.low_indice:
                x_quant = self.clipping(x_int, 0, self.n_levels_small - 1) ## modified here to replace torch.clamp for gradient prop
            elif self.current_step in self.high_indice:
                x_quant = self.clipping(x_int, 0, self.n_levels_big - 1)
            else:
                x_quant = self.clipping(x_int, 0, self.n_levels - 1)
            # x_quant = self.clipping(x_int, 0, self.n_levels - 1) ## modified here to replace torch.clamp for gradient prop
            # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - self.zp_list[self.current_step]) * self.delta_list[self.current_step]
            self.current_step = self.total_steps - 1 if  self.current_step - 1 < 0 else self.current_step - 1
            return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()

            ## comment below for faster initialization in inference
            # determine the scale and zero point channel-by-channel
            # for c in range(n_channels):
            #     delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)
    

def calculate_kurtosis_per_channel(data):
    mean = torch.mean(data, dim=-1, keepdim=True)
    std = torch.std(data, dim=-1)
    
    fourth_moment_mean = torch.mean((data - mean) ** 4, dim=-1)
    
    kurtosis_per_channel = fourth_moment_mean / (std ** 4)
    
    return kurtosis_per_channel

class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None, need_init=True):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        self.weight_quant_params = weight_quant_params
        # initialize quantizer
        if not need_init:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, weight_tensor=self.weight)
        else:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params) ## delta need to be inited

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=need_init)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        if not self.ignore_reconstruction and not self.weight_quantizer.inited:
            if len(self.weight.shape) == 4:
                scale = input.abs().max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
                scale_weight = self.weight.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=0)[0]
                scale = torch.sqrt(scale.permute(0, 2, 3, 1) / scale_weight).permute(0, 3, 1, 2)
                self.register_buffer('weight_scale', scale)
                self.register_buffer('activation_scale', 1.0 / scale)
                
                weight = self.weight * self.weight_scale
                weight = weight.flatten(1)
                weight = calculate_kurtosis_per_channel(weight)
            else:
                scale = input.abs().max(dim=0)[0]
                if len(scale.shape) == 2:
                    scale = scale.max(dim=0)[0]
                scale_weight = self.weight.abs().max(dim=0)[0]
                scale = torch.sqrt(scale / scale_weight)
                scale = scale.squeeze()
                self.register_buffer('weight_scale', scale)
                self.register_buffer('activation_scale', 1.0 / scale)
            
                weight = self.weight * self.weight_scale
                weight = calculate_kurtosis_per_channel(weight)
            small_wq_params = {'n_bits': self.weight_quant_params['n_bits']-1, 'channel_wise': True, 'scale_method': 'max'}
            mid_wq_params = {'n_bits': self.weight_quant_params['n_bits'], 'channel_wise': True, 'scale_method': 'max'}
            large_res_wq_params = {'n_bits': self.weight_quant_params['n_bits']+1, 'channel_wise': True, 'scale_method': 'max'}
            out_channel = weight.shape[0]
            now_err = torch.tensor(float('inf'))
            now_mask_small = None
            now_mask_mid = None
            now_mask_large = None
            best_split = 0
            fp_out = self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)
            # for search_num in range(0, out_channel//2+1, out_channel//10):
            for search_num in [0, out_channel//10]:
                _, top_half_columns = torch.topk(weight, search_num)
                _, least_half_columns = torch.topk(weight, search_num, largest=False)
                mask_small = torch.full(self.weight.shape, False).to(self.weight.device)
                mask_large = torch.full(self.weight.shape, False).to(self.weight.device)
                if len(self.weight.shape) == 4:
                    mask_large[top_half_columns, :, :, :] = True
                    mask_small[least_half_columns, :, :, :] = True
                else:
                    mask_large[top_half_columns, :] = True
                    mask_small[least_half_columns, :] = True
                mask_mid = ~(mask_large | mask_small)
                # mask_mid = ~(mask_small)
                small_weight_quantizer = UniformAffineQuantizer(**small_wq_params, mask=mask_small)
                small_weight = small_weight_quantizer(torch.where(mask_small, self.weight * self.weight_scale, 0.))
                mid_weight_quantizer = UniformAffineQuantizer(**mid_wq_params, mask=mask_mid)
                mid_weight = mid_weight_quantizer(torch.where(mask_mid, self.weight * self.weight_scale, 0.))
                large_res_weight_quantizer = UniformAffineQuantizer(**large_res_wq_params, mask=mask_large)
                large_res_weight = large_res_weight_quantizer(torch.where(mask_large, self.weight * self.weight_scale, 0.))
                quant_weight = small_weight + mid_weight+ large_res_weight
                quant_out = self.fwd_func(input * self.activation_scale, quant_weight, self.bias, **self.fwd_kwargs)
                tmp_err = F.mse_loss(quant_out, fp_out)
                if tmp_err < now_err:
                    now_err = tmp_err
                    now_mask_large = mask_large
                    now_mask_mid = mask_mid
                    now_mask_small = mask_small   
                    best_split = search_num     
            self.register_buffer('weight_mask_large', now_mask_large)
            self.register_buffer('weight_mask_mid', now_mask_mid)
            self.register_buffer('weight_mask_small', now_mask_small)
            del self.weight_quantizer
            n_bits = self.weight_quant_params['n_bits']
            self.weight_quant_params['n_bits'] = n_bits+1
            self.weight_quantizer_large_res = UniformAffineQuantizer(**self.weight_quant_params, mask=self.weight_mask_large)
            self.weight_quant_params['n_bits'] = n_bits-1
            self.weight_quantizer_small = UniformAffineQuantizer(**self.weight_quant_params, mask=self.weight_mask_small)
            self.weight_quant_params['n_bits'] = n_bits
            self.weight_quantizer_mid = UniformAffineQuantizer(**self.weight_quant_params, mask=self.weight_mask_mid)
            print(f'best split is {best_split / out_channel}')
                
        if self.use_weight_quant:
            if not self.ignore_reconstruction:
                # weight = self.weight_quantizer(self.weight)
                weight_mid = self.weight_quantizer_mid(torch.where(self.weight_mask_mid, self.weight * self.weight_scale, 0.))
                weight_large = self.weight_quantizer_large_res(torch.where(self.weight_mask_large, self.weight * self.weight_scale, 0.))
                weight_small = self.weight_quantizer_small(torch.where(self.weight_mask_small, self.weight * self.weight_scale, 0.))
                weight = weight_large + weight_mid + weight_small
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        if self.use_act_quant:
            if not self.ignore_reconstruction:
                input = self.act_quantizer(input * self.activation_scale)
                # input = self.act_quantizer(input)
            else:
                input = self.act_quantizer(input)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    

class QuantModule_lora(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, num_steps=100):
        super(QuantModule_lora, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        
        self.ori_shape = org_module.weight.shape
        self.register_buffer('weight', torch.randn(size=[self.ori_shape[0]]+list(self.ori_shape[1:])))

        # self.weight = org_module.weight
        # self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.intn_dequantizer = None ## to be inited
        
        
        n_bits = weight_quant_params['n_bits']
        weight_quant_params['n_bits'] = n_bits+1
        self.weight_quantizer_large_res = UniformAffineQuantizer(**weight_quant_params, weight_tensor=org_module.weight)
        weight_quant_params['n_bits'] = n_bits
        self.weight_quantizer_mid = UniformAffineQuantizer(**weight_quant_params, weight_tensor=org_module.weight)
        weight_quant_params['n_bits'] = n_bits - 1
        self.weight_quantizer_small = UniformAffineQuantizer(**weight_quant_params, weight_tensor=org_module.weight)
        weight_quant_params['n_bits'] = n_bits
        
        self.act_quantizer = TemporalActivationQuantizer(**act_quant_params, num_steps=num_steps)
        
        if isinstance(org_module, nn.Conv2d):
            self.register_buffer('weight_scale', torch.randn(size=(1, self.ori_shape[1], 1, 1)))
            self.register_buffer('activation_scale', torch.randn(size=(1, self.ori_shape[1], 1, 1)))
            mask = torch.full((self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]), False).to(self.weight.device)
            # self.register_buffer('weight_mask', mask)
            self.register_buffer('weight_mask_large', mask)
            self.register_buffer('weight_mask_mid', mask.clone())
            self.register_buffer('weight_mask_small', mask.clone())
        else:
            self.register_buffer('weight_scale', torch.randn(size=(self.ori_shape[1],)))
            self.register_buffer('activation_scale', torch.randn(size=(self.ori_shape[1],)))
            mask = torch.full((self.weight.shape[0], self.weight.shape[1]), False).to(self.weight.device)
            # self.register_buffer('weight_mask', mask)
            self.register_buffer('weight_mask_large', mask)
            self.register_buffer('weight_mask_mid', mask.clone())
            self.register_buffer('weight_mask_small', mask.clone())


        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr

        ## add lora here
        r = 32
        lora_dropout = 0.0
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()
        # if isinstance(org_module, nn.Linear) and self.weight_quantizer_0.n_bits <= 8:
        if isinstance(org_module, nn.Linear):
            self.loraA = nn.Linear(org_module.in_features, r, bias=False)
            self.loraB = nn.Linear(r, org_module.out_features, bias=False)
            # self.loraA.weight.data = torch.randn(self.loraA.weight.shape)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5)) ## what's the use of a=math.sqrt(5)?
            nn.init.zeros_(self.loraB.weight)
        elif isinstance(org_module, nn.Conv2d):
            self.loraA = nn.Conv2d(org_module.in_channels, r, org_module.kernel_size, org_module.stride, org_module.padding, \
                                   org_module.dilation, org_module.groups, bias=False)
            # self.loraB = nn.Conv2d(r, org_module.out_channels, 1, bias=False)
            self.loraB = nn.Conv2d(r, org_module.out_channels, org_module.kernel_size, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5)) ## what's the use of a=math.sqrt(5)?
            nn.init.zeros_(self.loraB.weight)

    def forward(self, input: torch.Tensor):
        # orig_weight = self.intn_dequantizer(self.weight)
        orig_weight = self.weight
        if self.fwd_func is F.linear:
            E = torch.eye(orig_weight.shape[1], device=input.device)
            lora_weight = self.loraB(self.loraA(self.lora_dropout_layer(E)))
            lora_weight = lora_weight.T
            weight = orig_weight + lora_weight
        elif self.fwd_func is F.conv2d:
            lora_weight = self.loraB.weight.permute(2,3,0,1) @ self.loraA.weight.permute(2,3,0,1)
            lora_weight = lora_weight.permute(2,3,0,1)
            weight = orig_weight + lora_weight
        else:
            weight = orig_weight

        if self.use_weight_quant:
            weight_0 = self.weight_quantizer_small(torch.where(self.weight_mask_small, weight * self.weight_scale, 0.))
            weight_1 = self.weight_quantizer_mid(torch.where(self.weight_mask_mid, weight * self.weight_scale, 0.))
            weight_2 = self.weight_quantizer_large_res(torch.where(self.weight_mask_large, weight * self.weight_scale, 0.))
            weight = weight_0 + weight_1 + weight_2
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        if self.use_act_quant:
            input = self.act_quantizer(input * self.activation_scale)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
