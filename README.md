# (AAAI 2025 Oral) MPQ-DM: Mixed Precision Quantization for Extremely Low Bit Diffusion Models

[arXiv](https://arxiv.org/abs/2412.11549) | [BibTeX](#bibtex)

------

This project is the official implementation of our "MPQ-DM: Mixed Precision Quantization for Extremely Low Bit Diffusion Models".

![framework](imgs\framework.png)

## Getting Started

Follow the step-by-step tutorial to set up.

### Step 1: Setup

Create a virtual environment and install dependencies as specified by LDM.

### Step 2: Download Pretrained Models

Download the pretrained models provided by [LDM](https://github.com/CompVis/latent-diffusion).

```shell
mkdir -p models/ldm/cin256-v2/
wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
```

### Step 3: Collect Input Data for Calibration

Gather input data required for model calibration. Remember to modified the ldm/models/diffusion/ddpm.py as indicated in the quant_scripts/collect_input_4_calib.py.

```python
python3 quant_scripts/collect_input_4_calib.py
```

### Step 4: Quantize and Calibrate the Model

We just apply a naive quantization method for model calibration because we will fine-tune it afterwards.

```python
python3 quant_scripts/quantize_ldm_naive.py
```

### Step 5: Fine-Tune with MPQ-DM

```python
python3 quant_scripts/train_ourdm.py
```

### Step 6: Sample with the MPQ-DM

```python
python3 quant_scripts/sample_lora_model.py
```

------

## Comments

- Our codebase is heavily builds on [latent-diffusion](https://github.com/CompVis/latent-diffusion) and [EfficientDM](https://github.com/ThisisBillhe/EfficientDM). Thanks for open-sourcing!

## BibTeX

If you find *MPQ-DM* is useful and helpful to your work, please kindly cite this paper:

```
@inproceedings{feng2025mpq,
  title={Mpq-dm: Mixed precision quantization for extremely low bit diffusion models},
  author={Feng, Weilun and Qin, Haotong and Yang, Chuanguang and An, Zhulin and Huang, Libo and Diao, Boyu and Wang, Fei and Tao, Renshuai and Xu, Yongjun and Magno, Michele},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={16},
  pages={16595--16603},
  year={2025}
}
```
