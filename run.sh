CUDA_VISIBLE_DEVICES=3 python3 quant_scripts/quantize_ldm_naive.py  --n_bits_w 2 --n_bits_a 6
CUDA_VISIBLE_DEVICES=3 python3 quant_scripts/train_ourdm.py  --n_bits_w 2 --n_bits_a 6
CUDA_VISIBLE_DEVICES=3 python3 quant_scripts/sample_lora_model.py  --n_bits_w 2 --n_bits_a 6