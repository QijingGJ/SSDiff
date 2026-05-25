# SSDiff: A Contrast-Free Virtual LGE Generator for Acute Myocardial Infarction with Joint Segmentation via Diffusion Model ([PDF](https://ieeexplore.ieee.org/document/11503374))

<img width="1186" height="1268" alt="image" src="https://github.com/user-attachments/assets/dfccddf1-6f3b-462f-bd2e-43b804aefcc7" />


## Training

```python
python scripts/image_train.py --image_size 128 --learn_sigma True --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 16
```

## Sampling

```python
python scripts/image_sample.py --model_path ./result_model/20250901/model050000.pt --image_size 128 --learn_sigma True --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
```

## Thanks

Thanks to the base code [IDDPM](https://github.com/openai/improved-diffusion) and [DPM solver](https://github.com/LuChengTHU/dpm-solver)

## Citation

```python
Qi J, Yue X, Hu M, et al. SSDiff: A Contrast-Free Virtual LGE Generator for Acute Myocardial Infarction with Joint Segmentation via Diffusion Model. IEEE J Biomed Health Inform. Published online May 4, 2026.
```
