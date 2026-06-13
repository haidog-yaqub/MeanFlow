<img src="assets/meanflow.gif" width="2000">

# MeanFlow

PyTorch implementation of [Mean Flows for One-step Generative Modeling](https://arxiv.org/pdf/2505.13447) (MeanFlow) and [Improved Mean Flows](https://arxiv.org/abs/2512.02012) (iMF).

> **Note:** Unofficial implementation, based on the papers above and the official JAX repo [imeanflow](https://github.com/Lyy-iiis/imeanflow).

> Contributions and feedback are welcome! Feel free to open an issue or pull request.

## Updates

**2026.06.13**
  - MeanFlow and iMF training (`meanflow.mode`: `"meanflow"` / `"i-meanflow"`)
  - Dual-head DiT: `u` for MeanFlow, `v` for flow matching
  - CFG scale as model input with CFG distillation (`cfg_scale`; `None` to disable)
  - JVP under `no_grad` for `dudt` only; separate grad-enabled forward pass for optimization
  - Config-based training for MNIST, CIFAR-10, and ImageNet latent

## Usage

`i-meanflow` (`meanflow.mode: "i-meanflow"`) is more stable and recommended for your projects. `meanflow` mode is kept for reference.

```bash
pip install torch accelerate torchvision einops tqdm diffusers
```

```bash
# single GPU
python train.py --config configs/mnist.py

# custom run name
python train.py --config configs/cifar10.py --run_suffix exp1

# multi-GPU
accelerate launch --num_processes 2 train.py --config configs/mnist.py
```

| Config | Dataset |
|--------|---------|
| `configs/mnist.py` | MNIST |
| `configs/cifar10.py` | CIFAR-10 |
| `configs/imagenet_latent.py` | ImageNet (latent, VAE) |

Common config fields: `n_steps`, `batch_size`, `grad_clip`, `mixed_precision`, `meanflow.mode`, `meanflow.cfg_scale`.

Training logs are saved to `logs/{run_name}/`:

```
logs/{run_name}/
├── config.py
├── train.log      # loss, FM/MF loss, MF_V_MSE, grad norm, LR
├── images/
└── ckpts/
```

## Examples

**MNIST** — 10k steps, 1-step sample:

![MNIST](assets/mnist_10k.png)

**MNIST** — 6k steps, 1-step CFG (w=2.0):

![MNIST-cfg](assets/mnist_6k_cfg2.png)

**CIFAR-10** — 200k steps, 1-step CFG (w=2.0):

![CIFAR-10-cfg](assets/cfg_200k_cfg2.png)

## TODO

- [x] Basic training and inference
- [x] Multi-GPU via Accelerate
- [x] Classifier-Free Guidance
- [x] Latent image training
- [x] Improved MeanFlow (iMF)
- [ ] Triton JVP + Flash Attention

## Known Issues

- `jvp` is currently incompatible with PyTorch's native Flash Attention (`scaled_dot_product_attention`).

- Solution in this repo:
  - Run the JVP pass under `no_grad` (for `dudt` only) without Flash Attention.
  - Run a separate gradient-enabled forward pass for optimization with Flash Attention.

- Other advanced solutions:
  - Triton-based kernels with JVP support (e.g. [rcm](https://github.com/NVlabs/rcm)) will allow Flash Attention within JVP.
  - Such kernels may offer additional speed and memory benefits, but are not currently supported in this repository.

## Acknowledgement

Building upon [Just-a-DiT](https://github.com/ArchiMickey/Just-a-DiT) and [EzAudio](https://github.com/haidog-yaqub/EzAudio).

iMF support is ported to PyTorch, building upon the official JAX repo [imeanflow](https://github.com/Lyy-iiis/imeanflow). Thanks to the authors for releasing their code and checkpoints.

## Like This Project?

If you find this repo helpful, consider dropping a ⭐, it really helps!

## Citation

If you find this repository useful in your research or projects, please consider citing the original MeanFlow papers as well as this implementation.

### MeanFlow

```bibtex
@article{geng2025meanflow,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Shechtman, Eli and Kolter, J. Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}
```

### Improved MeanFlow

```bibtex
@article{geng2025improved,
  title={Improved Mean Flows: On the Challenges of Fastforward Generative Models},
  author={Geng, Zhengyang and Lu, Yiyang and Wu, Zongze and Shechtman, Eli and Kolter, J. Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2512.02012},
  year={2025}
}
```

### This Repository

Repository: https://github.com/haidog-yaqub/MeanFlow

```bibtex
@misc{meanflow_pytorch,
  title={MeanFlow: Unofficial PyTorch Implementation},
  author={haidog-yaqub},
  year={2025},
  howpublished={\url{https://github.com/haidog-yaqub/MeanFlow}},
}
```
