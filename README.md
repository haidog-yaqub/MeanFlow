# MeanFlow
😈 This repository offers an **unofficial PyTorch implementation** of the paper [_Mean Flows for One-step Generative Modeling_](https://arxiv.org/pdf/2505.13447), building upon [Just-a-DiT](https://github.com/ArchiMickey/Just-a-DiT) and [EzAudio](https://github.com/haidog-yaqub/EzAudio).

## Examples
**MNIST** — 10k training steps, single-step sample result:

![MNIST 10k steps](assets/mnist_10k.png)

## TODO
- [x] Implement basic training and inference
- [ ] Add support for Classifier-Free Guidance (CFG) [WIP]
- [ ] Integrate support for latent image representations
- [ ] Hugging Face Space Demo
- [ ] Extend support to other modalities (e.g. audio, speech)
- [ ] Investigate applying MeanFlow to pre-trained models (e.g., via ControlNet or LoRA)

## Known Issues (PyTorch)
- `jvp` is incompatible with Flash Attention and likely also with Triton, Mamba, and similar libraries.  
- `jvp` significantly increases GPU memory usage, even when using `torch.utils.checkpoint`.
