# MeanFlow

😈 This repository provides an **unofficial PyTorch implementation** of the paper [_Mean Flows for One-step Generative Modeling_](https://arxiv.org/pdf/2505.13447).

It builds upon the following projects: [Just-a-DiT](https://github.com/ArchiMickey/Just-a-DiT) and [EzAudio](https://github.com/haidog-yaqub/EzAudio)

## TODO
- [x] Implement basic training and inference
- [ ] Add support for Classifier-Free Guidance (CFG) [WIP]
- [ ] Integrate support for latent image representations
- [ ] Hugging Face Space Demo
- [ ] Extend support to other modalities (e.g. audio, speech)
- [ ] Investigate applying MeanFlow to pre-trained models (e.g., via ControlNet or LoRA)

## Known Issues (PyTorch)
- `jvp` is incompatible with Flash Attention; likely also incompatible with Triton, Mamba, etc.
- `jvp` requires significantly more GPU memory
