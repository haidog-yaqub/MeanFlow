<img src="assets/meanflow.gif" width="2000">

# MeanFlow

😈 This repository offers an **unofficial PyTorch implementation** of the paper [_Mean Flows for One-step Generative Modeling_](https://arxiv.org/pdf/2505.13447), building upon [Just-a-DiT](https://github.com/ArchiMickey/Just-a-DiT) and [EzAudio](https://github.com/haidog-yaqub/EzAudio).

💬 Contributions and feedback are very welcome — feel free to open an issue or pull request if you spot something or have ideas!

🛠️ This codebase is kept as clean and minimal as possible for easier integration into your own projects — thus, frameworks like Wandb are intentionally excluded.

## Examples
**MNIST** -- 10k training steps, 1-step sample result:

![MNIST](assets/mnist_10k.png)

**MNIST** -- 6k training steps, 1-step CFG (w=2.0) sample result:

![MNIST-cfg](assets/mnist_6k_cfg2.png)

**CIFAR-10** -- 200k training steps, 1-step CFG (w=2.0) sample result:

![CIFAR-10-cfg](assets/cfg_200k_cfg2.png)

## 📢 Shameless Plug (Ad Spot)

🚀 Check out our recent project on stylized text-to-speech: [🧢CapSpeech](https://wanghelin1997.github.io/CapSpeech-demo/)

(yes, this is an ad. But a cool one. 😎)

## TODO
- [x] Implement basic training and inference
- [x] Enable multi-GPU training via 🤗 Accelerate
- [x] Add support for Classifier-Free Guidance (CFG)
- [x] Integrate latent image representation support
- [ ] Add tricks like improved CFG mentioned in Appendix
- [ ] Improve code clarity and structure, following 🤗 Diffusers style  
- [ ] Extend to additional modalities (e.g., audio, speech)

## Known Issues (PyTorch)
- `jvp` is incompatible with Flash Attention and likely also with Triton, Mamba, and similar libraries.  
- `jvp` significantly increases GPU memory usage, even when using `torch.utils.checkpoint`.
- CFG is implemented implicitly, leading to some limitations:
  - The CFG scale is fixed at training time and cannot be adjusted during inference.  
  - Negative prompts are not supported, such as "noise" or "low quality" commonly used in text-to-image diffusion models.
  
## 🌟 Like This Project?
If you find this repo helpful or interesting, consider dropping a ⭐ — it really helps and means a lot!
