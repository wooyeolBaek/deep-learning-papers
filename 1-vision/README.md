# vision
1. [VAEs](#vaes)
2. [Diffusion Fundamentals](#diffusion-fundamentals)
3. [Diffusion Image Editing](#diffusion-image-editing)
4. [Diffusion Fast Sampling](#diffusion-fast-sampling)
5. [Diffusion Improving Sample Quality(Fidelity)](#diffusion-improving-sample-qualityfidelity)

> ## VAEs
### Auto-Encoding Variational Bayes
- [VAE](https://www.notion.so/Auto-Encoding-Variational-Bayes-1d175f25daf64d34bba0ce10169b6910)
<!-- ### Generatibe Moments Matching Networks
- Generative Moments Matching Networks -->
<!-- ### Hierachical Variational Models
- Hierarchical VAE -->
<!-- ### Pixel Recurrent Neural Networks -->
<!-- ### Neural Discrete Representation Learning
- VQ-VAE -->
<!-- ### DVAE#: Discrete Variational Autoencoders with Relazed Boltzmann Priors
- dVAE -->
<!-- ### Generating Diverse High-Fidelity Image with VQ-VAE-2
- VQ-VAE 2 -->
<!-- ### Density estimation using Real NVP -->



<!-- > ## GAN -->
<!-- ### Generative Adversarial Networks
- [GAN](https://www.notion.so/Generative-Adversarial-Networks-1f75d05f655f4db89f4b6924c37d73e4?pvs=25) -->
<!-- ### Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
- DCGAN -->
<!-- ### Progressive Growing of GANs for Improved Quality, Stability, and Variation
- PCGAN -->
<!-- ### Large Svale GAN Training for High Fidelity Natural Image Synthesis
- BigGAN -->
<!-- ### A Style-Based Generator Architecture for Generative Adversarial Networks
- StyleGAN v1 -->
<!-- ### Analyzing and Improving the Image Quality of StyleGAN
- StyleGAN v2 -->



> ## Diffusion Fundamentals

### Deep Unsupervised Learning using Nonequilibrium Thermodynamics
- [The beginning of Diffusion method](https://www.notion.so/Deep-Unsupervised-Learning-using-Nonequilibrium-Thermodynamics-e11a140364174af8b8904350ec82de78)

### Generative Modeling by Estimating Gradients of the Data Distribution
- [NCSN: The beginning of Score-based models](https://www.notion.so/Generative-Modeling-by-Estimating-Gradients-of-the-Data-Distribution-08876e60a4da46bab93736c57a3f7bf9)
### Denoising Diffusion Probabilistic Model
- [DDPM: 노이즈 변화가 큰 부분에 집중하도록 Loss 단순화 & 다른 생성 방식들과의 연관성 발견](https://www.notion.so/Denoising-Diffusion-Probabilistic-Model-22ddaadcdb8c4c6a9f42ade7fecc2dc5?pvs=25)
### Score-based Generative Modeling through Stochastic Differential Equations
- [Score-based와 DDPM을 SDE로 묶음](https://www.notion.so/Scored-based-Generative-Modeling-through-Stochastic-Differential-Equations-1360b43bedc84d82be70c15960d6e0c3)
### Improved Denoising Diffusion Probabilistic Models
- [아키텍처 변화, 스케줄링 변화](https://www.notion.so/Improved-Denoising-Diffusion-Probabilistic-Models-96b0b384014443ac8010e31bab603c1e)
<!-- ### Zero-Shot Text-to-Image Generation
- DALLE-E1 -->
### Diffusion Models Beat GANs on Image Synthesis
- [Classifier Guidance](https://www.notion.so/Diffusion-Models-Beat-GANs-on-Image-Synthesis-748eee79547a4852999baa4a51863795?pvs=25)
<!-- ### Variational Diffusion Models
- [SNR](https://www.notion.so/Variational-Diffusion-Models-4dfcc70e9c1e4cf6b1d097595a64f724) -->
### Classifier-Free Diffusion Guidance
- [Classifier Free Guidance](https://www.notion.so/Classifier-Free-Diffusion-Guidance-616dd1cdc0ac4956815739e6b739ed1b)
### Common Diffusion Noise Schedules and Sample Steps are Flawed
- [학습시엔 T 시점에서 가우시안의 평균이 0이 아니지만, 추론시엔 평균이 0인 상태로 시작해 denoise 대상이 다른 문제 해결](https://www.notion.so/Common-Diffusion-Noise-Schedules-and-Sample-Steps-are-Flawed-184b89c9e0a04e9ead9d771c166b6fde)
### High-Resolution Image Synthesis with Latent Diffusion Models
- [Stable Diffusion: Latent Diffusion that reduce computational cost](https://www.notion.so/High-Resolution-Image-Synthesis-with-Latent-Diffusion-Models-d3a439b9f4e24a52959778a03fd7137c)
<!-- ### Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
- Imagen -->
<!-- ### Elucidating the Design Space of Diffusion-Based Generative Models
- 실험적으로 Diffusion Model을 어떻게 설계하는 것이 좋은지 -->
### SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
- [Stable Diffusion XL](https://www.notion.so/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis-0e9be93a846b48c49344ed9e95131b1c)
<!-- ### Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry -->
<!-- ### Dense Text-to-Image Generation with Attention Modulation -->
<!-- ### Adversarial Diffusion Distillation
- SDXL turbo -->
<!-- ### Diffusion Model Alignment Using Direct Preference Optimization
- Diffusion DPO -->


> ## Diffusion Image Editing
<!-- ### SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations
- SDEdit -->
### DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation
- [Image Editing method using CLIP Loss](https://www.notion.so/DiffusionCLIP-Text-Guided-Diffusion-Models-for-Robust-Image-Manipulation-129ebd9ecb2d4dbd8f5a34a47d679b07)
### Prompt-to-Prompt Image Editing with Cross Attention Control
- [Cross Attention Control을 사용한 Image Editing](https://www.notion.so/Prompt-to-Prompt-Image-Editing-with-Cross-Attention-Control-17a9b879a0e145069636ce42ba69f8a9?pvs=25)
<!-- ### InstructPix2Pix: Learning to Follow Image Editing Instructions -->
### Adding Conditional Control to Text-to-Image Diffusion Models
- [ControlNet](https://www.notion.so/Adding-Conditional-Control-to-Text-to-Image-Diffusion-Models-b36f05f1d1324313804cbee9eff828c0?pvs=25)
### Diffusion Models Already Have A Semantic Latent Space
- [h space를 사용한 Image Editing](https://www.notion.so/Diffusion-Models-Already-Have-A-Semantic-Latent-Space-96f2411ca3974298a69d795a3f0b9cb4)
<!-- ### Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning
- 훈련없이 사용자의 피드백만으로 출력 결과를 원하는 방향으로 조절 가능한 DM -->


> ## Diffusion Fast Sampling
### Denoising Diffusion Implicit Model
- [DDIM: Fast Deterministic non-Markovian sampling method](https://www.notion.so/Denoising-Diffusion-Implicit-Model-5d4e94c9c57e404ab47a1ffca6332f3e?pvs=25)
<!-- ### Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images -->
<!-- ### Progressive Distillation for Fast Sampling of Diffusion Models -->
<!-- ### DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps
- 빠른 샘플링 방식 -->
<!-- ### Denoising MCMC for Accelerating Diffusion-Based Generative Models -->
<!-- ### Pseudo Numerical Methods for Diffusion Models on Manifolds
- Fast Sampling -->


> ## Diffusion Improving Sample Quality(Fidelity)
### An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion
- [Textual Inversion: New Concept 학습시 Rare Token의 Embedding만 학습해 Unet의 적절한 조건으로 변형하는 방법](https://www.notion.so/An-Image-is-Worth-One-Word-Personalizing-Text-to-Image-Generation-using-Textual-Inversion-852d87a6930a48818399562e03b9c5c1)
<!-- ### DiffFace: Diffusion-based Face Swapping with Facial Guidance -->
<!-- ### DiffSwap: High-Fidelity and Controllable Face Swapping via 3D-Aware Masked Diffusion -->
### DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven
- [모델의 prior를 잊지 않는 New Concept 학습 방법](https://www.notion.so/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-a9198b27ce4a487ebcd7b9c5919e6736?pvs=25)
<!-- ### InstantBooth: Personalized Text-to-Image Generation without Text-Time Finetuning -->
### Inserting Anybody in Diffusion Models via Celeb Basis
- [celeb basis: Face embedding을 text space에 align해 연예인 얼굴들의 조합으로 표현](https://www.notion.so/Inserting-Anybody-in-Diffusion-Models-via-Celeb-Basis-f2cdff112e6f4a43a9a38bd838979bcd?pvs=25)
### Multi-Concept Customization of Text-to-Image Diffusion
- [Custom Diffusion: K와 V만 학습하고, Multi concept 학습 & 병합 가능](https://www.notion.so/Multi-Concept-Customization-of-Text-to-Image-Diffusion-fc74d10e2d684b0e88aaf919b186bc33?pvs=25)
### HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models
- [HyperNetwork로 모델의 Personzliaed weight를 예측하는 방법](https://www.notion.so/HyperDreamBooth-HyperNetworks-for-Fast-Personalization-of-Text-to-Image-Models-859df27772fd4d99936b3ab918cec427?pvs=25)
### MagiCapture: High-Resolution Multi-Concept Portrait Customization
- [Face embedding과 Attn refocusing을 추가한 New concept 학습 방법](https://www.notion.so/MagiCapture-High-Resolution-Multi-Concept-Portrait-Customization-23b3e532ffc347c19122d57648f906c4?pvs=25)
### FreeU: Free Lunch in Diffusion U-Net
- [UNet의 feature와 skip connection의 concat 지점에서의 비율 조절해 이미지 품질 개선](https://www.notion.so/FreeU-Free-Lunch-in-Diffusion-U-Net-98db27fac9714d3f9cd46d13481c503e?pvs=25)
<!-- ### Training Diffusion Models with Reinforcement Learning
- [Denoising과 RL을 매칭](https://maize-skink-ffe.notion.site/Training-Diffusion-Models-with-Reinforcement-Learning-1666f558c9ef45aea483d47f2f823805) -->
### Aligning Text-to-Image Models using Human Feedback
- [Human feedback을 학습한 reward 모델로 T2I 모델 finetuning(학습에 RL을 사용하진 않음)](https://maize-skink-ffe.notion.site/Aligning-Text-to-Image-Models-using-Human-Feedback-89067a8a3b0c4d2d821ffb0c9c7cc6b2)
<!-- ### DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models
- [online RL을 통해 T2I 모델의 alignment-fidelity tradeoff 문제 해결](https://maize-skink-ffe.notion.site/DPOK-Reinforcement-Learning-for-Fine-tuning-Text-to-Image-Diffusion-Models-0cea4ae9049d47bd8ef77ba4ab5099e4?pvs=74) -->
### Parrot: Pareto-optimal Multi-Reward Reinforcement Learning Framework for Text-to-Image Generation
- [생성한 이미지에서 Pareto-optimal set의 기울기만을 사용해 PEN과 T2I 모델 업데이트](https://maize-skink-ffe.notion.site/Parrot-Pareto-optimal-Multi-Reward-Reinforcement-Learning-Framework-for-Text-to-Image-Generation-75f1300547a94966a084367a61a23d9c)