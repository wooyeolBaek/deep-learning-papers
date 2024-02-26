# vision
1. [CNN](#cnn)
2. [Transformers](#transformers)
3. [Detection](#detection)
4. [GAN](#gan)
5. [Diffusion](#diffusion)


## 1. CNN
### Very Deep Convolutional Networks for Large-Scale Image Recognition
- VGG
### Going Deeper with Convolutions
- InceptionNet(GoogLeNet)
### U-Net: Convolutional Networks for Biomedical Image Segmentation
- UNet
<!-- ### Rethinking the Inception Architecture for Computer Vision
- InceptionNet v2, v3 -->
<!-- ### Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
- InceptionNet v4 -->
<!-- ### Indentity Mapping in Deep Residual Networks
- Improved ResNet(full pre-activation) -->
<!-- ### Residual Networks Behave Like Ensembles of Relatively Shallow Networks
- Analyzed ResNet -->
<!-- ### Wide Residual Networks
- Wide ResNet -->
### Densely Connected Convolutional Networks
- DenseNet
<!-- ### Aggregated Residual Transformations for Deep Neural Networks
- ResNeXt -->
<!-- ### MobileNets: Dfficient Convolutional Neural Networks for Mobile Vision Applications
- MobileNet v1 -->
<!-- ### MovbileNetV2: Inverted Residuals and Linear Bottlenecks
- MobileNet v2 -->
<!-- ### Implicit Generation and Generalization in Energy-Based Models
- MobileNet v3 -->
<!-- ### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- EfficientNet -->
<!-- ### A ConvNet for the 2020s
- ConvNext -->
<!-- ### ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
- ConvNeXt v2 -->


## 2. Transformers
### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- ViT


## 3. Detection
### Rich feature hierarchi accurate object detection and semantic segmentation
- R-CNN
### OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
- Overfeat
### Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
- SPPNet(RoI Projection)
### Fast R-CNN
- Fast R-CNN
### Faster R-CNN
- Faster R-CNN
### You Only Look Once: Unified, Real-Time Object Detection
- YOLOv1
### SSD: Singel Shot MultiBox Detector
- SSD
### R-FCN: Object Detection via Region-based Fully Convolutional Networks
- R-FCN
### Feature Pyramid Networks for Object Detection
- FPN
<!-- ### YOLO9000: Better, Faster, Stronger
- YOLOv2 -->
### Mask R-CNN
- Mask R-CNN
<!-- ### Focal Loss for Dense Object Detection
- RetinaNet -->
<!-- ### Squeeze-and-Excitation Networks
SE-Net -->
<!-- ### Single-Shot Refinement Neural Network for Object Detection
- RefineDet -->
<!-- ### YOLOv3: An Incremental Improvement
- YOLOv3 -->
<!-- ### M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network
- M2Det -->
<!-- ### End-to-End Object Detection with Transformers
- DETR -->







## 4. GAN
### Generative Adversarial Networks
- GAN
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



## 5.Diffusion
### Deep Unsupervised Learning using Nonequilibrium Thermodynamics
- The beginning of Diffusion method
### Generative Modeling by Estimating Gradients of the Data Distribution
- NCSN: The beginning of Score-based models
### Denoising Diffusion Probabilistic Model
- DDPM: 노이즈 변화가 큰 부분에 집중하도록 Loss 단순화 & 다른 생성 방식들과의 연관성 발견
### Denoising Diffusion Implicit Model
- DDIM: Fast Deterministic non-Markovian sampling method
### Score-based Generative Modeling through Stochastic Differential Equations
- Score-based와 DDPM을 SDE로 묶음
### Improved Denoising Diffusion Probabilistic Models
- 아키텍처 변화, 스케줄링 변화
<!-- ### Zero-Shot Text-to-Image Generation
- DALLE-E1 -->
### Diffusion Models Beat GANs on Image Synthesis
- Classifier Guidance
### Variational Diffusion Models
- SNR
<!-- ### SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations
- SDEdit -->
### Classifier-Free Diffusion Guidance
- Classifier Free Guidance
<!-- ### Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images -->
<!-- ### Progressive Distillation for Fast Sampling of Diffusion Models -->
<!-- ### Pseudo Numerical Methods for Diffusion Models on Manifolds
- Fast Sampling -->
### High-Resolution Image Synthesis with Latent Diffusion Models
- Stable Diffusion: Latent Diffusion that reduce computational cost
<!-- ### Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
- Imagen -->
<!-- ### Elucidating the Design Space of Diffusion-Based Generative Models
- 실험적으로 Diffusion Model을 어떻게 설계하는 것이 좋은지 -->
<!-- ### DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps
- 빠른 샘플링 방식 -->
### DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation
- Image Editing method using CLIP Loss
### An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion
- Textual Inversion: New Concept 학습시 Rare Token의 Embedding만 학습해 Unet의 적절한 조건으로 변형하는 방법
### Prompt-to-Prompt Image Editing with Cross Attention Control
- Cross Attention Control을 사용한 Image Editing
<!-- ### Denoising MCMC for Accelerating Diffusion-Based Generative Models -->
### Diffusion Models Already Have A Semantic Latent Space
- h space를 사용한 Image Editing
<!-- ### InstructPix2Pix: Learning to Follow Image Editing Instructions -->
<!-- ### DiffFace: Diffusion-based Face Swapping with Facial Guidance -->
<!-- ### DiffSwap: High-Fidelity and Controllable Face Swapping via 3D-Aware Masked Diffusion -->
### Adding Conditional Control to Text-to-Image Diffusion Models
- ControlNet
### DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven
- 모델의 prior를 잊지 않는 New Concept 학습방법
<!-- ### InstantBooth: Personalized Text-to-Image Generation without Text-Time Finetuning -->
### Common Diffusion Noise Schedules and Sample Steps are Flawed
- 학습시엔 T 시점에서 가우시안의 평균이 0이 아니지만, 추론시엔 평균이 0인 상태로 시작해 denoise 대상이 다른 문제 해결
<!-- ### DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models
- DPOK: Diffusion에 RLHF 적용한듯? -->
### Inserting Anybody in Diffusion Models via Celeb Basis
- celeb basis: Face embedding을 text space에 align해 연예인 얼굴들의 조합으로 표현
### Multi-Concept Customization of Text-to-Image Diffusion
- Custom Diffusion: K와 V만 학습하고, Multi concept 학습 & 병합 가능
### SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
- Stable Diffusion XL
<!-- ### Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning
- 훈련없이 사용자의 피드백만으로 출력 결과를 원하는 방향으로 조절 가능한 DM -->
### HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models
- HyperNetwork로 모델의 Personzliaed weight를 예측하는 방법
<!-- ### Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry -->
<!-- ### Dense Text-to-Image Generation with Attention Modulation -->
### MagiCapture: High-Resolution Multi-Concept Portrait Customization
- Face embedding과 Attn refocusing을 추가한 New concept 학습 방법
### FreeU: Free Lunch in Diffusion U-Net
- UNet의 feature와 skip connection의 concat 지점에서의 비율 조절해 이미지 품질 개선
<!-- ### Adversarial Diffusion Distillation
- SDXL turbo -->
<!-- ### Diffusion Model Alignment Using Direct Preference Optimization
- Diffusion DPO -->
