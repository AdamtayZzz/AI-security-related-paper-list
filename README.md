# AI Security Paper List
- [Attack-Related](#attack-related)
  * [Adversarial Attack](#adversarial-attack)
    + [Survey](#survey)
    + [Attack-CV](#attack-cv)
    + [Defense-CV](#defense-cv)
    + [Attack-others](#attack-others)
  * [Backdoor Attack](#backdoor-attack)
    + [Survey](#survey-1)
    + [Attack - Data poisoning](#attack---data-poisoning)
    + [Attack - not Data poisoning](#attack---not-data-poisoning)
    + [Attack - others](#attack---others)
    + [Defense](#defense)
  * [Data Poisoning](#data-poisoning)
  * [Hardware Fault Attack](#hardware-fault-attack)
    + [Preliminaries](#preliminaries)
    + [Applications](#applications)
- [Privacy-Related](#privacy-related)
  * [Data Reconstruction](#data-reconstruction)
  * [Membership Inference Attack](#membership-inference-attack)
  * [Property Inference Attack](#property-inference-attack)
  * [Others](#others)
- [Machine Learning Related](#machine-learning-related)
  * [Fingerprint](#fingerprint)
    + [Model Watermarking](#model-watermarking)
    + [Dataset Fingerprint](#dataset-fingerprint)
  * [Watermarking](#watermarking)
    + [Model Watermarking](#model-watermarking-1)
    + [Dataset Watermarking](#dataset-watermarking)
  * [Machine Unlearning](#machine-unlearning)
  * [Theoretical Analysis](#theoretical-analysis)
  * [Model Hiding](#model-hiding)
  * [Model Design for Efficiency](#model-design-for-efficiency)
    + [Quantization](#quantization)
    + [Dynamic Inference](#dynamic-inference)
  * [Neural Network Interpretability](#neural-network-interpretability)
- [DNN Application](#dnn-application)
  * [Transformer](#transformer)
    + [Transformer in Computer Vision](#transformer-in-computer-vision)
  * [Reinforcement Learning](#reinforcement-learning)
    + [Design](#design)
      - [Basic](#basic)
      - [Multi-Agents](#multi-agents)
    + [Adversarial Attack on RL](#adversarial-attack-on-rl)
    + [Others](#others-1)
  * [Person Re-identification](#person-re-identification)
    + [Survey](#survey-2)
    + [Toolbox](#toolbox)
    + [Design](#design-1)
    + [Adversarial Attack on ReID](#adversarial-attack-on-reid)
- [Mathematics](#mathematics)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


# Attack-Related

## Adversarial Attack

### Survey

- 2018 IEEE ACCESS [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://arxiv.org/abs/1801.00553)
- 2020 Engineering [Adversarial Attacks and Defenses in Deep Learning](https://www.sciencedirect.com/science/article/pii/S209580991930503X)

### Attack-CV

- 2014 ICLR [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)
- 2015 ICLR FGSM [Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572)
- 2016 S&P JSMA [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528)
- 2016 CVPR [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599)
- 2017 CCS [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/abs/1602.02697)
- 2016 arXiv[Delving into Transferable Adversarial Examples and Black-box Attacks](https://arxiv.org/abs/1611.02770)
- 2017 ICLR targeted FGSM [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)
- 2017 ICLR BIM&ICLM [Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533)
- 2017 S&P C&W [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644)
- 2017 CVPR [Universal adversarial perturbations](https://arxiv.org/abs/1610.08401)
- 2018 ICLR PGD[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
- 2018 IEEE TECV one-pixel attack [One pixel attack for fooling deep neural networks](https://arxiv.org/abs/1710.08864)
- 2018 AAAI [Adversarial Transformation Networks: Learning to Generate Adversarial Examples](https://arxiv.org/abs/1703.09387)
- 2018 CVPR MI-FGSM[Boosting Adversarial Attacks With Momentum](https://arxiv.org/abs/1710.06081v3)
- 2018 ICML [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/abs/1802.00420)
- 2020 USENIX [Interpretable Deep Learning under Fire](https://arxiv.org/abs/1812.00891)

### Defense-CV

- 2015 arxiv [Foveation-based Mechanisms Alleviate Adversarial Examples](https://arxiv.org/abs/1511.06292)
- 2016 CVPR [A study of the effect of JPG compression on adversarial images](https://arxiv.org/abs/1608.00853)
- 2016 S&P [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](https://arxiv.org/abs/1511.04508)
- 2017 ICCV [Adversarial Examples for Semantic Segmentation and Object Detection](https://arxiv.org/abs/1703.08603)
- 2017 arxiv [Mitigating adversarial effects through randomization](https://arxiv.org/abs/1711.01991)
- 2017 arxiv [DeepCloak: Masking Deep Neural Network Models for Robustness Against Adversarial Samples](https://arxiv.org/abs/1702.06763)
- 2017 arxiv [Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN](https://arxiv.org/abs/1705.03387)
- 2018 AAAI [Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients](https://arxiv.org/abs/1711.09404)
- 2018 CVPR [Defense against Universal Adversarial Perturbations](https://arxiv.org/abs/1711.05929)
- 2019 NeurIPS [Certified Adversarial Robustness with Additive Noise](https://arxiv.org/abs/1809.03113)
- 2019 NDSS [NIC: Detecting Adversarial Samples with Neural Network Invariant Checking](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf)
- 2020 arxiv [Detection Defense Against Adversarial Attacks with Saliency Map](https://arxiv.org/abs/2009.02738)
- [ ] 2021 arxiv [Meta Adversarial Training against Universal Patches](https://arxiv.org/abs/2101.11453)

### Attack-others
- 2018 NDSS [TextBugger: Generating Adversarial Text Against Real-world Applications](https://arxiv.org/abs/1812.05271)
- 2019 S&P [Intriguing Properties of Adversarial ML Attacks in the Problem Space](https://arxiv.org/abs/1911.02142)
- 2019 IJCNN [Adversarial Attacks on Deep Neural Networks for Time Series Classification](https://arxiv.org/abs/1903.07054)

## Backdoor Attack

### Survey
- 2012 [TROJANZOO: Everything you ever wanted to know about neural backdoors (but were afraid to ask)](https://arxiv.org/abs/2012.09302)
- 2020 [Backdoor Learning: A Survey](https://arxiv.org/abs/2007.08745)

### Attack - Data poisoning

- 2017 Arxiv [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733)
- 2018 NDSS [Trojaning Attack on Neural Networks](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech)
- 2018 CCS [Model-Reuse Attacks on Deep Learning Systems](https://arxiv.org/abs/1812.00483)
- 2019 CCS [Latent Backdoor Attacks on Deep Neural Networks](http://people.cs.uchicago.edu/~huiyingli/publication/fr292-yaoA.pdf)
- 2020 CCS [A Tale of Evil Twins: Adversarial Inputs versus Poisoned Models](https://arxiv.org/abs/1911.01559)
- 2020 AAAI [Hidden Trigger Backdoor Attacks](https://arxiv.org/abs/1910.00033)
- 2020 NeurIPS [Input-Aware Dynamic Backdoor Attack](https://arxiv.org/abs/2010.08138)
- 2020 arxiv [Dynamic Backdoor Attacks Against Machine Learning Models](https://arxiv.org/abs/2003.03675)
- 2020 arxiv [Backdoor Attacks on the DNN Interpretation System](https://arxiv.org/abs/2011.10698)
- 2021 USENIX Security [Blind Backdoors in Deep Learning Models](https://arxiv.org/abs/2005.03823)
- 2021 ICCV [Invisible Backdoor Attack with Sample-Specific Triggers](https://arxiv.org/abs/2012.03816)
- 2021 Infocom [Invisible Poison: A Blackbox Clean Label Backdoor Attack to Deep Neural Networks](https://ieeexplore.ieee.org/document/9488902)
- 2021 ICLR [WaNet -- Imperceptible Warping-based Backdoor Attack](https://arxiv.org/abs/2102.10369)


### Attack - not Data poisoning 
- 2018 Arxiv [Backdooring Convolutional Neural Networks via Targeted Weight Perturbations](https://arxiv.org/abs/1812.03128)
- 2020 arxiv [Don't Trigger Me! A Triggerless Backdoor Attack Against Deep Neural Networks](https://arxiv.org/abs/2010.03282)
- 2020 CVPR [TBT: Targeted Neural Network Attack with Bit Trojan](https://arxiv.org/abs/1909.05193)
- 2020 CIKM [Can Adversarial Weight Perturbations Inject Neural Backdoors?](https://arxiv.org/abs/2008.01761)
- 2020 SIGKDD [An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks](https://arxiv.org/abs/2006.08131)
- 2021 ICSE [DeepPayload: Black-box Backdoor Attack on Deep Learning Models through Neural Payload Injection](https://arxiv.org/abs/2101.06896)
- 2021 ICLR WORKSHOP [Subnet Replacement: Deployment-stage backdoor attack against deep neural networks in gray-box setting](https://arxiv.org/abs/2107.07240)

### Attack - others 
- [ ] 2020 PMLR [How To Backdoor Federated Learning](https://arxiv.org/abs/1807.00459)

### Defense
- 2018 RAID [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://arxiv.org/abs/1805.12185) 
- 2018 NuerIPS [Spectral Signatures in Backdoor Attacks](https://arxiv.org/abs/1811.00636)
- 2019 S&P [Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf)
- 2019 CCS [ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation](https://dl.acm.org/doi/10.1145/3319535.3363216)
- 2019 IJCAI [DeepInspect: A Black-box Trojan Detection and Mitigation Framework for Deep Neural Networks](https://www.ijcai.org/Proceedings/2019/647)
- 2019 AAAI [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://arxiv.org/pdf/1811.03728.pdf)
- 2019 ACSAC [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531)
- 2020 ECCV [One-pixel Signature: Characterizing CNN Models for Backdoor Detection](https://arxiv.org/abs/2008.07711)
- 2020 ICDM [TABOR: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems](https://arxiv.org/abs/1908.017632)
- 2020 ACSAC [Februus: Input Purification Defense Against Trojan Attacks on Deep Neural Network](https://arxiv.org/abs/1908.03369)
- 2021 ICLR [Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks](https://arxiv.org/abs/2101.05930)

## Data Poisoning

- 2017 ICML [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)
- 2018 NeurIPS [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)


## Hardware Fault Attack

### Preliminaries
- 2014 ISCA [Flipping Bits in Memory Without Accessing Them: An Experimental Study of DRAM Disturbance Errors](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf)

### Applications

- 2019 USENIX [Terminal brain damage: Exposing the graceless degradation in deep neural networks under hardware fault attacks](https://arxiv.org/abs/1906.01017)
- 2020 USENIX [DeepHammer: Depleting the Intelligence of Deep Neural Networks through Targeted Chain of Bit Flips](https://arxiv.org/abs/2003.13746)

# Privacy-Related 

## Data Reconstruction

- 2015 CCS [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures](https://dl.acm.org/doi/10.1145/2810103.2813677)
- 2017 CCS [Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning](https://arxiv.org/abs/1702.07464)
- 2019 NeurIPS [Deep Leakage from Gradients](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients.pdf)
- 2020 USENIX Security [Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning](https://arxiv.org/abs/1904.01067)
- 2021 CVPR [See through Gradients: Image Batch Recovery via GradInversion](https://arxiv.org/abs/2104.07586)
- 2022 USENIX Security [Theory-Oriented Deep Leakage from Gradients via Linear Equation Solver](https://arxiv.org/abs/2010.13356)

## Membership Inference Attack

- 2017 S&P [Membership Inference Attacks against Machine Learning Models](https://arxiv.org/abs/1610.05820)
- 2018 Arxiv [Understanding Membership Inferences on Well-Generalized Learning Models](https://arxiv.org/abs/1802.04889)
- 2018 CSF [Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting](https://arxiv.org/abs/1709.01604)
- 2018 CCS [Machine Learning with Membership Privacy using Adversarial Regularization](https://arxiv.org/abs/1807.05852)
- 2019 NDSS [ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models](https://arxiv.org/abs/1806.01246)
- 2019 CCS [Privacy Risks of Securing Machine Learning Models against Adversarial Examples](https://arxiv.org/abs/1905.10291)
- 2019 CCS [MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples](https://arxiv.org/abs/1909.10594)
- 2019 S&P [Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning](https://arxiv.org/abs/1812.00910)
- 2019 ICML [White-box vs Black-box: Bayes Optimal Strategies for Membership Inference](https://arxiv.org/pdf/1908.11229.pdf)
- 2020 USENIX [Stolen Memories: Leveraging Model Memorization for Calibrated White-Box Membership Inference](https://arxiv.org/abs/1906.11798)
- 2021 CCS [When Machine Unlearning Jeopardizes Privacy](https://arxiv.org/abs/2005.02205)
- 2021 ICML [Label-Only Membership Inference Attacks](https://arxiv.org/abs/2007.14321)
- 2021 ICCV [Membership Inference Attacks are Easier on Difficult Problems](https://arxiv.org/abs/2102.07762)
- [ ] 2021 ICML [When Does Data Augmentation Help With Membership Inference Attacks?](http://proceedings.mlr.press/v139/kaya21a.html) 
- 2021 CCS [Membership Leakage in Label-Only Exposures](https://arxiv.org/abs/2007.15528)
- 2021 AAAI [Membership Privacy for Machine Learning Models Through Knowledge Transfer](https://arxiv.org/abs/1906.06589)
- 2021 USENIX [Systematic Evaluation of Privacy Risks of Machine Learning Models](https://arxiv.org/abs/2003.10595)
- 2021 Arxiv [Source Inference Attacks in Federated Learning](https://arxiv.org/abs/2109.05659)
- 2021 CVPR [On the Difficulty of Membership Inference Attacks](https://arxiv.org/abs/2005.13702)
- 2021 AIES [On the Privacy Risks of Model Explanations](https://arxiv.org/abs/1907.00164)

## Property Inference Attack

- 2018 CCS [Property Inference Attacks on Fully Connected Neural Networksusing Permutation Invariant Representations](https://dl.acm.org/doi/10.1145/3243734.3243834)

## Others
- 2017 CCS [Machine Learning Models that Remember Too Much](https://arxiv.org/abs/1709.07886)

# Machine Learning Related
## Fingerprint
### Model Watermarking
- 2019 AsiaCCS [IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary](https://arxiv.org/abs/1910.12903)
- 2021 ESORICS [TAFA: A Task-Agnostic Fingerprinting Algorithm for Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-88418-5_26)
### Dataset Fingerprint
- 2022 IEEE TIFS [Your Model Trains on My Data? Protecting Intellectual Property of Training Data via Membership Fingerprint Authentication](https://ieeexplore.ieee.org/document/9724248)

## Watermarking
### Model Watermarking
- 2017 ICMR [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/abs/1701.04082)
- [ ] 2018 USENIX[Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://arxiv.org/abs/1802.04633)
- 2019 ASPLOS [DeepSigns: An End-to-End Watermarking Framework for Ownership Protection of Deep Neural Networks](https://dl.acm.org/doi/10.1145/3297858.3304051)\
- 2019 ICASSP [Attacks on Digital Watermarks for Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/8682202/)
- 2021 WWW [RIGA: Covert and Robust White-BoxWatermarking of Deep Neural Networks](https://arxiv.org/abs/1910.14268)

### Dataset Watermarking
- 2020 NeurIPS Workshop [Open-sourced Dataset Protection via Backdoor Watermarking](https://arxiv.org/abs/2010.05821)
- 2020 ICML [Radioactive data: tracing through training](http://proceedings.mlr.press/v119/sablayrolles20a.html)

## Machine Unlearning
- [ ] 2021 S&P oakland [Machine Unlearning](https://arxiv.org/abs/1912.03817)

## Theoretical Analysis
- [ ] 2014 MIPS [On the Number of Linear Regions of Deep Neural Networks](https://arxiv.org/abs/1402.1869)
- [ ] 2020 PNAS [Overparameterized neural networks implement associative memory](https://www.pnas.org/content/117/44/27162)

## Model Hiding
- 2019 NIPS [Superposition of many models into one](https://arxiv.org/abs/1902.05522)
- 2020 ICLR [Once-for-All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791)
- 2020 arxiv [On Hiding Neural Networks Inside Neural Networks](https://arxiv.org/abs/2002.10078)
- 2021 arxiv [Recurrent Parameter Generators](https://arxiv.org/abs/2107.07110)

## Model Design for Efficiency 

### Quantization

- 2017 ICLR [Loss-aware Binarization of Deep Networks](https://arxiv.org/abs/1611.01600)
- 2018 NIPS [Scalable Methods for 8-bit Training of Neural Networks](https://arxiv.org/abs/1805.11046)
- 2018 CVPR [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- 2018 Google White Paper [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342)
- 2018 ICLR [Loss-aware Weight Quantization of Deep Networks](https://arxiv.org/abs/1802.08635)

###  Dynamic Inference
- 2021 IEEE TETC [Fully Dynamic Inference with Deep Neural Networks](https://arxiv.org/abs/2007.15151)

## Neural Network Interpretability
- 2014 ECCV [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
- 2014 ICLR Workshop [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
- 2016 CVPR [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)
- 2017 ICCV [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- 2018 BMVC [RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421)
- [ ] 2019 NeurIPS [Full-Gradient Representation for Neural Network Visualization](https://arxiv.org/abs/1905.00780)


# DNN Application
## Transformer

- 2017 NeurIPS [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Transformer in Computer Vision
- 2018 ICML [Image Transformer](http://proceedings.mlr.press/v80/parmar18a.html)
- 2020 ECCV [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- 2020 ICLR [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- 2021 Arxiv [TransReID: Transformer-based Object Re-Identification](https://arxiv.org/abs/2102.04378)

## Reinforcement Learning

### Design

#### Basic 

- 2015 Nature [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

#### Multi-Agents 

- 2016 ICML [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

### Adversarial Attack on RL

- 2017 ICLR Workshop [Adversarial Attacks on Neural Network Policies](https://arxiv.org/abs/1702.02284)
- 2017 MLDM [Vulnerability of Deep Reinforcement Learning toPolicy Induction Attacks](https://arxiv.org/abs/1701.04143)
- 2017 IJCAI [Tactics of Adversarial Attack on Deep Reinforcement Learning Agents](https://arxiv.org/abs/1703.06748)
- 2018 CSCS [Sequential Attacks on Agents for Long-Term Adversarial Goals](https://arxiv.org/abs/1805.12487)
- 2020 S&P Workshop [On the Robustness of Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2003.03722.pdf)
- 2020 ICML [Policy Teaching via Environment Poisoning: Training-time Adversarial Attacks against Reinforcement Learning](https://arxiv.org/abs/2003.12909)

### Others

- 2017 ICML [Robust Adversarial Reinforcement Learning](https://arxiv.org/abs/1703.02702)

## Person Re-identification

### Survey

- 2020 arxiv [Deep Learning for Person Re-identification: A Survey and Outlook](https://arxiv.org/abs/2001.04193)

### Toolbox

- 2020 arxiv [FastReID: A Pytorch Toolbox for General Instance Re-identification](https://arxiv.org/abs/2006.02631)

### Design

- 2019 CVPR workshop [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://arxiv.org/abs/1903.07071)
- 2015 ICCV [Scalable Person Re-identification: A Benchmark](https://ieeexplore.ieee.org/document/7410490)
- 2016 ECCV workshop [Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking](https://arxiv.org/abs/1609.01775)
- 2016 arxiv [PersonNet: Person Re-identification with Deep Convolutional Neural Networks](https://arxiv.org/abs/1601.07255)

### Adversarial Attack on ReID

- 2018 ECCV [Adversarial Open-World Person Re-Identification](https://arxiv.org/abs/1807.10482)
- 2019 PAMI [Adversarial Metric Attack and Defense for Person Re-identification](https://arxiv.org/abs/1901.10650)
- 2019 ICCV [advPattern: Physical-World Attacks on Deep Person Re-Identification via Adversarially Transformable Patterns](https://arxiv.org/abs/1908.09327)
- 2020 CVPR [Transferable, Controllable, and Inconspicuous Adversarial Attacks on Person Re-identification With Deep Mis-Ranking
](https://arxiv.org/abs/2004.04199)
- 2020 IEEE ACCESS [An Effective Adversarial Attack on Person Re-Identification in Video Surveillance via Dispersion Reduction](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9195855)

# Mathematics
- Blind Signal Separation [Blind signal separation: statistical principles](https://ieeexplore.ieee.org/document/720250)

