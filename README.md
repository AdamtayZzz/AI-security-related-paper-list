# AI Security Paper List
- [Attack-Related](#attack-related)
  * [Adversarial Attack](#adversarial-attack)
    + [Survey](#survey)
    + [Attack-CV](#attack-cv)
    + [Defense-Robustness](#defense-robustness)
    + [Defense-Detection](#defense-detection)
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
    + [Model Fingerprint](#model-fingerprint)
    + [Dataset Fingerprint](#dataset-fingerprint)
  * [Watermarking](#watermarking)
    + [Model Watermarking](#model-watermarking)
    + [Dataset Watermarking](#dataset-watermarking)
  * [Machine Unlearning](#machine-unlearning)
  * [Theoretical Analysis](#theoretical-analysis)
  * [Model Hiding](#model-hiding)
  * [Model Design for Efficiency](#model-design-for-efficiency)
    + [Quantization](#quantization)
    + [Dynamic NN](#dynamic-nn)
      - [Security-related](#security-related)
  * [Neural Network Interpretability](#neural-network-interpretability)
    + [Analysis](#analysis)
    + [Attack](#attack)
    + [Defense](#defense-1)
- [DNN Application](#dnn-application)
  * [GAN & AE](#gan---ae)
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
  * [Face Recognition](#face-recognition)
    + [Survey](#survey-3)
    + [Anti FR](#anti-fr)
- [Mathematics](#mathematics)
  * [Probabilistic Graphical Model](#probabilistic-graphical-model)
  * [Convex Optimization](#convex-optimization)
  * [Manifold](#manifold)
  * [Learning Theory](#learning-theory)
  * [others](#others)

# Attack-Related

## Adversarial Attack

### Survey

- 2018 IEEE ACCESS [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://arxiv.org/abs/1801.00553)
- 2020 Engineering [Adversarial Attacks and Defenses in Deep Learning](https://www.sciencedirect.com/science/article/pii/S209580991930503X)
- 2021 arxiv [Adversarial Example Detection for DNN Models: A Review and Experimental Comparison](https://arxiv.org/abs/2105.00203)
- 2021 arxiv [Advances in adversarial attacks and defenses in computer vision: A survey](https://arxiv.org/abs/2108.00401)

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
- 2018 ICDM [Query-Efficient Black-Box Attack by Active Learning](https://arxiv.org/abs/1809.04913)
- 2021 CVPR [Simulating Unknown Target Models for Query-Efficient Black-box Attacks](https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Simulating_Unknown_Target_Models_for_Query-Efficient_Black-Box_Attacks_CVPR_2021_paper.pdf)
- 2022 CVPR [Improving Adversarial Transferability via Neuron Attribution-Based Attacks](https://arxiv.org/abs/2204.00008)
- 2022 IJCAI [A Few Seconds Can Change Everything: Fast Decision-based Attacks against DNNs](https://www.ijcai.org/proceedings/2022/464)

### Defense-Robustness

- 2015 arxiv [Foveation-based Mechanisms Alleviate Adversarial Examples](https://arxiv.org/abs/1511.06292)
- 2016 CVPR [A study of the effect of JPG compression on adversarial images](https://arxiv.org/abs/1608.00853)
- 2016 S&P [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](https://arxiv.org/abs/1511.04508)
- 2017 ICCV [Adversarial Examples for Semantic Segmentation and Object Detection](https://arxiv.org/abs/1703.08603)
- 2017 arxiv [Mitigating adversarial effects through randomization](https://arxiv.org/abs/1711.01991)
- 2017 arxiv [DeepCloak: Masking Deep Neural Network Models for Robustness Against Adversarial Samples](https://arxiv.org/abs/1702.06763)
- 2017 arxiv [Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN](https://arxiv.org/abs/1705.03387)
- 2018 AAAI [Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients](https://arxiv.org/abs/1711.09404)
- 2018 CVPR [Defense against Universal Adversarial Perturbations](https://arxiv.org/abs/1711.05929)
- 2019 NuerIPS [Adversarial Examples are not Bugs, they are Features](https://proceedings.neurips.cc/paper/2019/file/e2c420d928d4bf8ce0ff2ec19b371514-Paper.pdf)
- [ ] 2019 NeurIPS [Certified Adversarial Robustness with Additive Noise](https://arxiv.org/abs/1809.03113)
- 2020 NeurIPS [Adversarial Self-Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/file/1f1baa5b8edac74eb4eaa329f14a0361-Paper.pdf)
- 2020 ICML [Adversarial Neural Pruning with Latent Vulnerability Suppression](https://arxiv.org/abs/1908.04355)
- [ ] 2021 arxiv [Meta Adversarial Training against Universal Patches](https://arxiv.org/abs/2101.11453)
- 2020 CVPR [Adversarial Robustness- From Self-Supervised Pre-Training to Fine-Tuning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Adversarial_Robustness_From_Self-Supervised_Pre-Training_to_Fine-Tuning_CVPR_2020_paper.pdf)
- 2021 ICCV [Adversarial Attacks are Reversible with Natural Supervision](https://arxiv.org/abs/2103.14222)
- 2021 ICLR [Self-supervised adversarial robustness for the low-label, high-data regime](https://openreview.net/forum?id=bgQek2O63w)
- 2021 NeurIPS [When Does Contrastive Learning Preserve Adversarial Robustness from Pretraining to Finetuning?](https://arxiv.org/abs/2111.01124)
- 2022 USENIX [Transferring Adversarial Robustness Through Robust Representation Matching](https://arxiv.org/abs/2202.09994)

### Defense-Detection

- 2017 ICLR [On Detecting Adversarial Perturbations](https://arxiv.org/abs/1702.04267)
- 2017 Arixv [Detecting Adversarial Samples from Artifacts](https://arxiv.org/abs/1703.00410)
- 2017 ICCV [SafetyNet: Detecting and Rejecting Adversarial Examples Robustly](https://arxiv.org/abs/1704.00103)
- 2017 CCS [MagNet: a Two-Pronged Defense against Adversarial Examples](https://arxiv.org/abs/1705.09064)
- 2018 ICLR [Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality](https://arxiv.org/abs/1801.02613)
- 2018 arxiv [Detecting Adversarial Perturbations with Saliency](https://arxiv.org/abs/1803.08773)
- 2018 NDSS [Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks](https://arxiv.org/abs/1704.01155)
- 2019 NDSS [NIC: Detecting Adversarial Samples with Neural Network Invariant Checking](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf)
- 2019 arxiv [Model-based Saliency for the Detection of Adversarial Examples](https://openreview.net/pdf?id=HJe5_6VKwS)
- 2020 ICJNN [Detection of Adversarial Examples in Deep Neural Networks with Natural Scene Statistics](https://ieeexplore.ieee.org/document/9206959)
- 2020 arxiv [Detection Defense Against Adversarial Attacks with Saliency Map](https://arxiv.org/abs/2009.02738)
- 2020 arxiv [RAID: Randomized Adversarial-Input Detection for Neural Networks](https://arxiv.org/abs/2002.02776)
- 2020 AAAI [ML-LOO: Detecting Adversarial Examples with Feature Attribution](https://arxiv.org/abs/1906.03499)
- 2021 Springer [Adversarial example detection based on saliency map features](https://link.springer.com/article/10.1007/s10489-021-02759-8)
- 2020 KDD [Interpretability is a Kind of Safety: An Interpreter-based Ensemble for Adversary Defense](https://dl.acm.org/doi/abs/10.1145/3394486.3403044)
- 2020 SPAI [Stateful Detection of Black-Box Adversarial Attacks](https://dl.acm.org/doi/10.1145/3385003.3410925)
- 2021 arxiv [ExAD: An Ensemble Approach for Explanation-based Adversarial Detection](https://arxiv.org/abs/2103.11526)
- 2022 USENIX [Blacklight: Scalable Defense for Neural Networks against Query-Based Black-Box Attacks](https://arxiv.org/abs/2006.14042)

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
- 2022 IJCAI [Data-Efficient Backdoor Attacks](https://arxiv.org/abs/2204.12281)
- 2022 AAAI [Hibernated Backdoor- A Mutual Information Empowered Backdoor Attack to Deep Neural Networks](https://aaai-2022.virtualchair.net/poster_aaai6346)
- 2022 CVPR [Backdoor Attacks on Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Saha_Backdoor_Attacks_on_Self-Supervised_Learning_CVPR_2022_paper.pdf)
- 2022 CVPR [BppAttack: Stealthy and Efficient Trojan Attacks against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning](https://arxiv.org/abs/2205.13383)


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
- 2021 NeurIPS [Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://arxiv.org/abs/2110.11571)
- 2021 NeurIPS [Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://arxiv.org/abs/2110.14430)
- 2022 USENIX [Poison Forensics: Traceback of Data Poisoning Attacks in Neural Networks](https://arxiv.org/abs/2110.06904)

## Data Poisoning

- 2017 ICML [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)
- 2018 NeurIPS [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)
- 2021 NeurIPS [Manipulating SGD with Data Ordering Attacks](https://arxiv.org/pdf/2104.09667.pdf)
- 2022 AAAI [CLPA: Clean-Label Poisoning Availability Attacks Using Generative Adversarial Nets](https://aaai-2022.virtualchair.net/poster_aaai3872)
- 2022 USENIX [PoisonedEncoder-Poisoning the Unlabeled Pre-training Data in Contrastive Learning](http://arxiv.org/abs/2205.06401)


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
- 2020 Arxiv [iDLG: Improved Deep Leakage from Gradients](https://arxiv.org/abs/2001.02610)
- 2020 USENIX Security [Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning](https://arxiv.org/abs/1904.01067)
- 2020 CVPR [Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://arxiv.org/abs/1912.08795)
- 2021 CVPR [See through Gradients: Image Batch Recovery via GradInversion](https://arxiv.org/abs/2104.07586)
- 2022 USENIX Security [Theory-Oriented Deep Leakage from Gradients via Linear Equation Solver](https://arxiv.org/abs/2010.13356)
- 2022 Arxiv [Reconstructing Training Data from Trained Neural Networks](https://arxiv.org/abs/2206.07758)

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
- 2022 USENIX [Membership Inference Attacks and Defenses in Neural Network Pruning](https://arxiv.org/abs/2202.03335)
- 2020 S&P [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570)
- 2021 AsiaCCS [Membership Feature Disentanglement Network](https://dl.acm.org/doi/abs/10.1145/3488932.3497772)
- 2022 Arxiv [Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets](https://arxiv.org/abs/2204.00032)
- 2022 ECCV [Semi-Leak- Membership Inference Attacks Against Semi-supervised Learning](https://arxiv.org/abs/2207.12535)
- 2022 TDSC [Membership Inference Attacks against Machine Learning Models via Prediction Sensitivity](https://ieeexplore.ieee.org/document/9793586/)

## Property Inference Attack

- 2018 CCS [Property Inference Attacks on Fully Connected Neural Networksusing Permutation Invariant Representations](https://dl.acm.org/doi/10.1145/3243734.3243834)

## Others
- 2017 CCS [Machine Learning Models that Remember Too Much](https://arxiv.org/abs/1709.07886)
- 2022 ICML [On the Difficulty of Defending Self-Supervised Learning against Model Extraction](https://arxiv.org/abs/2205.07890)

# Machine Learning Related
## Fingerprint
### Model Fingerprint
- 2019 AsiaCCS [IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary](https://arxiv.org/abs/1910.12903)
- 2019 CVPR [Sensitive-Sample Fingerprinting of Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.pdf)
- 2020 Computer Communications [AFA: Adversarial fingerprinting authentication for deep neural networks](https://www.sciencedirect.com/science/article/pii/S014036641931686X)
- 2020 ACSAC [Secure and Verifiable Inference in Deep Neural Networks](https://dl.acm.org/doi/10.1145/3427228.3427232)
- 2021 ESORICS [TAFA: A Task-Agnostic Fingerprinting Algorithm for Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-88418-5_26)
- 2021 ICLR [Deep Neural Network Fingerprinting by Conferrable Adversarial Examples](https://arxiv.org/abs/1912.00888)
- 2021 ISCAS [Fingerprinting Deep Neural Networks - a DeepFool Approach](https://ieeexplore.ieee.org/document/9401119)
- 2021 ISSTA [ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection](https://arxiv.org/abs/2106.08890)
- 2021 BMVC [Intrinsic Examples: Robust Fingerprinting of Deep Neural Networks](https://www.bmvc2021-virtualconference.com/assets/papers/0625.pdf)
- 2021 Journal of Computer Research and Development [An Evasion Algorithm to Fool Fingerprint Detector for Deep Neural Networks](https://crad.ict.ac.cn/CN/10.7544/issn1000-1239.2021.20200903)
- 2022 S&P [Copy, Right? A Testing Framework for Copyright Protection of Deep Learning Models](https://arxiv.org/abs/2112.05588)
- 2022 USENIX [Teacher Model Fingerprinting Attacks Against Transfer Learning](https://arxiv.org/abs/2106.12478)
- 2022 CVPR [Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations](https://arxiv.org/abs/2202.08602)
- 2022 IJCAI [MetaFinger: Fingerprinting the Deep Neural Networks with Meta-training](https://www.ijcai.org/proceedings/2022/0109.pdf)
- 2022 ICIP [Neural network fragile watermarking with no model performance degradation](https://arxiv.org/abs/2208.07585)
- 2022 TIFS [A DNN Fingerprint for Non-Repudiable Model Ownership Identification and Piracy Detection](https://ieeexplore.ieee.org/document/9854806/)

### Dataset Fingerprint
- 2022 IEEE TIFS [Your Model Trains on My Data? Protecting Intellectual Property of Training Data via Membership Fingerprint Authentication](https://ieeexplore.ieee.org/document/9724248)

## Watermarking
### Model Watermarking
- 2017 ICMR [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/abs/1701.04082)
- [ ] 2018 USENIX [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://arxiv.org/abs/1802.04633)
- 2019 ASPLOS [DeepSigns: An End-to-End Watermarking Framework for Ownership Protection of Deep Neural Networks](https://dl.acm.org/doi/10.1145/3297858.3304051)\
- 2019 ICASSP [Attacks on Digital Watermarks for Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/8682202/)
- 2020 AsiaCCS [Robust Membership Encoding: Inference Attacks and Copyright Protection for Deep Learning](https://dl.acm.org/doi/abs/10.1145/3320269.3384731)
- 2021 WWW [RIGA: Covert and Robust White-BoxWatermarking of Deep Neural Networks](https://arxiv.org/abs/1910.14268)
- 2021 KSEM [Fragile Neural Network Watermarking with Trigger Image Set](https://link.springer.com/chapter/10.1007/978-3-030-82136-4_23)
- 2022 arxiv [AWEncoder: Adversarial Watermarking Pre-trained Encoders in Contrastive Learning](https://arxiv.org/abs/2208.03948)
- 2022 AAAI [DeepAuth: A DNN Authentication Framework by Model-Unique and Fragile Signature Embedding](https://www.aaai.org/AAAI22Papers/AAAI-3901.LaoY.pdf) 

### Dataset Watermarking
- 2020 NeurIPS Workshop [Open-sourced Dataset Protection via Backdoor Watermarking](https://arxiv.org/abs/2010.05821)
- 2020 ICML [Radioactive data: tracing through training](http://proceedings.mlr.press/v119/sablayrolles20a.html)

## Machine Unlearning
- 2019 NeurIPS [Making AI Forget You: Data Deletion in Machine Learning](https://arxiv.org/abs/1907.05012)
- 2020 MICCAI [Have you forgotten? A method to Assess If Machine Learning Models Have Forgotten Data](https://arxiv.org/abs/2004.10129)
- 2020 CVPR [Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks](https://arxiv.org/abs/1911.04933)
- 2021 ECCV [Forgetting Outside the Box- Scrubbing Deep Networks of Information Accessible from Input-Output Observations](https://arxiv.org/abs/2003.02960)
- 2021 S&P Oakland [Machine Unlearning](https://arxiv.org/abs/1912.03817)
- 2021 AAAI [Amnesiac Machine Learning](https://arxiv.org/abs/2010.10981)
- 2021 MICCAI [EMA: Auditing Data Removal from Trained Model](https://arxiv.org/abs/2109.03675)
- 2021 ICML [Certified Data Removal from Machine Learning Models](https://arxiv.org/abs/1911.03030)
- 2022 TDSC [Learn to Forget: Machine Unlearning via Neuron Masking](https://arxiv.org/abs/2003.10933)
- 2022 AAAI [PUMA: Performance Unchanged Model Augmentation for Training Data Removal](https://www.aaai.org/AAAI22Papers/AAAI-10608.WuG.pdf)
- 2022 AAAI [Hard to Forget: Poisoning Attacks on Certified Machine Unlearning](https://arxiv.org/abs/2109.08266)
- 2022 Arxiv [A Survey of Machine Unlearning](https://arxiv.org/abs/2209.02299)
- 2022 Arxiv [Zero-shot machine unlearning](https://arxiv.org/abs/2201.05629)
- 2022 ECCV [Learning with Recoverable Forgetting](https://arxiv.org/abs/2207.08224)
- 2022 USENIX [On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning](https://arxiv.org/abs/2110.11891)


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

###  Dynamic NN
- 2017 CVPR [Spatially Adaptive Computation Time for Residual Networks](https://arxiv.org/abs/1612.02297)
- 2018 ICLR [Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/abs/1703.09844)
- 2018 ECCV [SkipNet: Learning Dynamic Routing in Convolutional Networks](https://arxiv.org/abs/1711.09485)
- 2019 ICML [Shallow-Deep Networks: Understanding and Mitigating Network Overthinking](https://arxiv.org/abs/1810.07052)
- 2021 PAMI [Dynamic Neural Networks: A Survey](https://arxiv.org/abs/2102.04906)
- 2021 IEEE TETC [Fully Dynamic Inference with Deep Neural Networks](https://arxiv.org/abs/2007.15151)

#### Security-related
- 2020 ICLR [Triple Wins: Boosting Accuracy, Robustness and Efficiency Together by Enabling Input-Adaptive Inference](https://arxiv.org/abs/2002.10025)
- 2020 CVPR [ILFO: Adversarial Attack on Adaptive Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Haque_ILFO_Adversarial_Attack_on_Adaptive_Neural_Networks_CVPR_2020_paper.pdf)
- 2021 ICLR [A Panda? No, It's a Sloth: Slowdown Attacks on Adaptive Multi-Exit Neural Network Inference](https://arxiv.org/abs/2010.02432)
- 2021 IEEE IOTJ [DefQ: Defensive Quantization against Inference Slow-down Attack for Edge Computing](https://ieeexplore.ieee.org/document/9664815)
- 2021 Arxiv [Fingerprinting Multi-exit Deep Neural Network Models via Inference Time](https://arxiv.org/abs/2110.03175)
- 2022 CCS [Auditing Membership Leakages of Multi-Exit Networks](https://arxiv.org/abs/2208.11180)
- 2022 ICSE [EREBA: Black-box Energy Testing of Adaptive Neural Networks](https://arxiv.org/abs/2202.06084)


## Neural Network Interpretability
- 2014 ECCV [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
- 2014 ICLR Workshop [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
- 2016 CVPR [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)
- 2017 ICCV [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- 2018 BMVC [RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421)
- [ ] 2019 NeurIPS [Full-Gradient Representation for Neural Network Visualization](https://arxiv.org/abs/1905.00780)
- [ ] 2020 IJCNN [Black-Box Saliency Map Generation Using Bayesian Optimisation](https://arxiv.org/abs/2001.11366)
- 2021 CVPR [Black-box Explanation of Object Detectors via Saliency Maps](https://arxiv.org/abs/2006.03204)
- 2021 Arxiv [Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability](https://arxiv.org/abs/2108.01335)
- 2022 ECCV [SESS：Saliency Enhancing with Scaling and Sliding](https://arxiv.org/abs/2207.01769)
- 2022 AAAI [Interpretable Generative Adversarial Networks](https://ojs.aaai.org/index.php/AAAI/article/view/20015)

### Analysis
- 2017 Arxiv [The (Un)reliability of saliency methods](https://arxiv.org/abs/1711.00867)
- 2018 NeurlPS [Sanity Checks for Saliency Maps](https://papers.nips.cc/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf)
- 2019 NeurlPS [On the (In)fidelity and Sensitivity for Explanations](https://arxiv.org/abs/1901.09392)
- 2022 ICLR Workshop [Saliency Maps Contain Network "Fingerprints"](https://openreview.net/pdf?id=SWlpXB_bWc)
### Attack 
- 2019 AAAI [Interpretation of Neural Networks Is Fragile](https://arxiv.org/abs/1710.10547)
- 2019 ICCV [Fooling Network Interpretation in Image Classification](https://arxiv.org/abs/1812.02843)
- 2019 NeurlPS [Fooling Neural Network Interpretations via Adversarial Model Manipulation](https://arxiv.org/abs/1902.02041)
- 2019 NeurlPS [Explanations can be manipulated and geometry is to blame](https://arxiv.org/abs/1906.07983)
- 2020 USENIX [Interpretable Deep Learning under Fire](https://arxiv.org/abs/1812.00891)
### Defense
- 2020 arxiv [A simple defense against adversarial attacks on heatmap explanations](https://arxiv.org/pdf/2007.06381.pdf)
- 2022 arxiv [Defense Against Explanation Manipulation](https://arxiv.org/abs/2111.04303)


# DNN Application
## GAN & AE
- 2018 arxiv [Anomaly Detection for Skin Disease Images Using Variational Autoencoder](https://arxiv.org/abs/1807.01349) 

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
- 2020 ECCV [Adversarial T-shirt! Evading Person Detectors in A Physical World](https://arxiv.org/abs/1910.11099)

## Face Recognition
### Survey
- 2021 Arxiv [SoK: Anti-Facial Recognition Technology](https://arxiv.org/abs/2112.04558)
### Anti FR
- 2016 CCS [Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition](https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf)
- 2019 ISVC [DeepPrivacy: A Generative Adversarial Network for Face Anonymization](https://arxiv.org/abs/1909.04538)
- 2019 CVPR [Efficient Decision-based Black-box Adversarial Attacks on Face Recognition](https://arxiv.org/abs/1904.04433)
- 2019 IMWUT [VLA: A Practical Visible Light-based Attack on Face Recognition Systems in Physical World](https://dl.acm.org/doi/10.1145/3351261)
- 2020 CVPR Workshop [Adversarial Light Projection Attacks on Face Recognition Systems: A Feasibility Study](https://arxiv.org/abs/2003.11145)
- 2020 USENIX [Fawkes: Protecting Privacy against Unauthorized Deep Learning Models](https://arxiv.org/abs/2002.08327)
- 2021 ICCV [Towards Face Encryption by Generating Adversarial Identity Masks](https://arxiv.org/abs/2003.06814)
- 2021 ICLR [LowKey: Leveraging Adversarial Attacks to Protect Social Media Users from Facial Recognition](https://arxiv.org/abs/2101.07922)
- 2021 ICLR [Unlearnable Examples: Making Personal Data Unexploitable](https://arxiv.org/abs/2101.04898)


# Mathematics
## Probabilistic Graphical Model
- 1999 SIGIR [Probabilistic Latent Semantic Indexing](https://dl.acm.org/doi/10.1145/312624.312649)
- 2007 NeurIPS[Probabilistic Matrix Factorization](https://dl.acm.org/doi/10.5555/2981562.2981720)
- 2017 AAAI[SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)
- 2017 ICML [Toward Controlled Generation of Text](https://arxiv.org/abs/1703.00955)

## Convex Optimization
- 2014 [Convex Optimization: Algorithms and Complexity](https://arxiv.org/abs/1405.4980)
- 2019 AAAI [AutoZOOM: Autoencoder-based Zeroth Order Optimization Method for Attacking Black-box Neural Networks](https://arxiv.org/pdf/1805.11770.pdf)

## Manifold
- [Optimization Algorithms on Matrix Manifolds](https://press.princeton.edu/absil)

## Learning Theory
- 2014 NeurIPS [Learning, Regularization and Ill-Posed Inverse Problems](https://proceedings.neurips.cc/paper/2004/file/33267e5dc58fad346e92471c43fcccdc-Paper.pdf)
- 2015 ICML Workshop [Norm-Based Capacity Control in Neural Networks](https://arxiv.org/abs/1503.00036)

## others

- Blind Signal Separation [Blind signal separation: statistical principles](https://ieeexplore.ieee.org/document/720250)

