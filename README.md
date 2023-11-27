# [Semi-Supervised Adaptation for Skeletal Data Based Human Action Recognition](https://sciforum.net/paper/view/16083)


## Introduction

Recent research on human action recognition is largely facilitated by skeletal data, a compact representation composed of key joints of the human body that is efficiently extracted from 3D imaging sensors and that offers the merit of being robust to variations in the environment. However, leveraging the capabilities of artificial intelligence on such sensory input imposes the collection and annotation of a large volume of skeleton data, which is extremely time-consuming, troublesome and prone to subjectivity in practice. In this paper, a trade-off approach is proposed that utilizes the recent contrastive learning technique to surmount the high requirements imposed by traditional machine learning methods on labeled skeletal data while training a capable human action recognition model. Specifically, the approach is designed as a two-phase semi-supervised learning framework. In the first phase, an unsupervised learning model is trained under a contrastive learning fashion to extract high-level human action activity semantic representations from unlabeled skeletal data. The resulting pre-trained model is then fine-tuned on a small number of properly labeled data in the second phase. The overall strategy helps identify rules for using least amounts of labeled data while achieving a human action recognition model compatible with state-of-the-art performance. The framework integrates the popular graph convolutional neural networks into the proposed semi-supervised learning framework and experimentation is conducted on the large-scale human action recognition dataset, NTU-RGBD. The paper provides comprehensive comparisons between experimental results obtained with the semi-supervised learning model and with fully supervised learning models. Relative usage of labeled data is emphasized to demonstrate the potential of the proposed approach.

## Prerequisites

The code in contrastive learning is heavily rely on ["Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition"](https://github.com/Levigty/AimCLR). Please read this paper if interested. However, the semi-supervised learning protocol presented in this repository is open to different contrastive learning techniques. 

```bash
pip install requirements.txt
```

### Data preparation

* Download [ntu60_frame50](https://drive.google.com/drive/folders/1WrTG9g-dit7RnaXZ6MR5STOiuaEptfuf?usp=drive_link) and install as following structure:
```bash
<root_dir>/data/ntu60_frame50/                                                  
```

* Download [pku_part1_frame50](https://drive.google.com/drive/folders/1pu4P1ZcZvzg70BE1TKW7ro62NO4J0I3E?usp=drive_link) and install as following  structure:
```bash
<root_dir>/data/pku_part1_frame50/                        
```


## 1. Pre-training (contrastive learning) 


 ```
  $ cd <root_dir>
  $ python stage1.py
 ```

### 2. Fine-tuning

  
```
$ cd <root_dir>
$ python stage2.py
```



## Results

Checkpoints can be found under the folder 
 ```
 ./results
 ```

## Acknowledgements
Code in this repository is heavily borrowed from the official PyTorch implementation of ["Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition"](https://github.com/Levigty/AimCLR) in AAAI2022.
Kindly consider cite the paper.