==
Mutlimodal Poisson Gamma Belief Network
==

This is the code for the paper "Mutlimodal Poisson Gamma Belief Network" presented at AAAI 2018 

Created by Chaojie Wang , Bo Chen at Xidian University and Mingyuan Zhou at University of Texas at Austin

==
Introduciton 
==

The mPGBN unsupervisedly extracts a nonnegative latent representation using an upward-downward Gibbs sampler. It imposes sparse connections between different layers, making it simple to visualize the generative process and the relationships between the latent features of different modalities. Our experimental results on bi-modal data consisting of images and tags show that the mPGBN can easily impute a missing modality and hence is useful for both image annotation and retrieval. We further demonstrate that the mPGBN achieves state-of-the-art results on unsupervisedly extracting latent features from multimodal data.

==
OVERVIEW
==

most important files:
- Multimodal_PGBN.m sets all parameters and is the main code of the projects
- Multi_PGBN_Gibbs_Burin_Collect.m includes training phase and testing phase of mPGBN
- flicker_theta_1000_one_vs_all_logsitics_regression_for_mPGBN evaluates the MAP and Pre@50 in flicker 25K
- sampler file includes the whole Gibbs samplers of mPGB.for more detail of these samplers,you can find original codes in following website: https://github.com/mingyuanzhou/GBN
- ficker_data file includes filcker25k download from http://www.cs.toronto.edu/~nitish/multimodal/ and we have translate original data to mat file using flicker_data_split.m 

Copyright (c), 2018, Chaojie Wang 
xd_silly@163.com
==
CONTACT
==
Contact Bo Chen <bchen@mail.xidian.edu.cn> or Chaojie Wang <xd_silly@163.com>
