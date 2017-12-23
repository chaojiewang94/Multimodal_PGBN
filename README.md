=======================================================================
Mutlimodal Poisson Gamma Belief Network
=======================================================================

This is the code for the paper "Mutlimodal Poisson Gamma Belief Network" presented at AAAI 2018 

Created by Chaojie Wang , Bo Chen at Xidian University and Mingyuan Zhou at University of Texas at Austin

=======================================================================
Introduciton 
=======================================================================

The mPGBN unsupervisedly extracts a nonnegative latent representation using an upward-downward Gibbs sampler. It imposes sparse connections between different layers, making it simple to visualize the generative process and the relationships between the latent features of different modalities. Our experimental results on bi-modal data consisting of images and tags show that the mPGBN can easily impute a missing modality and hence is useful for both image annotation and retrieval. We further demonstrate that the mPGBN achieves state-of-the-art results on unsupervisedly extracting latent features from multimodal data.

=======================================================================
OVERVIEW
=======================================================================

most important files:
- Multimodal_PGBN.m sets all parameters and is the main code of the projects
- Multi_PGBN_Gibbs_Burin_Collect.m includes training phase and testing phase of mPGBN
- flicker_theta_1000_one_vs_all_logsitics_regression_for_mPGBN evaluates the MAP and Pre@50 in flicker 25K
- sampler file includes the whole Gibbs samplers of mPGB.for more detail of these samplers,you can find original codes in following website: https://github.com/mingyuanzhou/GBN
- ficker_data file includes filcker25k download from http://www.cs.toronto.edu/~nitish/multimodal/ and we have translate original data to mat file using flicker_data_split.m 

=======================================================================
LICENSE
=======================================================================

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Copyright (c), 2018, Chaojie Wang 
xd_silly@163.com
=======================================================================
CONTACT
=======================================================================
Contact Bo Chen <bchen@mail.xidian.edu.cn> or Chaojie Wang <xd_silly@163.com>
