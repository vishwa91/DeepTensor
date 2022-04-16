**DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors**

**Paper**: https://arxiv.org/abs/2204.03145

**Contents**
- data/ -- contains mat files required to run some python scripts. The folder contains a README listing the sources
	- cat_video.mat -- required for run_figure11.py and run_figure11_TV.py		
	- icvl.mat -- required for run_figure9.py and run_figure9_bm3d.py
	- pet.mat -- required for run_figure12.py and run_figure12_TV.py
	- weizzman.mat -- required for run_figure2.py
- modules/ -- contains several python scripts required for all experiments
	- deep_models/ -- folder downloaded from https://github.com/DmitryUlyanov/deep-image-prior
	- cosine_annealing_with_warmup.py -- code for cosine annealing scheduler downloaded from  https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
	- deep_decoder.py -- convenience code downloaded from https://github.com/reinhardh/supplement_deep_decoder
	- deep_prior.py -- convenience code downloaded from https://github.com/DmitryUlyanov/deep-image-prior
	- lin_inverse.py -- implements functions required for linear inverse problems
	- losses.py -- implements losses required for training include L1, L2, and 2D TV
	- models.py -- implements multilayer perceptron, and multi-dimensional (1D, 2D, and 3D) U-Net
	- spectral.py -- required for run_figure6.py
	- utils.py -- miscellaneous utilities
- requirements.txt -- contains a list of requirements to run the code base. Install the required packages using `pip install -r requirements.txt`
- run_figure2.py -- run this to replicate results in figure 2
- run_figure6.py -- run this to replicate results in figure 6
- run_figure7.py -- run this to replicate results in figure 7
- run_figure9.py -- run this to replicate results with DeepTensor and SVD in figure 9
- run_figure9_bm3d.py -- run this to replicate results with BM3D in figure 9
- run_figure11.py -- run this to replicate results with DeepTensor in figure 11
- run_figure11_TV.py -- run this to replicate results with TV in figure 11
- run_figure12.py -- run this to replicate results with DeepTensor in figure 12
- run_figure12_TV.py -- run this to replicate results with TV in figure 12
- run_table1.py -- run this to replicate results in table 1
