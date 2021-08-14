# HOI_toolbox
High-order interactions toolbox 


This repository includes a Python based toolbox on higher-order interactions such as O-information and dO-information which fills the greatest gap at the moment and  since these measures scale well with the number of time series involved. This work was done by Pranav Mahajan as a part of GSOC-2021 at INCF under the guidance and mentorship of Daniele Marinazzo and Fernando Rosas.


#### The goals and deliverables achieved as a part of this work - 
1. Implement a Python based implementation of Oinfo and dOinfo which computes redundancies and synergies, sorted in decreasing order along with their indices from which one extract the exact combination corresponding to the redundancy or synergy value and also check its significance by using bootstrapping
2. Implements two distinct methods of indices: (1) All n-plet combinations precomputed in advance and stored in memory and (2) Each combination generated iteratively and mapped to a unique and unchanged index using the combinatorial numbering system, this helps extend to higher orders without running out of memory
3. Implements two distinct estimators for computation of entropy and conditional mutual information in Oinfo and dOinfo respectively: (1) Gaussian Copula based estimators (based on Robin Ince's [gcmi](https://github.com/robince/gcmi)) and (2) Covariance matrix based linear estimators (implemented from scratch by refering to [ITS](http://www.lucafaes.net/its.html) and [MuTE](http://www.lucafaes.net/its.html))
4. Dockerize the app for easy use with brainlife.io in the future
5. Rigorously tested on timeseries data with variables less than 20.
6. Additionally also translate [brincolab's Oinfo and Sinfo code](https://github.com/brincolab/High-Order-interactions) to Python (available under alternative_codes) and also use frites backend for gcmi estimators (can be found in frites-exp branch)

#### Drawbacks and opportunities of future work -
1. Currently the Python implementation in this repository is slower than the original MATLAB implementation by Daniele, which uses GCMI estimators and does not use any combinatorial numbering system (thus precomputes the combinations and runs out of memory for higher orders). We think the root cause lies in the scipy special functions used in gcmi computation (Please refer to chech_gcmi_timing.py for more). We arrived at the conclusion that the discrepancy is not in the bootstrapping as a small discrepancy remained even after turning off the bootci. Covariance matrix based linear estimator was implemented in hopes of solving this issue, but they don't seem to speedup substantially. Other approaches which were tried involve using Numba and future work can also try multi-threading to speedup.
2. Because of runtime issues, the algorithm was never tested on timeseries input with more than 20 variables on a laptop. Immediate future work involves, running existing app directly on brainlife.io, after the preprocessing pipeline setup based on https://github.com/faskowit/app-fmri-2-mat/tree/0.1.6
3. The implementation does a full sweep of n-plet size to a predefined maxsize and usually one stops when the informational content is compatible with zero. The other way would be to do a greedy search and can be a possible future work.

#### Key points on difference between this implementation and Brinco lab's -
1. Brinco Lab's implementation only works for continuous variables, but this gcmi based implementation should ideally work with both continuous and discrete variables
2. There are multiple ways of calculating redundancy and synergy, Brinco Lab's implementation calculates for each variable based on all the combinations the variable is a part of. This implementation, based on Daniele's instead finds the redundancy and synergy values of all the combinations and finds the n_best sorted in decreasing order.
3. Brinco Lab's implementation does not implement bootstrapping to check the significance of the redundancy and synergy values whereas this implementation does it.

A weekly progress of the GSOC project can be found in this [document](https://docs.google.com/document/d/1Euvho8-evbD7iP5deRoiehxTfjDJP9EHh3oivtO9cS0/edit?usp=sharing).

#### Relevant papers and resources - 
1. Oinfo: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.032305
2. dOinfo: https://www.frontiersin.org/articles/10.3389/fphys.2020.595736/full
3. Gaussian copula based estimation: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23471
4. Combinatorial Numbering System (CNS): https://en.wikipedia.org/wiki/Combinatorial_number_system#Place_of_a_combination_in_the_ordering
5. Example of the CNS: https://docs.google.com/spreadsheets/d/1A-JTEIu2pMHfYKUtrXUwUxDwllwveWbNJN0l2jB6vkA/edit?usp=sharing
6. Covariance Matrix based estimation of entropy and MI: https://www.math.nyu.edu/~kleeman/infolect7.pdf


# How to use this toolbox?
If you are running on brainlife, it'll create a `config.json` or else if you are runnning locally you'll need to create a `config.json` by looking at `config-sample.json`. You can either run the toolbox by running the docker by just running `./main`, which would setup the necessary environment and run `main.py` with `config.json` as the argument or else you could install the libraries from `requirements.txt` and run `main.py`. You will need to mention things like path to the input data and it's datatype and all the arguments for the Oinfo and dOinfo code in the `config.json` before hand.

# How to read and interpret the outputs? 
Please refer to `read_outputs.py` for a detailed walkthrough through examples! It uses some sample outputs already generated and saved in `outputs` folder.
