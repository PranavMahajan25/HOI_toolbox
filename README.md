# HOI_toolbox
High-order interactions toolbox 


This repository includes a unified Python-based toolbox to compute higher-order interactions with metrics such as [O-information](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.032305) and [dO-information](https://www.frontiersin.org/articles/10.3389/fphys.2020.595736/full). These measures scale well with the number of time series involved in comparison to most other metrics proposed so far to distinguish high-order features in the data. 

## Contents
1. Introduction to high-order interactions
2. What does the toolbox do? and how to use it?
3. The past, the present and the future of the project


# 1. Introduction to high-order interactions

### What are high-order interactions and why should one use these measures?

The functioning of complex systems (i.e. the brain, and many others) depends on the interaction between different units; crucially, the resulting dynamics is different from the sum of the dynamics of the parts. What allows these systems to be more than the sum of their parts is not in the nature of the parts, but in the
structure of their interdependencies. In order to deepen our understanding of these systems, we need to make sense of these interdependencies. Several tools and frameworks have been developed to look at different statistical dependencies among multivariate datasets. Among these, information theory offers a powerful and versatile framework; notably, it allows detecting high-order interactions that determine the joint informational role of a group of variables.

Now these interactions amongst variables can either be synergestic or redundant, and O-information and dO-information provide us with scalable metrics which are capable of characterising synergy- and redundancy-dominated systems and whose computational complexity scales gracefully with system size, making it suitable for practical data
analysis. 

### What is redundancy and synergy?

Concepts like redundancy and synergy can be understood well intuitively, but in order to understand them at a greater detail (using mathematical expressions) one might need to first familiarize with negentropy, collective constraints and shared randomness. For more details, one can refer to the section II (fundamentals) from the [paper](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.032305) which introduced O-information. This section would walk one through how the total information <img src="https://latex.codecogs.com/gif.latex?(\sum_{j=1}^nlog|\chi_j|)" /> of a system <img src="https://latex.codecogs.com/gif.latex?\mathbf{X}^n=(X_1,...,X_n)" />  is divided by a given state of knowledge (i.e. a probability distribution) into what is determined by the constraints <img src="https://latex.codecogs.com/gif.latex?(\mathcal{N}(\mathbf{X}^n))" /> and what is not instantiated until an actual measurement takes place <img src="https://latex.codecogs.com/gif.latex?(H(\mathbf{X}^n))" />. Moreover, both terms can be further decomposed into their individual and collective components, yielding different perspectives on interdependency seen as either collective constraints <img src="https://latex.codecogs.com/gif.latex?(C(\mathbf{X}^n))" /> or shared randomness <img src="https://latex.codecogs.com/gif.latex?(B(\mathbf{X}^n))" />.

![image](https://user-images.githubusercontent.com/33349653/129573884-a062ac2b-85ac-4d1d-831d-c76a461933ff.png)

In short, if the interdependencies can be more effeciently explained as shared randomness then we call the system to be redundancy-dominated, or else if the interdependencies can be more effeciently explained as collective constraints then we call the system to be synergy-dominated. We would eventually see that this can be extracted based on whether O-info or dO-info of the system is greater than zero (redundancy-dominated) or less than zero (synergy-dominated).

### What is O-information and dO-information?
The O-information (shorthand for "information about Organisational structure") of the system described by the vector of n random variables <img src="https://latex.codecogs.com/gif.latex?\mathbf{X}" /> (minor change in notation, dropping the superscript "n" for convenience) is defined as -

<img src="https://latex.codecogs.com/gif.latex?\Omega(\mathbf{X})=C(\mathbf{X})-B(\mathbf{X})" />
<img src="https://latex.codecogs.com/gif.latex?\Omega(\mathbf{X})=(n-2)H(\mathbf{X})+\sum_{j=1}^n[H(X_j)-H(\mathbf{X}\setminus{X}_j)]" />

where <img src="https://latex.codecogs.com/gif.latex?H" /> stands for the entropy, and <img src="https://latex.codecogs.com/gif.latex?\mathbf{X}\setminus{X}_j" /> is the complement of <img src="https://latex.codecogs.com/gif.latex?X_j" /> with respect to <img src="https://latex.codecogs.com/gif.latex?\mathbf{X}" />

Now further if we add a stochastic variable Y to the set of X variables and extend it to measure the character of the information flow from the X variables to the target Y and further condition on the state vector of the target variable in order to remove shared information due to common history and input signals, then we arrive at dynamic O-information or dO-information from the group of variables X to the target series Y, defined as -

<img src="https://latex.codecogs.com/gif.latex?d\Omega=(1-n)I(Y;\mathbf{X}|Y_0)+\sum_{j=1}^nI(Y;\mathbf{X}\setminus{X}_j))" />

where <img src="https://latex.codecogs.com/gif.latex?I" /> stands for the mutual information and the state vectors are of order <img src="https://latex.codecogs.com/gif.latex?m" />.

For further details, readers are urged to read the papers on [O-information](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.032305) and [dO-information](https://www.frontiersin.org/articles/10.3389/fphys.2020.595736/full)

# 2. What does the toolbox do? and how to use it?

### What are the main deliverables of this toolbox?

The goal of this toolbox is to collect and refine existing metrics of high-order interactions currently implemented in Matlab or Java, and integrate them in a unified python-based toolbox. The main deliverable of the project is a toolbox, whose inputs are the measurements of different variables plus some parameters, and whose outputs are the measures of higher-order interdependencies.

The input is usually a timeseries, currently .mat and .tsv formats are supported by the toolbox. The `exhaustive_loop_zerolag` function from `toolbox.Oinfo` and `exhaustive_loop_lagged` from `toolbox.dOinfo` takes a timeseries of shape (number of variables, number of timepoints). If running locally, feel free to use either of the .mat or .tsv input formats, but if running on brainlife .tsv format will be recommended as it matches the default timeseries datatype in .tsv.gz. Note that you wouldn't need to seperately run the functions `exhaustive_loop_zerolag` or `exhaustive_loop_lagged` as `main.py` would do the preprocessing (such as removing the columns with all zero timeseries and generate `data/cleaned_timeseries.tsv` in case of .tsv input) and then pass it on to these functions. These functions will return a dictionary output which will be saved in pickle format. Ideally reading or writing to pickle shouldn't give any errors, but just in case it does, then try using pickle5 instead of pickle in utils.py. The main outputs would be an array of sorted redundancy and synergy values, the indices of the variable combinations that gave those values and the bootstrap signifiance for each of those values.


### How to use this toolbox?
If you are running on brainlife, it'll create a `config.json` or else if you are runnning locally you'll need to create a `config.json` by looking at `config-sample.json`. You can either run the toolbox by running the docker by just running `./main`, which would setup the necessary environment and run `main.py` with `config.json` as the argument or else you could install the libraries from `requirements.txt` and run `main.py`. You will need to mention things like path to the input data and its datatype and all the arguments for the Oinfo and dOinfo code in the `config.json` beforehand.

### How to read and interpret the outputs? 
Please refer to `read_outputs.py` for a detailed walkthrough through examples! It uses some sample outputs already generated and saved in `outputs` folder.



# 3. The past, the present and the future of this project

This work was done by Pranav Mahajan as a part of GSOC-2021 at INCF under the guidance and mentorship of [Daniele Marinazzo](https://users.ugent.be/~dmarinaz/) and [Fernando Rosas](https://www.imperial.ac.uk/people/f.rosas).


#### The goals and deliverables achieved as a part of this work - 
1. Implement a Python based implementation of Oinfo and dOinfo which computes redundancies and synergies, sorted in decreasing order along with their indices from which one can extract the exact combination of variables composing the multiplet with a specific redundancy or synergy value and also check its significance by using bootstrapping
2. Implement two distinct methods of indexing: (1) All n-plet combinations precomputed in advance and stored in memory and (2) Each combination generated iteratively and mapped to a unique and unchanged index using the combinatorial numbering system, this helps extend to higher orders without running out of memory
3. Implements two distinct estimators for computation of entropy and conditional mutual information in Oinfo and dOinfo respectively: (1) Gaussian Copula based estimators (based on Robin Ince's [gcmi](https://github.com/robince/gcmi)) and (2) Covariance matrix based linear estimators (implemented from scratch by refering to [ITS](http://www.lucafaes.net/its.html) and [MuTE](http://www.lucafaes.net/its.html))
4. Dockerize the app for easy use with brainlife.io in the future
5. Rigorously tested on timeseries data with variables less than 20.
6. Additionally also translate [brincolab's Oinfo and Sinfo code](https://github.com/brincolab/High-Order-interactions) to Python (available under alternative_codes) and also use [frites](https://github.com/brainets/frites) backend for gcmi estimators (can be found in frites-exp branch)

#### Drawbacks and opportunities of future work -
1. Currently the Python implementation in this repository is slower than the original MATLAB implementation by Daniele, which uses GCMI estimators and does not use any combinatorial numbering system (thus precomputes the combinations and runs out of memory for higher orders). We think the root cause lies in the scipy special functions used in gcmi computation (Please refer to chech_gcmi_timing.py for more). We arrived at the conclusion that the discrepancy is not in the bootstrapping as a small discrepancy remained even after turning off the bootci. Covariance matrix based linear estimator was implemented in hopes of solving this issue, but they don't seem to speedup substantially. Other approaches which were tried involve using Numba and future work can also try multi-threading to speedup.
2. Because of runtime issues, the algorithm was never tested on timeseries input with more than 20 variables on a laptop. Immediate future work involves, running existing app directly on brainlife.io, after the preprocessing pipeline setup based on https://github.com/faskowit/app-fmri-2-mat/tree/0.1.6
3. The implementation does a full sweep of n-plet size to a predefined maxsize and usually one stops when the informational content is compatible with zero. The other way would be to do a greedy search and can be a possible future work.

#### Key points on difference between this implementation and Brinco lab's -
1. Brinco Lab's [implementation](https://github.com/brincolab/High-Order-interactions) only works for continuous variables, but this gcmi based implementation should ideally work with both continuous and discrete variables
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



