# This python file has sample codes as to how to read the outputs generated for different configs.
# The output dictionary is slightly different whether you are using higher_order = False or True

# necessary imports
import numpy as np
import itertools

from toolbox.utils import load_obj, combinations_manager

## NOTE: If you face any issues with loading pickled outputs then use pickle5 instead of pickle in utils.py


## Case 1: argument higher_order = False. 
# Pregenerating all possible combinations, not using the combinatorial numbering system.
# To follow along,
# Run the toy_example.py on fmri timeseries of just first 10 variables for the following config
# "higher_order": false,
# "estimator": "gcmi",
# "modelorder":3,
# "maxsize":4,
# "n_best":10, 
# "nboot":100

# Run for both Oinfo and dOinfo and you will have generated outputs Odict_Oinfo.pkl and Odict_dOinfo.pkl
# Load the dicts (equivalent of structs from MATLAB)
Odict_Oinfo = load_obj('sample_Odict_Oinfo')
Odict_dOinfo = load_obj('sample_Odict_dOinfo')


print("Oinfo readout example, higher_order = False")
# How to read the Oinfo output?
# Say you want to know the outputs for multiplet size 3
# then you would print the following
isize = 3
print(Odict_Oinfo[isize])

# Which will give the result:
#{'sorted_red': array([0.14653095, 0.13152774, 0.12752084, 0.12326999, 0.09590787,
#        0.07990326, 0.07964545, 0.07481491, 0.06944254, 0.06820533]), 'index_red': array([105, 115, 113,  80, 114,  88,  95, 119,  35,  28], dtype=int64), 'bootsig_red': array([[1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.]]), 'sorted_syn': array([-0.1963071 , -0.16986031, -0.13420828, -0.12996375, -0.10197577,
#        -0.09634448, -0.09016494, -0.08868647, -0.08328115, -0.08035267]), 'index_syn': array([ 48,   1,  56,  92,   6,  55,  82,   2,  63, 116], dtype=int64), 'bootsig_syn': array([[1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.]])}

# The result includes'sorted_red', 'index_red','bootsig_red', 'sorted_syn', 'index_syn', 'bootsig_syn'.
# Length of each of these will be n_best (argument in config)
# 'index_red' is the index of the n_best (here 10) combinations with the highest redundancies in descending order
# and 'sorted_red' is the actual redundancies of those combinations.
# 'bootsig_red' tells us whether the redundancy value for that combination is significant (sufficiently far from zero)
# similarly for synergy values

# How do I retrieve the combination from the index?
# Say here the index of the 3-plet combination with the highest redundancy value is 105
# since higher_order = False, first let's construct all possible 3-plet combinations from 10 variables
nvartot = 10
nplets_iter=itertools.combinations(range(1,nvartot+1),isize)
nplets = []
for nplet in nplets_iter:
    nplets.append(nplet)
C = np.array(nplets) 

# Now C will have all possible 10C3 combinations - 
# That means the 2d array looks like [[ 1  2  3], [ 1  2  4],..., [ 7  9 10], [ 8  9 10]]
# The combination at index 105 can be retrieved as follows
print("Combination with highest redundancy:", C[105])
# Which should output [5 7 9]

print("\n \n")




##################
print("dOinfo readout example, higher_order = False")
# How to read the dOinfo output?
# dOinfo computation additionally requires fixing a target
# So lets say you fix the first variable (ROI/timeseries) as target which is 0 index and
# Say you want to know the outputs for multiplet size 3 
# then you would print the following
target_var_index = 0
isize = 2
print(Odict_dOinfo[target_var_index][isize])

# Which will give the result:
# {'sorted_red': array([0.01387719, 0.00827556, 0.00461848, 0.00423925, 0.00421622,
#        0.004088  , 0.00388924, 0.00359825, 0.0033909 , 0.00250356]), 'index_red': array([22,  6, 33, 32, 23, 18, 25, 28,  9, 30], dtype=int64), 'bootsig_red': array([[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]]), 'sorted_syn': array([-0.02292608, -0.01824932, -0.01787444, -0.01531304, -0.01455274,
#        -0.01077729, -0.00781092, -0.00726193, -0.00587415, -0.00477357]), 'index_syn': array([31, 17,  4, 29,  2, 14, 24,  1, 15, 34], dtype=int64), 'bootsig_syn': array([[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]]), 'var_arr': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10])}

# You would notice this has an additional output 'var_arr', rest is exactly the same as Oinfo
# 'var_arr' is the array of variables (or ROIs) created after removing the fixed target variable
# The n-plet combinations are drawn from this var_arr array, this will aid us in retrieving the combination

# In order to get back the combination of the max Redundancy value (of 0.01387719)
# which lies in the index 22, do the following
nvartot = 10
var_arr = Odict_dOinfo[target_var_index][isize]['var_arr']
nplets_iter=itertools.combinations(var_arr,isize)
nplets = []
for nplet in nplets_iter:
    nplets.append(nplet)
C = np.array(nplets) # n-tuples without repetition over N modules


print("Combination with highest redundancy:", C[22])
# Which should output [ 5 7]
print("\n \n")



##################
## Case 2: argument higher_order = True.
# Using the combinatorial numbering system to generate combinations iteratively and not run out of memory
# To follow along, use the same config as before but with higher_order = True and save the outputs

# Loading the outputs
Odict_Oinfo = load_obj('sample_Odict_Oinfo_higher_order')
Odict_dOinfo = load_obj('sample_Odict_dOinfo_higher_order')


print("Oinfo readout example, higher_order = True")
# How to read the Oinfo output?
# Say you want to know the outputs for multiplet size 3
# then you would print the following... (same as before)
isize = 3
print(Odict_Oinfo[isize])

# Which will give the result:
# {'sorted_red': array([0.14653095, 0.13152774, 0.12752084, 0.12326999, 0.09590787,
#        0.07990326, 0.07964545, 0.07481491, 0.06944254, 0.06820533]), 'index_red': array([ 75., 117.,  82.,  73., 110.,  65.,  74., 119., 112.,  66.]), 'bootsig_red': array([[1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.]]), 'sorted_syn': array([-0.1963071 , -0.16986031, -0.13420828, -0.12996375, -0.10197577,
#        -0.09634448, -0.09016494, -0.08868647, -0.08328115, -0.08035267]), 'index_syn': array([ 88.,   1.,  67.,  69.,  56.,  46.,  79.,   4., 113.,  83.]), 'bootsig_syn': array([[1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.],
#        [1.]])}


# This would give the same output as before but with different indices. Don't worry!
# The indices don't vary each run and are fixed for a fixed N and a fixed K in N choose K combinations


# For those who are interested: 
# This Wikipedia page explains it nicely - https://en.wikipedia.org/wiki/Combinatorial_number_system#Place_of_a_combination_in_the_ordering
# The class combinations_manager in utils.py which implements this mapping (both combination to a number and the other way round) along with your nextchoose function. 
# Using the nextchoose function to generate subsequent combinations is still faster than looping through say 1 to nCk and getting the combination for each number, 
# so I continue to use the nextchoose function to generate nCk combinations in the main exhaustive loop
# I've tried to illustrate it in a simple 5C3 example here - https://docs.google.com/spreadsheets/d/1A-JTEIu2pMHfYKUtrXUwUxDwllwveWbNJN0l2jB6vkA/edit?usp=sharing

# Back from the detour, it's straightforward to retrieve the combination from the index using the combinations_manager
# To get the 3-plet combination with highest redundancy (of 0.14653095)
# We need to retrieve combination paired with combinatorial numbered index 75

H = combinations_manager(nvartot,isize)
print("Combination with highest redundancy:", H.number2combination(75))
# Which should output [5 7 9] which also obviously matches the output from higher_order = False
print("\n \n")



##################
print("dOinfo readout example, higher_order = True")
# How to read the dOinfo output?
# dOinfo computation additionally requires fixing a target
# So lets say you fix the first variable (ROI/timeseries) as target which is 0 index and
# Say you want to know the outputs for multiplet size 3 
# then you would print the following... (same as before)

target_var_index = 0
isize = 2
print(Odict_dOinfo[target_var_index][isize])

# Which will give the result:
# {'sorted_red': array([0.01387719, 0.00827556, 0.00461848, 0.00423925, 0.00421622,
#        0.004088  , 0.00388924, 0.00359825, 0.0033909 , 0.00250356]), 'index_red': array([13., 21., 27., 33., 18., 17., 31., 25.,  4., 20.]), 'bootsig_red': array([[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]]), 'sorted_syn': array([-0.02292608, -0.01824932, -0.01787444, -0.01531304, -0.01455274,
#        -0.01077729, -0.00781092, -0.00726193, -0.00587415, -0.00477357]), 'index_syn': array([26., 12., 10., 32.,  3., 29., 24.,  1.,  5., 34.]), 'bootsig_syn': array([[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]]), 'var_arr': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10])}

# In order to get back the combination of the max Redundancy value (of 0.01387719)
# which lies in the combinatorial numbered index 13, do the following
var_arr = Odict_dOinfo[target_var_index][isize]['var_arr']
H = combinations_manager(nvartot,isize)
print("Combination with highest redundancy:", var_arr[H.number2combination(13)-1])
# Which should output [ 4 10]
print("\n \n")
