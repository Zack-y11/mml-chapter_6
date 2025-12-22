import numpy as np
from collections import Counter

#Coin Flip Probability Space
#This is an example of using the probability space using the coin flip

#This is our space of events
sample_space = ['HH', 'HT', 'TH', 'TT']

#Random variables: map outcomes to number of HEADS
def random_variable(outcome):
    return outcome.count('H')

#Compute the probability distribution P(X=k) for k in {0,1,2}
target_space = {0: [], 1:[], 2:[]}
for outcome in sample_space:
    num_of_heads = random_variable(outcome)
    target_space[num_of_heads].append(outcome)
    
#prob mass function (PMF)
pmf = {k: len(v)/ len(sample_space) for k, v in target_space.items()}
print(f"P(X=0) = {pmf[0]}") #P no heads
print(f"P(X=1) = {pmf[1]}") #P one heads
print(f"P(X=2) = {pmf[2]}") #P two heads


