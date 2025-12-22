sample_space = [(i, j) for i in range(1,7) for j in range(1,7)]

def random_variable(outcome):
    return sum(outcome)

#compute PMF for sums 2-12
target_space = {k: [] for k in range(2,13)}
for outcome in sample_space:
    dice_sum = random_variable(outcome)
    target_space[dice_sum].append(outcome)
    
print(target_space)

pmf={k: len(v)/len(sample_space) for k, v in target_space.items()}

for k, prob in pmf.items():
    print(f"P(sum={k}) = {prob:.4f}")
    
