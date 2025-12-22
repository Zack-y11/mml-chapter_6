from collections import Counter
suits = ["Hearts", "Diamonds", "Clubs", "Spades"]

sample_space = [(s1,s2) for s1 in suits for s2 in suits if s1 != s2]

def random_variable(outcome):
    #count red suits (hearts, diamnods)
    return sum(1 for suit in outcome if suit in ["Hearts", "Diamonds"])

rv_values = [random_variable(outcome) for outcome in sample_space]

value_counts = Counter(rv_values)
total_outcomes = len(sample_space)
pmf = {value: count / total_outcomes for value, count in value_counts.items()}

print("Sample space size:", total_outcomes)
print("\n PMF (Probability Mass Function):")
for value in sorted(pmf.keys()):
    print(f"P(X = {value}) = {pmf[value]:.4f} ({value_counts[value]}/{total_outcomes})")