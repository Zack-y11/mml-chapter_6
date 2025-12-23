import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#univariate example 
print("------------UNIVARIATE: fliping a biased coing--------------")
#Bernuolli: flip coin with p = 0.7 probability of heads

p = 0.7 
X_values = [0, 1] #tails=0, heads = 1
pmf = [1-p, p]

mean = sum(x * px for x, px in zip(X_values, pmf))
variance = sum((x-mean)**2 * px for x, px in zip(X_values, pmf))

print(f" P(X=1) = {p:.2f} ")
print(f" E[X]= {mean:.2f} ")
print(f" Var[X] = {variance:.2f} ")
print(f"σ = √Var(X)= {np.sqrt(variance):.4f} ")

#how works on multivariate? 
#this multivariate example is based on heights and weights
print("\n =====================MULTIVARIATE: heights vs weights =======================")

np.random.seed(42)

#generated correlated data
#what is correlated data: basically there is the relation between two variables
#positive: variables move together, negative: move on opposite directions, Zero: independent or uncorrelted

cov_matrix = [[1.0, 0.8], [0.8, 1.0]] # var(height=1), var(weight=1), cov=0.8 (positive)
mean_vector = [170, 70] #cm and kg
n_samples = 1000

data = np.random.multivariate_normal(mean_vector, cov_matrix, n_samples)
heights, weights = data[:, 0], data[:, 1]

#computing statistics
computed_mean = data.mean(axis=0)
computed_cov = np.cov(data.T)

print(f"computed mean: {computed_mean}")
print(f"sample covariance matrix: ")
print(computed_cov)
print(f"\n covariance[0,1] = {computed_cov[0,1]:.4f}")
print(f"-> Strong positive correlation: tall people then to be heavier")

#Plot
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))

ax1.scatter(heights, weights, alpha=0.5, s=20)
ax1.set_xlabel("Height (cm)")
ax1.set_ylabel("Weight (cm)")
ax1.set_title("Height vs Weight (Positive covariance)")
ax1.grid(True, alpha=0.3)

#covariance matrix heap
im = ax2.imshow(computed_cov, cmap="coolwarm", aspect='auto', vmin=-1, vmax=1)
ax2.set_xticks([0,1])
ax2.set_yticks([0,1])
ax2.set_xticklabels(['Height', 'Height'])
ax2.set_yticklabels(['Weight', 'Weight'])
ax2.set_title('Covariance Matrix')

for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{computed_cov[i,j]:.2f}",
                ha='center', va='center', color='black')

plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig('statistics_mean_cov/mean_covariance.jpg')
                
        