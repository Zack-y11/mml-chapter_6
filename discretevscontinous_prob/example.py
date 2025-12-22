#Code Discrete vs continous Probabilities
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

#discrete: Binomial distribution (n=10 coin flips, p=0.6)
n, p = 10, 0.6
x_discrete = np.arange(0, n+1)
pmf = binom.pmf(x_discrete, n,p)
print(f"P(X=5) = {binom.pmf(5, n, p):.4f}") #extract the prob of 5heads
print(f"Sum of all PMF values = {pmf.sum():.4}") #should be equal to 1


#Continuous
mu, sigma = 0, 1
x_continous = np.linspace(-4, 4, 100)
pdf = norm.pdf(x_continous, mu, sigma)

print("======== CONTINUOUS Gausian ============")
print(f"f(x=0) = {norm.pdf(0, mu, sigma):.4f}") #not a probability
print(f"f(x=0 exactly) = 0 (infinitesimal)") #always 0 for continous

#integral under curve   = probability of interval
prob_interval = norm.cdf(1, mu, sigma) - norm.cdf(-1, mu, sigma)
print(f"P(-1 <= X 1 <= 1) = {prob_interval:.4f}") #+- 68% for standard normal

#plot both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,4))
ax1.bar(x_discrete, pmf, color='blue', alpha=0.7, label="PMF")
ax1.set_xlabel("Number of Heads (X)")
ax1.set_ylabel("Probablity of p(x=k)")
ax1.set_title("Discrete Binomial (n=10, p=0.6)")
ax1.legend()

ax2.plot(x_continous, pdf, 'b-', linewidth=2, label='PDF')
ax2.fill_between(x_continous, pdf, alpha=0.3)
ax2.set_xlabel('Value(X)')
ax2.set_ylabel('Density f(x)')
ax2.set_title('Continuous: N(0, 1)')
ax2.legend()

plt.tight_layout()
plt.savefig('discrete_vs_continous.png', dpi=100)