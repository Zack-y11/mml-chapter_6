import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta

print("======== CONJUGATE PRIORS: Beta-Binomial Example ==========\n")

#Coin flip experiment
#prior belief: coin is fair-ish
alpha_prior, beta_prior = 2,2 #Beta (2,2); weakly centered at 0.5
print (f"PRIOR: beta (a={alpha_prior}, B={beta_prior})")
print(f"-> E[p] = a/(a+B) = {alpha_prior/(alpha_prior + beta_prior):.2f}")
print("Uniform-ish belief about coin fairness \n")

#experiment: flip coins N timmes, observe k heads
n_flips=[10,50,100]
n_heads = [7,25,70] #empirical heads 70%

fig, axes = plt.subplots(1,3, figsize=(15, 4))

for idx, (n,k) in enumerate(zip(n_flips, n_heads)):
    #posterior update(cn=onjugay)
    #p(p | data) a p(data | p) * p(p)
    #Binomial * Beta = Beta
    alpha_post = alpha_prior + k
    beta_post = beta_prior + (n-k)

    print(f"Data: {k} heads in {n} flips")
    print(f"Posterior: Beta(a={alpha_post}, B={beta_post})")
    print(f"-> E[p] = {alpha_post/(alpha_post+beta_post):.3f}")

    #plot prior , likelihood (scaled), posterior
    p_values = np.linspace(0,1,200)

    prior_pdf = beta.pdf(p_values, alpha_prior, beta_prior)
    posterior_pdf = beta.pdf(p_values, alpha_post, beta_post)

    #Binomial likelihood (scaled for visualization)
    likelihood = binom.pmf(k, n, p_values) * 1000

    axes[idx].plot(p_values, prior_pdf, "b-", linewidth = 2, label='Prior')
    axes[idx].plot(p_values, likelihood, "g--", linewidth = 2.5, label="Posterior")
    axes[idx].axvline(k/n, color='orange', linestyle=':', linewidth=2, label=f"empirical : {k/n:.2f}")

    axes[idx].set_xlabel('Coin Bias P')
    axes[idx].set_ylabel('Density')
    axes[idx].set_title(f"{k}/{n} heads (N = {n})")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0,1)

plt.suptitle("Bayesian Learning: Prior -> Posterior as we collected data", fontsize=14)
plt.tight_layout()
plt.savefig("exponential_fam_&conjugacy/conjugate_prior.png", dpi=100)

print("Key Insight:")
print("- Prior: week belief in fair coin")
print("- see 7/10 heads: update toward p ~ 0.7")
print("- see 35/50 heads: confidence increase, still p ~ 0.7")
print("- see 70/100 heads: very confident p~0.7")
print("\n No integrals needed: just update a and B")
