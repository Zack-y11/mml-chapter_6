"""
Docstring for bayes_sum_product.example
Intuitive Understanding
Think of diagnosing disease from a test:
Prior p(disease): General population prevalence
Likelihood p(test+|disease): Sensitivity of test
Posterior p(disease|test+): Your updated belief after seeing test result
Evidence: How common are positive tests in general
In ML: Prior = model assumptions, Likelihood = fit to data, Posterior = updated model.
"""
#Code Example: Bayes Theorem for Disease Testing
import numpy as np
from scipy.stats import binom

#Scenario: Disease prevalence, test accuracy
disease_prevalence = 0.01 #1% have disease
test_sensitivity = 0.95 # P(test+ | disease) = 95%
test_specificity = 0.90 # P(test- | no disease) = 90%
test_fpr = 1 - test_specificity #false positive rate

#Prior : p(disease)
prior_disease = disease_prevalence
prior_no_disease = 1 - disease_prevalence

#likehood: p(tets+ | disease) and p(test+ | no disease)
lik_positive_given_disease=test_sensitivity
lik_positive_given_no_disease = test_fpr

#BAYES THEOREM CALCULATION
#p(disease | test+ ) = p(test+|diasease) * p(disease) / p(test+)
numerator = lik_positive_given_disease * prior_disease
marginal_likehood = (
    lik_positive_given_disease * prior_disease + 
    lik_positive_given_no_disease * prior_no_disease
)

posterior_disease = numerator / marginal_likehood

print("=== BAYES THEOREM: Disease Testing ===")
print(f"Prior p(disease) = {prior_disease:.4f}")
print(f"p(test+ | disease) = {lik_positive_given_disease:.4f}")
print(f"p(test+ | no disease) = {lik_positive_given_no_disease:.4f}")
print(f"\np(test+) [evidence] = {marginal_likehood:.6f}")
print(f"Posterior p(disease | test+) = {posterior_disease:.4f}")
print(f"\n→ Even with positive test, only {posterior_disease*100:.1f}% chance of disease!")
print("  (Because disease is rare, false positives dominate)")
