import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import multivariate_normal, norm
from scipy.linalg import cholesky 

print("===========GAUSSIAN PROPERTIES FRO ML ==========\n")

#1 univariate gaussian
mu, sigma = 0, 1
x = np.linspace(-4,4, 200)
pdf = norm.pdf(x, mu, sigma)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(x, pdf, 'b-', linewidth=2, label=f"N({mu} , {sigma}²)")
axes[0,0].fill_between(x, pdf, alpha=0.3)
axes[0,0].set_title("Univariate Gaussian PDF")
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('p(x)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

#Property 2: Multivariate Gaussian (2D)
print("Property 2: Miltivariate Gaussian 2D \n")
mu_2d = np.array([0,0])
sigma_2d = np.array([[1,0.6], [0.6,1]])

#2D grid
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3,3,100)
X1,X2 = np.meshgrid(x1, x2)
pos = np.dstack((X1, X2))

#evaluate PDF 
rv = multivariate_normal(mu_2d, sigma_2d)
Z = rv.pdf(pos)

contour = axes[0,1].contourf(X1, X2, Z, levels=15, cmap='viridis')
axes[0,1].set_title("2D Gaussian Countors")
axes[0,1].set_xlabel("X1")
axes[0,1].set_ylabel("X2")
plt.colorbar(contour, ax=axes[0,1])
    
#Property 3: Sampling from gaussian
print("Property 3: Sampling from multivariate Gaussian \n" )
n_samples = 2000
samples = np.random.multivariate_normal(mu_2d, sigma_2d, n_samples)

axes[1,0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=20)
axes[1,0].set_title("Samples from N(mu, sigma)")
axes[1,0].set_xlabel("X1")
axes[1,0].set_ylabel("X2")
axes[1,0].grid(True, alpha=0.3)

#verifying mean and covariance

print(f"True mean:  {mu_2d}")
print(f"Sample mean: {samples.mean(axis=0)}")
print(f"\nTrue Covariance: \n {sigma_2d}")
print(f"\n Sample covariance: \n {np.cov(samples.T)}")

# property 4: Linear transformation (key for regression)
print("\n\n Property 4: Linear Transformation of Gaussian \n")
print("If X ~ N(mu, sigma) and y = Ax + b, them y ~ N(Amu+b, AsigmaA^T)")
print("This is why gaussian assumptions propagate through linear models")

#example: linear regressiom y = W*x + noise

W = np.array([[2,-1]]) #1x2 weight matrix
b = 0.5
x_transformed = (samples @ W.T).ravel() + b  # 1D

axes[1,1].hist(x_transformed, bins=50, density=True, alpha=0.7, label="Transformed Samples")

#Compute theorical distribution
x_theory = np.linspace(x_transformed.min(), x_transformed.max(), 200)
mu_y = float((W @ mu_2d).squeeze() + b)
sigma_y = float((W @ sigma_2d @ W.T).squeeze())  # variance (scalar)
pdf_y = norm.pdf(x_theory, mu_y, np.sqrt(sigma_y))

axes[1,1].plot(x_theory, pdf_y, 'r-', linewidth=2, label='Theory: N(Wmu+b, WsigmaW^T)')

axes[1,1].set_title("Linear Transformation: y = Wx + b")
axes[1,1].set_xlabel('y')
axes[1,1].set_ylabel('Density')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path(__file__).resolve().parent / "gaussian_props.png"
fig.savefig(out_path, dpi=100)
print("\n Fig saved")

#property 5: conditional and mamrginal gaussian
print("\n\n Property 5: Conditioning (Critical for Bayesian Inference) \n")
#Bivariate Gaussian : p(x1, x2)
mu_12 = np.array([0,2])
sigma_12 = np.array([[1, 0.8], [0.8, 1]])

#condition on x2 = 1: p(x1, x2)
#formula from book eq: 6.66-6.67
x2_obs = 1
mu1, mu2 = mu_12
sigma11 = sigma_12[0, 0]
sigma12 = sigma_12[0, 1]
sigma22 = sigma_12[1, 1]

#Conditional Mean
mu_cond = mu1 + (sigma12 / sigma22) * (x2_obs - mu2)

#conditional variance
var_cond = sigma11 - (sigma12**2 / sigma22)

print(f"Joint: p(x1, x2) ~ N(mu = {mu_12}, sigma = ...)")
print(f"Observation: x2= {x2_obs}")
print(f"Conditional: p(x1 | x2 ={x2_obs}) ~ N({mu_cond:.3f}, {var_cond:.3f})")
print(f"Observing x2 = {x2_obs} shifts our belief about x1")

#THIS MATTERS!
#-Gaussian Linear Regression: Data = W^tx + noise ~ Gaussian
#-Kalman Filter: Conditional Gaussians for state estimation
#- Variational Inferences : Aproximate posterios with Gaussians