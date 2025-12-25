#Variable Transformation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

print(" =========== VARIABLE TRANSFORMATION =========== ")

#Example 1: Linear Transformation
print("Example 1: Linear Transformation \n")
print("Original: X ~ N(0,1)")
print("Transform: Y = 2X + 3")
print("Result: Y ~ N(2*0+3, 2^2*1) = N(3, 4)\n")

x = np.linspace(-3,3, 200)
y = 2*x+3 #Linear Transform
y_plot = np.linspace(-3, 9, 200)

fig, axes = plt.subplots(2,2, figsize = (12,10))

#original distribution
pdf_x = norm.pdf(x, 0, 1)
axes[0,0].plot(x, pdf_x, "b-", linewidth=2)
axes[0,0].fill_between(x, pdf_x, alpha=0.3)
axes[0,0].set_title("Original: X ~ N(0,1)")
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('f(x)')
axes[0,0].grid(True, alpha=0.3)

#Transformed distribution (theory)
pdf_y = norm.pdf(y_plot, 3,2)
axes[0,1].plot(y_plot, pdf_y, 'r-', linewidth=2)
axes[0,1].fill_between(y_plot, pdf_y, alpha=0.3)
axes[0,1].set_title("Transformed: Y=2X+3 ~ N(3,4)")
axes[0,1].set_xlabel('y')
axes[0,1].set_ylabel('f(y)')
axes[0,1].grid(True, alpha=0.3)

#verify by sampling
np.random.seed(42)
x_samples = np.random.normal(0,1,5000)
y_samples = 2*x_samples + 3

axes[1,0].hist(x_samples, bins=50, density=True, alpha=0.7, label='Sample from X')
axes[1,0].plot(x, pdf_x, 'b-', linewidth=2, label='Theory')
axes[1,0].set_title("X samples histogram")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].hist(y_samples, bins=50, density=True, alpha=0.7, label='Samples from Y=2X+3')
axes[1,1].plot(y_plot, pdf_y, 'r-', linewidth=2, label="Theory")
axes[1,1].set_title('Y samples histogram')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('change_variables/transformation_linear.png', dpi=100)
print("fig transformation linear saved")

#example 2: non linear transformation
print("Example 2: NonLinear Transformation \n")
print("Original X~N(0,1)")
print("Transform: Y = X^2")
print("Result: Y ~ Chi-square(df=1)\n")

#For Y = X^2
#if x ~ N(0,1), then x^2 ~ X^2(1)
#using change of variables :
#f_Y(y) =  f_X(√y) * |d(√y)/dy| + f_X(-√y) * |d(-√y)/dy|
#        = f_X(√y) * (1/2√y) + f_X(-√y) * (1/2√y)
#        = 2 * (1/√(2π)) * exp(-y/2) * (1/2√y)
#        = χ²(y; df=1)

fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(15,4))

#Original
x_nonlin = np.linspace(-3,3,200)
pdf_x_nonlin = norm.pdf(x_nonlin, 0, 1)
ax0.plot(x_nonlin, pdf_x_nonlin, 'b-', linewidth=2)
ax0.fill_between(x_nonlin, pdf_x_nonlin, alpha=0.3)
ax0.set_title('Original: X ~ N(0,1)')
ax0.set_xlabel('x')
ax0.set_ylabel('f(x)')
ax0.grid(True, alpha=0.7)

#Transformed (chi-square)
y_nonlin = np.linspace(0,6, 200)
pdf_y_nonlin = chi2.pdf(y_nonlin, df=1)
ax1.plot(y_nonlin, pdf_y_nonlin, 'r-', linewidth=2)
ax1.fill_between(y_nonlin, pdf_y_nonlin, alpha=0.3)
ax1.set_title("'Transformed : Y = X ^2 ~ X ^ 2 (1)'")
ax1.set_xlabel('y')
ax1.set_ylabel("f(y)")
ax1.grid(True, alpha=0.3)

#Jacobian factor
ax2.plot(y_nonlin, 1/(2*np.sqrt(y_nonlin)), 'g-', linewidth=2)
ax2.fill_between(y_nonlin, 1/(2*np.sqrt(y_nonlin)), alpha=0.3)
ax2.set_title("Jacobian: |dy/dx| = 1/(2rootsquare(y))")
ax2.set_xlabel('y')
ax2.set_ylabel('|Jacobian|')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0,6)

plt.tight_layout()
plt.savefig('change_variables/transformation_nonlinear.png', dpi=100)

print("KEY FORMULA:")
print("f_Y(y) = f_X(g⁻¹(y)) * |dg⁻¹(y)/dy|")
print("\nFor Y = X²:")
print("g⁻¹(y) = ±√y")
print("|dg⁻¹/dy| = 1/(2√y)")
print("f_X(√y) = (1/√(2π)) exp(-y/2)")
print("→ Chi-square PDF!")