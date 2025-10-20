#Q1.
import numpy as np
from scipy import stats
arr=np.random.randint(1,100,size=20)
print(arr)
mean=stats.tmean(arr)
median=np.median(arr)
variance=stats.tvar(arr)
print(mean)
print(median)
print(variance)

#Q2.
import numpy as np
from scipy.fftpack import fft2
arr_2d=np.random.rand(4,4)
print(arr_2d)
fft_result=ff2(arr_2d)
print(fft_result)

#Q3.
import numpy as np
from scipy import linalg
A=np.array([[4,2],[3,5]])
print(A)
det=linalg.det(A)
print(det)
inv=linalg.inv(A)
print(inv)
eigenvalues,eigenvectors=linalg.eig(A)
print(eigenvalues)
print(eigenvectors)

#Q4.
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(0,10,num=100,endpoint=True)# endpoint=true means last value is included            
y=np.sin(x)
f=interpolate.interp1d(x,y,kind='nearest') #kind=nearest means graph is continuous
xnew=np.linspace(0,10,num=10,endpoint=True)
ynew=f(xnew)
plt.plot(x,y,'o',label='original') #label is used to identify the graph
plt.plot(xnew,ynew,'-',label='interpolated')
plt.legend() # from this we can the box that what is what at top 
plt.show()    #from this graph can be seen

#Q5.
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
time=np.linspace(0,1,num=500,endpoint=True)
date=np.cos(1*np.pi*7*time)+np.random.rand(500)*0.5
b,a=signal.butter(2,0.1)












#Q7.
import numpy as np
from scipy import stats
students=np.array(["Arin","Aditya","Chirag","Gurleen","Kunal"])
marks=np.array([
    [85,78,92,88],
    [79,82,74,90],
    [90,85,89,92],
    [66,75,80,78],
    [70,68,75,85]
])
subjects=np.array(["Maths","Physics","Chemistry","English"])
total=np.sum(marks,axis=1)
average=np.mean(marks,axis=1)
subjectavg=np.mean(marks,axis=0)
top=np.argmax(total)
bottom=np.argmin(total)
passingstudents=np.sum(marks>=40,axis=1)
passingpercentage=(passingstudents/len(students))*100
mean=stats.tmean(marks)
median=stats.tmedian(marks)
print(subjects,total,average,subjectavg,top,bottom,passingstudents,passingpercentage)

#Q8.
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
t=np.array([0,1,2,3,4,5])
v=np.array([2,3.1,7.9,18.2,34.3,56.2]) #import values
def quadratic(t,a,b,c):
    return a*t**2+b*t+c #define quadratic equation
params,covariance=curve_fit(quadratic,t,y) #params is 1d array with a,b,c
a,b,c=params
print(f"Fitted equation: v(t) = {a:.3f} t² + {b:.3f} t + {c:.3f}") #human readable form
t_fit = np.linspace(0, 5, 100)  
v_fit = quadratic(t_fit, a, b, c)
plt.scatter(t, v, color='red', label="Given Data") #drwas original measure points
plt.plot(t_fit, v_fit, label="Fitted Curve", linewidth=2)
plt.title("Curve Fitting for Velocity Data")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.grid()
plt.show()

#Q9.
import numpy as np
import matplotlib.pyplot as plt
students=np.array(["Arin","Aditya","Chirag","Gurleen","Kunal"])
subjects=np.array(["Maths","Physics","Chemistry","English"])

marks=np.array([
    [85,78,92,88],
    [79,82,74,90],
    [90,85,89,92],
    [66,75,80,78],
    [70,68,75,85
    ]
])
total=np.sum(marks,axis=1)
average=np.mean(marks,axis=1)
subjectavg=np.mean(marks,axis=0)
topidx=np.argmax(total)
bottomidx=np.argmin(total)
print(total)
print(average)
print(subjectavg)
print(topidx)
print(bottomidx)
plt.bar(students, total_marks, color='skyblue')
plt.title("Total Marks per Student")
plt.ylabel("Total Marks")
plt.show()

plt.bar(subjects, subject_avg, color='lightgreen')
plt.title("Subject-wise Average Marks")
plt.ylabel("Average Marks")
plt.show()

#Q10.
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
t = np.array([0, 1, 2, 3, 4, 5])              
v = np.array([2, 3.1, 7.9, 18.2, 34.3, 56.2]) 
def quadratic(t, a, b, c):
    return a * t**2 + b * t + c
params, _ = curve_fit(quadratic, t, v)
a, b, c = params
print(f"Fitted Equation: v(t) = {a:.3f}t² + {b:.3f}t + {c:.3f}")
t_fit = np.linspace(0, 5, 100)
v_fit = quadratic(t_fit, a, b, c)
plt.scatter(t, v, color='red', label='Original Data', zorder=3)
plt.plot(t_fit, v_fit, color='blue', linewidth=2, label='Fitted Quadratic Curve')
plt.title("Curve Fitting: Velocity vs Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#Q11.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interpolate
years = np.array([2000, 2005, 2010, 2015, 2020])
pop = np.array([50, 55, 70, 80, 90])

r, p_value = stats.pearsonr(years, pop)

slope, intercept, r_value, p_reg, std_err = stats.linregress(years, pop)
reg_2008 = slope*2008 + intercept

lin_interp = interpolate.interp1d(years, pop, kind='linear')
interp_2008 = float(lin_interp(2008))

print("Pearson correlation r =", round(r,4))
print("Regression equation: population =", round(slope,4), "* year +", round(intercept,2))
print("R-squared =", round(r_value**2,4))
print("Estimated population in 2008 (regression):", round(reg_2008,2))
print("Estimated population in 2008 (interpolation):", round(interp_2008,2))

# Plot
plt.scatter(years, pop, color='red', label="Data")
plt.plot(years, slope*years+intercept, label="Regression")
plt.plot(years, lin_interp(years), '--', label="Interpolation")
plt.scatter(2008, reg_2008, marker='x', color='blue', label="2008 (regression)")
plt.scatter(2008, interp_2008, marker='D', color='green', label="2008 (interp)")
plt.xlabel("Year")
plt.ylabel("Population (thousands)")
plt.legend()
plt.show()

#Q12.
# Finding Roots of a Polynomial using SciPy
import numpy as np
from scipy import roots  # (we can also use numpy.roots directly)
# Polynomial:  x³ - 6x² + 11x - 6 = 0
# Coefficients in decreasing powers of x
coeff = [1, -6, 11, -6]
# Method 1: Using NumPy (simple)
r = np.roots(coeff)
print("Roots using NumPy:", r)
# Method 2: Using SciPy (from scipy.optimize or poly1d)
from scipy import poly1d
p = poly1d(coeff)
roots_scipy = p.r
print("Roots using SciPy poly1d:", roots_scipy)

#Q13.
# Integration using SciPy
import numpy as np
from scipy import integrate
#Define the function to integrate
def f(x):
    return x**2
# Integrate f(x) from 0 to 3
result, error = integrate.quad(f, 0, 3)
print("Integration of x^2 from 0 to 3 = ", result)
print("Estimated error =", error)

# Q14 - Differentiation using SciPy
import numpy as np
from scipy.misc import derivative
# Define the function
def f(x):
    return x**3 + 2*x**2 + 3*x + 4
# Find derivative at x = 2
x_val = 2
result = derivative(f, x_val, dx=1e-6)
print("Derivative of f(x) = x^3 + 2x^2 + 3x + 4 at x = 2 is:", result)

# Q15: Damped Oscillator (θ'' + 0.2θ' + 4θ = 0)
# Initial conditions: θ(0) = 1, θ'(0) = 0
# Solve for t = 0 to 20 s using scipy.odeint
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# Equation parameters
c = 0.2   # damping coefficient
k = 4.0   # stiffness (ωn²)
m = 1.0   # mass
# Differential equation
def dydt(y, t):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(c/m)*omega - (k/m)*theta
    return [dtheta_dt, domega_dt]
# Initial conditions
y0 = [1.0, 0.0]
# Time range
t = np.linspace(0, 20, 2001)
# Solve ODE
sol = odeint(dydt, y0, t)
theta = sol[:, 0]
# Find maximum displacement and its time
max_disp = np.max(theta)
max_time = t[np.argmax(theta)]
# Display results
print(f"Equation: θ'' + {c}θ' + {k}θ = 0")
print(f"Maximum displacement = {max_disp:.4f} rad at t = {max_time:.4f} s")
# Plot
plt.figure(figsize=(9, 4.5))
plt.plot(t, theta, label="θ(t)")
plt.title("Damped Oscillation of a System")
plt.xlabel("Time (s)")
plt.ylabel("Displacement θ (rad)")
plt.grid(True)
plt.legend()
plt.show()




















































