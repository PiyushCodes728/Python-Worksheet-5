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

# 1) Correlation coefficient
r, p_value = stats.pearsonr(years, pop)

# 2) Linear regression
slope, intercept, r_value, p_reg, std_err = stats.linregress(years, pop)
reg_2008 = slope*2008 + intercept

# 3) Interpolation
lin_interp = interpolate.interp1d(years, pop, kind='linear')
interp_2008 = float(lin_interp(2008))

# Results
print("Pearson correlation r =", round(r,4))
print("Regression equation: population =", round(slope,4), "* year +", round(intercept,2))
print("R-squared =", round(r_value**2,4))
print("Estimated population in 2008 (regression):", round(reg_2008,2))
print("Estimated population in 2008 (interpolation):", round(interp_2008,2))

plt.scatter(years, pop, color='red', label="Data")
plt.plot(years, slope*years+intercept, label="Regression")
plt.plot(years, lin_interp(years), '--', label="Interpolation")
plt.scatter(2008, reg_2008, marker='x', color='blue', label="2008 (regression)")
plt.scatter(2008, interp_2008, marker='D', color='green', label="2008 (interp)")
plt.xlabel("Year")
plt.ylabel("Population (thousands)")
plt.legend()
plt.show()

























