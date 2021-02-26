import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

def AngleSpeed(radius):
    (v,h,theta,sigmaProp) = (np.zeros(len(t1)-2),np.zeros(len(t1)-2),np.zeros(len(t1)-2),np.zeros(len(t1)-2))
    # Initialising arrays of zeros two shorter than the data sets. This is because the velocity is calculated
    # using one r and t value above and below a given time value, so no velocity values can be calculated for
    # the two points at the extreme ends.
    for i in range(1,len(t1) - 1):
        v[i-1] = (radius[i+1] - radius[i-1])/(t1[i+1] - t1[i-1])*1e6
        CubicCoeffs = [1,0,3*radius[i]**2,6*7.6e-15/math.pi]
        roots = np.roots(CubicCoeffs) #Using numerical methods to solve cubic in h.
        for j in range(0,3):
            root = roots[j]
            if abs(root.imag) < 1e-5: #Finding real roots, since numerical methods could give a tiny imaginary part.
                h[i-1] = root.real
        theta[i-1] = 90 + 180/math.pi*(np.arctan((radius[i]**2 - h[i-1]**2)/(2*radius[i]*h[i-1])))
        # Calculating theta using the equation given, but + instead of - in order to keep the angles just above
        # 0 degrees.
        sigmaProp[i-1] = v[i-1]/math.sqrt(2)*math.sqrt(1e-12/radius[i]**2 + 1e-12/t1[i]**2)
        # Propogating errors on for v using both sigma_R and sigma_H = 1e-6, as suggested by the data.
    PlotArray = np.array([theta,v,sigmaProp])
    return PlotArray

def PolynomialFit(x,y,sigma,n):
    (invSigma,residual) = (np.zeros(len(sigma)),np.zeros(len(sigma)))
    # invSigma gives 1/sigma, and residual gives the difference between the measured v-values and the v-values as
    # predicted by the polynomial fit.
    for i in range(0,len(sigma)):
        invSigma[i] = 1/sigma[i]
    fit = np.polyfit(x,y,n,w=invSigma)
    # fitting a polynomial based on the least squares method.
    paramError = np.polyfit(x,y,n,w=invSigma,cov=True)[1]
    # this could have been done simultaneously with the fit = ..., but I put this in later, and rewriting would
    # be a pain, so this will have to do.
    xarray = np.arange(np.amin(PlotData[0]),np.amax(PlotData[0]),0.01)
    # makes an array with evenly spaced theta-values, and the spacing is set to 0.01 to make a fairly cotinuous plot.
    yarray = np.zeros(len(xarray))
    for i in range(0,len(xarray)):
        for j in range(0,(n+1)):
            yarray[i] = yarray[i] + fit[j]*xarray[i]**(n-j)
    # calculating the values resulting from the polynomial fit, in order to plot them alongside.
    chiSq = 0
    for i in range(0,len(x)):
        yfit = 0
        for j in range(0,(n+1)):
            yfit = yfit + fit[j]*x[i]**(n-j)
            # This is the fitted polynomial for each x point.
        chiSq = chiSq + ((y[i] - yfit)/sigma[i])**2
        # calculating chi-squared values.
        residual[i] = y[i] - yfit
    chiSq = chiSq/(len(x)-n-1)
    return [[xarray],[yarray],chiSq,fit,residual,paramError]

def deGennes(x,a,b):
    dG = a*x**2 - b
    return dG

def deGennesFit(x,y,sigmaY):
    residual = np.zeros(len(sigmaY))
    # residual gives the difference between the measured v-values and the v-values as
    # predicted by the de Gennes fit.
    xarray = np.arange(np.amin(PlotData[0]),np.amax(PlotData[0]),0.01)
    # makes an array with evenly spaced theta-values, and the spacing is set to 0.01 to make a fairly cotinuous plot.
    fit = curve_fit(deGennes,x,y,sigma=sigmaY)
    # fitting a polynomial based on the least squares method.
    yarray = np.zeros(len(xarray))
    for i in range(0,len(xarray)):
        yarray[i] = deGennes(xarray[i],fit[0][0],fit[0][1])
    # calculating the values resulting from the polynomial fit, in order to plot them alongside.
    chiSq = 0
    for i in range(0,len(x)):
        yfit = deGennes(x[i],fit[0][0],fit[0][1])
        chiSq = chiSq + ((y[i] - yfit)/sigmaY[i])**2
        # calculating chi-squared values.
        residual[i] = y[i] - yfit
    chiSq = chiSq/(len(x)-3)
    return [[xarray],[yarray],chiSq,fit,residual]

def CoxVoinov(x,a,b):
    dG = a*x**3 - b
    return dG

def CoxVoinovFit(x,y,sigmaY):
    residual = np.zeros(len(sigmaY))
    # residual gives the difference between the measured v-values and the v-values as
    # predicted by the de Gennes fit.
    xarray = np.arange(np.amin(PlotData[0]),np.amax(PlotData[0]),0.01)
    # makes an array with evenly spaced theta-values, and the spacing is set to 0.01 to make a fairly cotinuous plot.
    fit = curve_fit(CoxVoinov,x,y,sigma=sigmaY)
    # fitting a polynomial based on the least squares method.
    yarray = np.zeros(len(xarray))
    for i in range(0,len(xarray)):
        yarray[i] = CoxVoinov(xarray[i],fit[0][0],fit[0][1])
    # calculating the values resulting from the polynomial fit, in order to plot them alongside.
    chiSq = 0
    for i in range(0,len(x)):
        yfit = CoxVoinov(x[i],fit[0][0],fit[0][1])
        chiSq = chiSq + ((y[i] - yfit)/sigmaY[i])**2
        # calculating chi-squared values.
        residual[i] = y[i] - yfit
    chiSq = chiSq/(len(x)-3)
    return [[xarray],[yarray],chiSq,fit,residual]

data1 = np.loadtxt('Top_view_drop_1_data_run1.txt')
t1 = data1[:,0]
radius1 = data1[:,1]*1e-6
# t1 is the time data, it is identical for each data set in the same run, so is only defined once.

data2 = np.loadtxt('Top_view_drop_1_data_run2.txt')
radius2 = data2[:,1]*1e-6

data3 = np.loadtxt('Top_view_drop_1_data_run3.txt')
radius3 = data3[:,1]*1e-6

AngleSpeed1 = AngleSpeed(radius1)
AngleSpeed2 = AngleSpeed(radius2)
AngleSpeed3 = AngleSpeed(radius3)
# defining AngleSpeedX here means the function only has to be called once for each run, saving time.

velocity = [[],[],[]]
# Initialising an array of arrays for velocity, to make calculating standard deviation from the spread of data easier.

velocity[0] = AngleSpeed1[1]
velocity[1] = AngleSpeed2[1]
velocity[2] = AngleSpeed3[1]

PlotData = (AngleSpeed1 + AngleSpeed2 + AngleSpeed3)/3
# Averaging the data resulting from the anglespeed function. Again, there's a function to do this, but this works
# too.

sigmaMes = np.zeros(len(t1)-2)
# Initialising an array to store the uncertainties calculated from the spread of data.

for i in range(0,len(t1)-2):
    iSigma = 0
    for j in range(0,3):
        iSigma = iSigma + (velocity[j][i] - PlotData[1][i])**2
    sigmaMes[i] = math.sqrt(iSigma/2)/math.sqrt(2)
# Calculating the standard deviation resulting from the spread of data. I realised after writing this that there
# is a python function to do this for me, but this works too.

FittedAxes = PolynomialFit(PlotData[0],PlotData[1],sigmaMes,3)
# Again, the function only has to be called once, saving some time to run the program.

# Determining the equilibrium angle for a general polynomial fit.
allAngle = np.roots(FittedAxes[3])
eqAngle = 90
for j in range(0,3): # 2 or 3 depending on cubic or quadratic fit
    root = allAngle[j]
    if abs(root.imag) < 1e-5: #Finding real roots, since numerical methods could give a tiny imaginary part.
        if root.real < eqAngle:
            eqAngle = root.real
errp1 = math.sqrt(FittedAxes[5][0][0]) # For polynomial fit
print(eqAngle)

deGennesAxes = deGennesFit(PlotData[0],PlotData[1],sigmaMes)
#print(deGennesAxes[3][1][0][0])
#errp1 = math.sqrt(deGennesAxes[3][1][0][0])

CoxVoinovAxes = CoxVoinovFit(PlotData[0],PlotData[1],sigmaMes)
#print(CoxVoinovAxes[3][1][0][0])
#errp1 = math.sqrt(CoxVoinovAxes[3][1][0][0])

print(errp1)

plt.gcf().clear()

# Clearing the plots, otherwise they get very messy.

plt.title("Plot of the top view small contact angle data fitted to a general cubic.")

plt.ylabel("Speed of contact point (μm/s)")

plt.plot(PlotData[0],PlotData[1],"b.") # Plots the v data against theta
plt.errorbar(PlotData[0],PlotData[1],yerr=sigmaMes,fmt="none") # Puts in the error bars
#plt.plot(FittedAxes[0],FittedAxes[1],"r.") # Puts in the general polynomial fit
#plt.plot(deGennesAxes[0],deGennesAxes[1],"r.") # Puts in the de Gennes fit
plt.plot(CoxVoinovAxes[0],CoxVoinovAxes[1],"r.") # Puts in the Cox Voinov fit

plt.xlabel("Contact angle (degrees).")

#plt.plot(FittedAxes[0],np.zeros(len(FittedAxes[0])),"g.") # Puts in a line at v = 0 for reference
#plt.plot(PlotData[0],FittedAxes[4],"r.") # Plots residual y for general polynomial against x
#plt.plot(PlotData[0],deGennesAxes[4],"r.") # Puts residual y for de Gennes fit
#plt.plot(PlotData[0],CoxVoinovAxes[4],"r.") # Puts residual y for Cox Voinov fit
#plt.errorbar(PlotData[0],CoxVoinovAxes[4],yerr=sigmaMes,fmt="none") # Puts in the error bars#

#plt.ylabel("Residuals (μm/s)")

plt.show()

#print("The reduced Chi Squared value is %f" % (FittedAxes[2])) # General polynomial prinout.
#print("The reduced Chi Squared value is %f" % (deGennesAxes[2])) # de Gennes polynomial prinout.
print("The reduced Chi Squared value is %f" % (CoxVoinovAxes[2])) # Cox Voinov polynomial prinout.
print(CoxVoinovAxes[3])