'''
OVERVIEW
========

  With the L{LinearRegression} class you can perform linear regressions, 
  hypothesis testing on the slope and intercept of said regression and 
  calculate confidence bands.

MAIN CLASSES
============

  - L{LinearRegression} is a class which performs linear regression
  - L{linearRegression} a function which given values for the response and 
    independent variable returns corresponding a L{LinearReg } object.

TYPICAL USAGE
=============

  For typical usage see class documentation.

THEORY OF OPERATION
===================

  A linear model assumes that a response variable, y, is related to an 
  independent variable, x, by a linear relationship of the form:
  
  y = S{beta}x + S{alpha}
  
  This module solves for the slope, m, and the intercept, c, by ordinary 
  least squares.

REFERENCES
==========
    
  - Draper, N.R.; Smith, H. (1998). Applied Regression Analysis (3rd ed.). 
    John Wiley. ISBN 0-471-17082-8.
  - W. Hardle, M. Muller, S. Sperlich, A. Werwatz (2004), 
    Nonparametric and Semiparametric Models, Springer, ISBN 3540207228

This file is part of Simon Wilshin's Regression module.

Simon Wilshin's Regression module is free software: you can redistribute 
it and/or modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation, either version 3 of the License, 
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
@author: Simon Wilshin
@contact: swilshin@gmail.com
@date: Jan 2016
'''

from numpy import mean,sum,sqrt,std

from scipy.stats import t as tdist

class LinearRegression(object):
  '''
  Computes a linear regression. Typical usage is to instantiate with 
  L{linearRegression} function, which calls the fit method 
  with a independent and response variable. Example:
  
  >>> from regression import LinearRegression, linearRegression
  >>> from numpy import linspace
  >>> from numpy.random import standard_normal,seed
  >>>
  >>> # Generate data
  >>> alphaA = 1.0
  >>> betaA = 2.0
  >>> sigmaN = 0.5
  >>> x = linspace(0,1,20)
  >>> xint = linspace(0,1,1000)
  >>> # Seed
  >>> seed(0)
  >>> y = alphaA + betaA*x + sigmaN*standard_normal(x.shape)
  >>>
  >>> # Fit model
  >>> rho = LinearRegression()
  >>> rho.fit(x,y)
  >>> # Alternatively for identical effect
  >>> rho = linearRegression(x,y)
  >>>
  >>> # Get CI
  >>> print "alpha: ", rho.getInterceptCI()
  alpha:  (1.1982168202431409, 1.9397307002909874)
  >>> print "beta: ", rho.getSlopeCI()
  beta:  (0.79750362486927528, 2.0652705199537369)
  >>> # Run Tests
  >>> rho.testIntercept()
  Testing intercept against hypothesis both found p of  5.28088324181e-08 .
  Significant at the 0.05 significance level.
  (8.8907076244941443, 5.2808832418094198e-08)
  >>> rho.testSlope()
  Testing slope against hypothesis both found p of  0.000162012943822 .
  Significant at the 0.05 significance level.
  (4.7441413089509128, 0.00016201294382156561)


  
  @ivar n: number of points 
  @type n: int
  @ivar dof: number of degrees of freedom in fit
  @type dof: int
  @ivar mux: average of independent variable
  @type mux: float
  ivar  muy: average of response variable
  @type muy: float
  @ivar kappa: square difference between indepedent variable and its mean
  @type kappa: float
  @ivar beta: slope estimate
  @type bate: float
  @ivar alpha: intercept estimate
  @type alpha: float
  @ivar r: R statistic
  @type r: float
  @ivar eps: residuals
  @type eps: float
  @ivar sbeta: s used for t-test for beta
  @type sbeta: float
  @ivar salpha: s used for t-test for alpha
  @type salpha: float
  '''
  def __init__(self):
    '''
    Constructor, currently takes no parameters and does nothing as there 
    is no initialisation.
    '''
    pass

  def fit(self,x,y):
    '''
    Given independent variable values, x, and response variable values, y, 
    perform a linear regression. Once this function is run the regression 
    object can be used for hypothesis testing, prediction, and calculating 
    confidence bounds and bands.
    
    @param x: values for the independent variable
    @type x: float array
    @param y: values for the response variable
    @type y: float array
    '''
    # Number of points and degrees of freedom
    self.n = x.size
    self.dof = self.n-2

    # mean x, mean y, and kappa
    self.mux = mean(x)
    self.muy = mean(y)
    self.kappa = sum((x-self.mux)**2)

    # Parameter estimates
    self.beta = (mean(x*y)-self.mux*self.muy)/(mean(x*x)-self.mux**2)
    self.alpha = self.muy-self.beta*self.mux

    # R statistic, residuals, standard errors
    self.r = self.beta*std(x)/std(y)
    self.eps = y-self.alpha-self.beta*x
    self.sbeta = sqrt(
                 (self.RSS()/self.dof)
                  /
                 sum((x-self.mux)**2)
                )
    self.salpha = self.sbeta*sqrt(mean(x**2))

  def RSS(self):
    '''
    Calculate the residual sum of squares (RSS) for the fit. Returns the RSS.
    
    @return: The residual sum of squares for the fit.
    @rtype: float
    '''
    return(sum(self.eps**2))

  def gettstat(self,x,sx,x0):
    '''
    Calculate the t-statistic for x-x0 with sx the appropriate variability, 
    returns the t statistic as a float.
    
    @param x: sample mean
    @type x: float
    @param sx: population and sample standard deviation ratio 
    @type sx: float
    @param x0: population mean
    @type x0: float
    @return: t-statistic
    @rtype: float
    '''
    return((x-x0)/sx)

  def test(self,x,sx,gamma,h0,x0):
    '''
    Performs a t-test with to determine if x-x0 is significantly different 
    from zero with sx the appropriate variability for a student test 
    at level gamma (0.05 corresponds to p<0.05) under hypothesis 
    h0 which can be 'greater', 'less', or 'both'.
    Returns to t statistic and the p-value
    
    @param x: sample mean
    @type x: float
    @param sx: population and sample standard deviation ratio 
    @type sx: float
    @param gamma: alpha value (significance critereon)
    @type gamma: float
    @param h0: sidedness, can be 'greater', 'less', or 'both'
    @type h0: float
    @param x0: population mean
    @type x0: float
    @return: t-statistic and p-value
    @rtype: tuple of two floats
    '''
    t = self.gettstat(x,sx,x0)
    # Great and less than p-values
    pg = 1-tdist.cdf(t,self.dof)
    pl = tdist.cdf(t,self.dof)
    if h0=='greater':
      p=pg
    if h0=='less':
      p=pl
    if h0=='both':
      if t<0:
        p=2*pl
      else:
        p=2*pg
    return(t,p)

  def testSlope(self,gamma=0.05,h0='both',beta0=0.0,verbose=True):
    '''
    Perform a t-test to determine significance of the slope term. Here gamma 
    is the significance level (the default 0.05 corresponds to p<0.05), h0 
    can be both, greater or less and controls the sidedness of the test, 
    beta0 is the value to test against (default zero), and if verbose is set 
    to True (the default) will print test results.
    Returns two floats, the test statistic and the p-value.
    
    @param gamma: alpha value (significance critereon)
    @type gamma: float
    @param h0: sidedness, can be 'greater', 'less', or 'both'
    @type h0: float
    @param beta0: hypothesized intercept
    @type beta0: float
    @param verbose: if True print results
    @type verbose: bool
    @return: t-statistic and p-value
    @rtype: tuple of two floats
    '''
    t,p = self.test(self.beta,self.sbeta,gamma,h0,beta0)
    if verbose:
      print "Testing slope against hypothesis", h0, "found p of ", p, "."
      if p<gamma:
        print "Significant at the",gamma,"significance level."
      if p>=gamma:
        print "Not significant at the",gamma,"significance level."
    return(t,p)

  def testIntercept(self,gamma=0.05,h0='both',alpha0=0.0,verbose=True):
    '''
    Perform a t-test to determine significance of the intercept term. Here 
    gamma is the significance level (the default 0.05 corresponds to p<0.05), 
    h0 can be 'both', 'greater' or 'less' and controls the sidedness 
    of the test, alpha0 is the value to test against (default zero), and if 
    verbose is set to True (the default) will print test results.
    Returns two floats, the test statistic and the p-value.
    
    @param gamma: alpha value (significance critereon)
    @type gamma: float
    @param h0: sidedness, can be 'greater', 'less', or 'both'
    @type h0: float
    @param alpha0: hypothesized slope
    @type alpha0: float
    @param verbose: if True print results
    @type verbose: bool
    @return: t-statistic and p-value
    @rtype: tuple of two floats
    '''
    t,p = self.test(self.alpha,self.salpha,gamma,h0,alpha0)
    if verbose:
      print "Testing intercept against hypothesis", h0, "found p of ", p, "."
      if p<gamma:
        print "Significant at the",gamma,"significance level."
      if p>=gamma:
        print "Not significant at the",gamma,"significance level."
    return(t,p)

  def getSlopeCI(self,gamma=0.95,sided='both'):
    '''
    Calculate the confidence interval on the slope. gamma controls the 
    confidence level, default is 0.95 which gives a 95% CI. sided can be 'both',  less or greater for double sided 
    or single sided confidence intervals.
    Returns a tuple with two entries, which are either floats or None. The 
    entries are the lower and upper confidence intervals respectively and 
    None indicates that this bound is infinite for a single sided CI.
    
    @param gamma: confidence level
    @type gamma: float
    @param sided: sidedness, can be 'greater', 'less', or 'both'
    @type sided: float
    @return: confidence bounds on slope
    @rtype: tuple, two floats
    '''    
    if sided == 'both':
      tstar = tdist.ppf(1-((1-gamma)/2),self.dof)
      return(self.beta-tstar*self.sbeta,self.beta+tstar*self.sbeta)
    if sided == 'less':
      tstar = tdist.ppf(gamma,self.dof)
      return(self.beta-tstar*self.sbeta,None)
    if sided == 'greater':
      tstar = tdist.ppf(gamma,self.dof)
      return(None,self.beta+tstar*self.sbeta)


  def getInterceptCI(self,gamma=0.95,sided='both'):
    '''
    Calculate the confidence interval on the intercept. gamma controls the 
    confidence level, default is 0.95 which gives a 95% CI. sided can be 
    'both',  'less ' or 'greater' for double sided or single sided confidence 
    intervals.
    Returns a tuple with two entries, which are either floats or None. The 
    entries are the lower and upper confidence intervals respectively and 
    None indicates that this bound is infinite for a single sided CI.
    
    @param gamma: confidence level
    @type gamma: float
    @param sided: sidedness, can be 'greater', 'less', or 'both'
    @type sided: float
    @return: confidence bounds on intercept
    @rtype: tuple, two floats
    '''
    if sided == 'both':
      tstar = tdist.ppf(1-((1-gamma)/2),self.dof)
      return(self.alpha-tstar*self.salpha,self.alpha+tstar*self.salpha)
    if sided == 'less':
      tstar = tdist.ppf(gamma,self.dof)
      return(self.alpha-tstar*self.salpha,None)
    if sided == 'greater':
      tstar = tdist.ppf(gamma,self.dof)
      return(None,self.alpha+tstar*self.salpha)

  def predict(self,xn):
    '''
    Uses the co-efficients of the linear regression to predict the behaviour 
    of the dependent variable at the locations xn of the independent 
    variable. Returns an array the same shape as xn.
    
    @param xn: values of independent variable
    @type xn: float array
    @return: predicted values for response variable
    @rtype: float array
    '''
    return(self.alpha+self.beta*xn)

  def confidenceBands(self,xn,gamma=0.95):
    '''
    Constructs confidence bands at locations xn of the independent variable. 
    gamma controls the confidence level, default is 0.95 which gives a 95% 
    CB. Returns two arrays the same size as xn, one the lower confidence 
    band, the other the upper.
    
    @param xn: values of independent variable
    @type xn: float array
    @param gamma: confidence level
    @type gamma: float    
    @return: confidence bands for response variable
    @rtype: tuple of two float arrays
    
    '''
    yn = self.predict(xn)
    tstar = tdist.ppf(1-((1-gamma)/2),self.dof)
    Q = sqrt((self.RSS()/self.dof)*((1.0/self.n)+((xn-self.mux)**2.0/self.kappa)))
    return(yn+tstar*Q,yn-tstar*Q)

def linearRegression(x,y):
  '''
  Constructs a linear regression for the independent variable x and the 
  dependent variable y. See the L{LinearRegression} class for more 
  details.
  
  @param x: values for the independent variable
  @type x: float array
  @param y: values for the response variable
  @type y: float array
  @return: a linear regression
  @rtype: L{LinearRegression}
  '''
  rho = LinearRegression()
  rho.fit(x,y)
  return(rho)

if __name__=="__main__":
  import doctest
  doctest.testmod()

  try:
    # Make example figure
    from numpy import linspace
    from numpy.random import standard_normal,seed
    from pylab import (figure,plot,scatter,savefig,xlabel,ylabel,
      fill_between,legend)
    
    # Generate data
    alphaA = 1.0
    betaA = 2.0
    sigmaN = 0.5
    x = linspace(0,1,20)
    xint = linspace(0,1,1000)
    # Seed
    seed(0)
    y = alphaA + betaA*x + sigmaN*standard_normal(x.shape)
    rho = linearRegression(x,y)

    # Get confidence bandss
    yu,yd = rho.confidenceBands(xint)
    
    figure()
    plot(xint,rho.predict(xint),'b',label='linear regression')
    plot(xint,yu,'r',label='confidence band')
    plot(xint,yd,'r')
    fill_between(xint,yd,yu,color='r',alpha=0.2)
    scatter(x,y,marker='x',color='k',label='sample data')
    xlabel("dependent variable, x (arb)")
    ylabel("response variable, x (arb)")
    legend()
    savefig("regressionExample.png")
  except ImportError as e:
    print "Skipping example figure, this depends on numpy and pylab."