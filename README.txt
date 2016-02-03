Simon Wilshin's Regression module, swilshin@gmail.com, Jan 2016

This work was completed with support from the Royal Veterinary College, 
London (www.rvc.ac.uk).

A linear regression package with hypothesis testing, confidence bounds and 
confidence bands.

During the course of my research I discovered that while there are plenty of 
good pieces of software that will do linear regressions in python, they 
didn't have all the features I wanted, especially when it comes to confidence 
bands. So I've written my own. Features include:

 - Perform linear regression with hypothesis testing on the value of the 
   slope and intercept
 - Predict the response variable for a given value of the independent 
   variable
 - Confidence bounds on the parameters and confidence bands for the fit
 - Depends only on numpy and scipy
 - Inline testing and documentation through doctest and epydoc

doctests can be run on linux by running

epydoc regression -o regression/html

in the parent folder to the regression, and will generate html documentation 
in the html folder. To view this documentation open the 'index.html' file 
with the web browser of your choice.

The doctests can by run by executing linearregression.py as main, so in the 
parent of the regression folder run

python linearregression.py

If pylab is installed this will also produce the example figure for randomly 
generated data from the example in the documentation in this folder with the 
name 'regressionExample.png'.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
