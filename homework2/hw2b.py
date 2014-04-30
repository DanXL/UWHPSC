
"""
Demonstration module for quadratic interpolation.
Update this docstring to describe your code.
Modified by: ** Dan Xiaolei **
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

def quad_interp(xi,yi):
    """
    Quadratic interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2.

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==3 and len(yi)==3, error_message

    # Set up linear system to interpolate through data points:
	#A = np.array([[1., xi[1], xi[1]**2],[1., xi[2], xi[2]**2], [1., xi[3], xi[3]**2]])
    A = np.vstack([np.ones(3), xi, xi**2]).T
    b = yi
    c = solve(A, b)
    print "The polynomial coefficients are:"
    print c
    ### Fill in this part to compute c ###

    return c



def plot_quad(xi, yi):
	"""
    takes two numpy arrays xi and yi of length3, calls quad_interp to compute c, and then
    plots both the interpolating polynomial and the data points, and saves the resulting
    figure as quadratic.png.
    """
	c = quad_interp(xi, yi)
	x = np.linspace(xi.min() - 1, xi.max() + 1, 1000)
	y = np.dot(c,np.vstack([np.ones(1000), x, x**2]))
	plt.figure(1)
	plt.clf()
	plt.plot(x,y,'b-')
	plt.plot(xi, yi, 'ro')
	plt.title("Data points and interpolating polynomial")
	plt.savefig('quadratic.png')





def test_quad1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2.])
    yi = np.array([ 1., -1.,  7.])
    c = quad_interp(xi,yi)
    c_true = np.array([-1.,  0.,  2.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)

def test_quad2():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  3.,  2.])
    yi = np.array([ 1., 9.,  7.])
    c = quad_interp(xi,yi)
    c_true = np.array([3., 2.,0])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)
        
if __name__=="__main__":
    # "main program"
    # the code below is executed only if the module is executed at the command line,
    #    $ python demo2.py
    # or run from within Python, e.g. in IPython with
    #    In[ ]:  run demo2
    # not if the module is imported.
    print "Running test..."
    test_quad1()








def cubic_interp(xi,yi):
    """
    Cubic interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,3.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3.

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 4"
    assert len(xi)==4 and len(yi)==4, error_message

    # Set up linear system to interpolate through data points:
	#A = np.array([[1., xi[1], xi[1]**2],[1., xi[2], xi[2]**2], [1., xi[3], xi[3]**2]])
    A = np.vstack([np.ones(4), xi, xi**2, xi**3]).T
    b = yi
    c = solve(A, b)
    print "The polynomial coefficients are:"
    print c
    ### Fill in this part to compute c ###

    return c



def plot_cubic(xi, yi):
	"""
    takes two numpy arrays xi and yi of length 4, calls quad_interp to compute c, and then
    plots both the interpolating polynomial and the data points, and saves the resulting
    figure as cubic.png.
    """
	c = cubic_interp(xi, yi)
	x = np.linspace(xi.min() - 1, xi.max() + 1, 1000)
	y = np.dot(c,np.vstack([np.ones(1000), x, x**2, x**3]))
	plt.figure(1)
	plt.clf()
	plt.plot(x,y,'b-')
	plt.plot(xi, yi, 'ro')
	plt.title("Data points and interpolating polynomial")
	plt.savefig('cubic.png')





def test_cubic1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2., 3.])
    yi = np.array([ 1., -1.,  7., 9.])
    c = cubic_interp(xi,yi)
    c_true = np.array([-1.,  1.33333333,  2.66666667, -0.66666667])
    print "c =      ", c
    print "c_true = ", c_true
    plot_cubic(xi, yi)
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)







def poly_interp(xi,yi, n):
    """
    polynomial interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,3,...,n.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3 + ... + c[n-1]*x**(n-1).

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length %s" % (n)
    assert len(xi)==n and len(yi)==n, error_message

    # Set up linear system to interpolate through data points:
	#A = np.array([[1., xi[1], xi[1]**2],[1., xi[2], xi[2]**2], [1., xi[3], xi[3]**2]])
    A = []
    for j in range(n):
    	A.append(xi**j)
    A = np.array(A).T
    b = yi
    c = solve(A, b)
    print "The polynomial coefficients are:"
    print c
    ### Fill in this part to compute c ###

    return c



def	plot_poly(xi, yi,n):
	"""
    takes two numpy arrays xi and yi of length 4, calls quad_interp to compute c, and then
    plots both the interpolating polynomial and the data points, and saves the resulting
    figure as cubic.png.
    """
	c = poly_interp(xi, yi,n)
	x = np.linspace(xi.min() - 1, xi.max() + 1, 1000)
	A = [np.ones(1000)]
	# Horner's rule
	y = c[n-1]
	for j in range(n-1, 0, -1):
		y = y*x + c[j-1]
	#A = np.array(A)
	#y = np.dot(c, A)
	plt.figure(1)
	plt.clf()
	plt.plot(x,y,'b-')
	plt.plot(xi, yi, 'ro')
	plt.title("Data points and interpolating polynomial")
	plt.savefig('poly.png')





def test_poly1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2., 3.])
    yi = np.array([ 1., -1.,  7., 9.])
    n = 4
    c = poly_interp(xi,yi,n)
    c_true = np.array([-1.,  1.33333333,  2.66666667, -0.66666667])
    print "c =      ", c
    print "c_true = ", c_true
    plot_poly(xi, yi,n)
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)



def test_poly2():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2., 3., 4.])
    yi = np.array([ 1., -1.,  7., 9., 17.])
    n = 5
    c = poly_interp(xi,yi,n)
    c_true = np.array([-1.,  3.23333333,  2.98333333, -1.93333333, 0.316666667])
    print "c =      ", c
    print "c_true = ", c_true
    plot_poly(xi, yi,n)
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)


