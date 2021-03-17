import math
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------
def _check_deriv(deriv, indexes):
	if deriv < 0:
		raise ValueError('Derive degree must be positive integer')
	if deriv >= len(indexes):
		raise ValueError('Derive degree must be smaller than length of the array')

#--------------------------------------------------------------------------------------
def _check_acc(acc):
	if acc % 2 == 1 or acc <= 0:
		raise ValueError('Accuracy order acc must be positive EVEN integer')

#--------------------------------------------------------------------------------------
def Coefficients(der, index = 'centered', acc=2):
	"""
	Calculates the finite difference coefficients for given derivative.
	  : param der   : The derivative order.
	  : type deriv  : int > 0
	  : param index : The offsets for which to calculate the coefficients.
	  : type index  : list of ints
	  : param acc   : The accuracy order. Taken into account only when index='centered'
      : type acc    : even int > 0:
	  : return      : dict with the finite difference coefficients, index and acc
	"""

	if str(index) =='centered': 
		_check_acc(acc)
		val_init = int(acc/2 + int(0.5*(der-1)))
		index    = np.arange(-1*val_init, val_init + 1)
	
	_check_deriv(der, index)
		
	index = np.array(index)
		
	results = {}
	results['index'] = index

	mA = np.array([index**i for i in range(len(index))])

	mB      = np.zeros_like(index)
	mB[der] = math.factorial(der)

	coeff = np.linalg.solve(mA, mB)
	
	results['coeff'] = coeff
	results['order'] = len(index)-der
	if str(index) =='centered': results['order'] += 1

	return results
	
#--------------------------------------------------------------------------------------
def der_fun(fun, dx, der, index = 'centered', acc=2, boundary=True, printData=False):
	"""
	We compute the derivative of a function using Finite Difference Method
	  : param fun		: Function to be derived
	  : type fun		: numpy list
	  : param dx		: spacing of the grid
	  : type dx			: double
	  : return			: dict with the finite difference coefficients, index and acc
	  
	  Optionals
	  : param index		: The indices for which to calculate the coefficients.
	  : type index		: list of ints. DEFAULT: 'centered'
	  : param acc		: The accuracy order. Taken into account only when index='centered'
      : type acc		: even int > 0.  DEFAULT: 2
      : param boundary	: Coumpute Boundary data
      : type boundary	: bool, True(Default) or False. 
      : param printData	: print on the screen coefficients 
      : type printData	: bool, True or False(Default)
	"""

	derivative = np.zeros_like(fun)
	der_data   = Coefficients(der, index, acc)
	
	i_start  = abs(der_data['index'][ 0])
	i_finish = abs(der_data['index'][-1])
	
	for i in range(i_start, len(fun)-i_finish):
		derivative[i] = np.sum(fun[der_data['index'] + i]*der_data['coeff'])/dx**der
	if printData: print(der_data)
		
	if boundary:
	
		val_boun = der + 1 + der_data['order'] - 1
	
		for i in range(i_start):
			der_data_left = Coefficients(der, np.arange(val_boun + 1) - i)
			derivative[i] = np.sum(fun[der_data_left['index'] + i]*der_data_left['coeff'])/dx**der
		
		for j in range(i_finish):
			der_data_right = Coefficients(der, np.arange(-val_boun, 1) + j)
			i = len(derivative) - 1 - j
			derivative[i] = np.sum(fun[der_data_right['index'] + i]*der_data_right['coeff'])/dx**der
			
		if printData: 
			print(der_data_left)
			print(der_data_right)

	return derivative

#=====================================================================================
"""
In the following we check the above code as an example
We compute the derivative of 'Func', and plot the results
"""

x_range = np.linspace(0, 10, 1000)
dx      = abs(x_range[1] - x_range[0])
Func    = np.sin(x_range)
der     = der_fun(Func, dx, 1, index='centered', acc=2, boundary=True)

plt.plot(x_range, Func)
plt.plot(x_range, der)
plt.plot(x_range, np.cos(x_range))
plt.show()

