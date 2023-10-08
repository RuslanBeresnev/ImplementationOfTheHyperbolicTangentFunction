# Implementation of the Hyperbolic Tangent Function
- Manual implementation of the hyperbolic tangent function module using the Python standard library
- It was implemented as a test task for the project "Implementation of Triton kernels to accelerate the work of neural networks"

# Results
- Approximate results for the number 0.65748 (in milliseconds):
--------------------------------------------------
			     tanh() 	 tanh_differential()
Manual: 	 0.36 		 0.37
NumPy: 		 0.05 		 0.01
--------------------------------------------------
The tanh() function in the NumPy module approximately is 7.2 times faster than the manual implementation
The tanh_differential() function in the NumPy module approximately is 36.9 times faster than the manual implementation



- Approximate results for the matrix 100 x 100 (in milliseconds):
--------------------------------------------------
			     tanh() 	 tanh_differential()
Manual: 	 56.82 		 66.29
NumPy: 		 1.96 		 2.07
--------------------------------------------------
The tanh() function in the NumPy module approximately is 29.0 times faster than the manual implementation
The tanh_differential() function in the NumPy module approximately is 32.0 times faster than the manual implementation
