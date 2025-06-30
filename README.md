# Implementation of the Hyperbolic Tangent Function
- Manual implementation of the hyperbolic tangent function module using the Python standard library
- It was implemented as a test task for the project "Implementation of Triton kernels to accelerate the work of neural networks"

# Results
- Approximate results of calculation performance (TOTAL TIME IN MILLISECONDS FOR 1000 LAUNCHES) for the number 0.78397, and comparison of the speed of implementation via NumPy and manual implementation: 

| Implementation\Function | tanh() | tanh_diff() |
| ----------------------- | ------ |-------------|
|          Manual         |  2.21  | 3.0         |
|          NumPy          |  6.18  | 7.99        |
| NumPy is n times faster |  0.4   | 0.4         |

- Approximate results of calculation performance (TIME IN MILLISECONDS FOR ONE LAUNCH) for the 50 by 100 matrix and comparison of the speed of implementation via NumPy and manual implementation:

| Implementation\Function | tanh() | tanh_diff() |   f()   | f_diff() |
| ----------------------- | ------ | ----------- | ------- |----------|
|          Manual         |  9.11  |    15.95    | 1504.53 | 1364.45  |
|          NumPy          |  0.9   |     2.7     |   18.3  | 37.89    |
| NumPy is n times faster |  10.1  |     5.9     |   82.2  | 36.0     |
