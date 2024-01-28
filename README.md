# gpbo
This is a python module for gaussian process and bayesian optimization.  

## How to use
### gaussian process
Create a instance of gaussian process class defined in `./gp/__init__.py`.
### bayesian optimization
Use gaussian process for a prediction model. Calculate evaluation values and gradient by functions defined in `./bo/acquisition.py`. Update input values by gradient method(e.g., SGD) defined in `./bo/gradopt.py`.