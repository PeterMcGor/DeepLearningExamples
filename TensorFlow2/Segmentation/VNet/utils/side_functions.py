import inspect
from collections.abc import Iterable 
import numpy as np

def extract_kwargs(fn, kwargs):
    function_parameters = inspect.signature(fn).parameters.keys()
    return {k:kwargs[k] for k in kwargs if k in function_parameters}

def concat_fn(fns, *params, **kwargs):
    for f in fns:
        params = [params] if (not isinstance(params, Iterable) or isinstance(params, (np.ndarray, np.generic)) ) else params
        fn_params = list(inspect.signature(f).parameters.keys())
        params = { fn_params[p]:params[p] for p in range(len(params)) }
        f_kwargs = extract_kwargs(f, kwargs)
        #In case of coinicide retain the args value
        params = f(**{**f_kwargs, **params})

    
    return [params] if (not isinstance(params, Iterable) or isinstance(params, (np.ndarray, np.generic)) ) else params
      
    #return params


if __name__ == '__main__':
  
    def f_a(a, a2 = None):
        print("From A", a, a2)
    def f_b(b, b2 = None):
        print("From B", b, b2)
    
    def f(f_param, **kwargs):
        f_a_params = extract_kwargs(f_a, kwargs)
        f_b_params = extract_kwargs(f_b, kwargs)
        print(f_a_params)
        print(f_b_params)
        f_a(**f_a_params)
        f_b(**f_b_params)
        print("FROM F", f_param)
    
    def c_1(a, b, h=3):
        return a+1, b-1, h+10
    def c_2(a,b,h):
        return a+b+h
        
    f("F", a = "A", b = "b", z = "K")
    
    print(concat_fn([c_1, c_2], 1,2, h=70, a = 100)) # a is overwritted by positional arguments, 100 does nothing# h is ignored in the second one cause is positional
    
    def trans(arr):
        return np.transpose(arr)
    def inv(arr):
        return arr.T
    
    a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    b = concat_fn([trans, inv], a )