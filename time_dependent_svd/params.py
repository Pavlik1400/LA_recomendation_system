import numpy as np

# hyper parameters
LEARNING_RATES = {
    'b_u': 5e-3,
    'alpha_u': 1e-4,
    'b_i': 2e-2,
    'c_u': 5e-3,
    'b_ut': 2e-2,
    'c_ut': 2e-2,
    'b_ibin': 1e-4,
    'p': 5e-3,
    'q': 5e-3,
    'alpha_p': 1e-4
}

REGULARIZATIONS = {
    'b_u': 3e-5,
    'alpha_u': 5e-3,
    'b_i': 3e-3,
    'c_u': 3e-5,
    'b_ut': 5e-2,
    'c_ut': 5e-2,
    'b_ibin': 1e-4,
    'p': 5e-4,
    'q': 5e-4,
    'alpha_p': 1e-4,
}


class Parameters(object):
    alpha_p = dict()
    alpha_u = dict()
    b_i = dict()
    b_ibin = np.array([])
    b_u = dict()
    b_ut = np.array([])
    c_u = dict()
    c_ut = np.array([])
    p = dict()
    q = dict()

