import numpy as np
import itertools

# forward-backward algorithm is also known as sum_product algorithm.

def potential_func(w, vfeature_func, sentence):
    """w: numpy array of weights for the features
       vfeature_func: Vectorised feature function generated by crf_feature.make_vfeature_func
       sentence: List of words.
       
       sentence is assumed to have low-frequency words already substituted for #UNK#
       
       returns a function that calculates:
       potential: e^(weight . features(input))
    """
    def potential(yp, y, i): # Potential of state yp to y at index i.
        features = vfeature_func(yp, y, i, sentence)
        return np.exp(np.dot(w, features))
    
    return potential

def forward_backward(potential, sentence, states):
    """potential: Potential function e^(weight . features)
       sentence: List of words.
       state: list of states/tags. Don't include START/STOP.
       
       sentence is assumed to have low-frequency words already substituted for #UNK#
       
       Returns 2D numpy array of [[state1_score, state2_score...], ...]
           # row = index, col = state.
           # Each row is index of sentence.
           # Each column is the score for that state.
           # !!NOTE!! Order of columns follows order of the states argument.
    """
    statelen = len(states)
    
    # Calculate alpha scores
    alpha_table = np.ndarray((len(sentence), statelen))
    # Initial forward step.
    for j, u in enumerate(states):
        alpha_table[0,j] = potential("START", u, 0)
    # subsequent forward steps
    for i in range(1,len(sentence)):
        for j, u in enumerate(states):
            alpha_table[i,j] = np.sum( alpha_table[i-1, k]*potential(v, u, i) for k,v in enumerate(states) )
    
    # Calculate beta scores
    beta_table = np.ndarray((len(sentence), statelen))
    # Initial backward step.
    i = len(sentence)-1
    for j, u in enumerate(states):
        beta_table[i, j] = potential(u, "END", None)
    # subsequent backward steps
    for i in range(len(sentence)-2, -1, -1):
        for j, u in enumerate(states):
            beta_table[i,j] = np.sum( beta_table[i+1, k]*potential(u, v, i+1) for k,v in enumerate(states) )
    
    return alpha_table, beta_table



def calc_z_marginals(alpha_table, beta_table, potential, states):
    """Calculate:
       1. normalization constant Z = sum of all potentials.
       2. marginal probability P(y_{i-1}=u, y_i=v |x) for each value of i, u and v.
          (except at START/STOP edges, i.e. from i=0 to len(sentence)-1)
       
       returns:
        1. Scalar value z
        2. a 3D numpy ndarray of [index (i,u,v): probability]
    """
    # Calculate normalization constant
    # Any index should work
    last_a = alpha_table[-1]
    last_b = beta_table[-1]
    z = np.sum(last_a*last_b)
    
    # Calculate marginals
    marginals = np.ndarray((len(alpha_table), len(states), len(states)))
    for k in range(len(alpha_table)-1):
        for (i,u),(j,v) in itertools.product(enumerate(states), repeat=2):
            marginals[k,i,j] = alpha_table[k,i]*potential(u,v,k+1)*beta_table[k, j]/z
    k = len(alpha_table)-1
    for (i,u),(j,v) in itertools.product(enumerate(states), repeat=2):
        marginals[k,i,j] = alpha_table[k,i]*potential(u,v,None)/z
    
    return z, marginals