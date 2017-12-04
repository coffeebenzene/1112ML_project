import itertools
import numpy as np



def bind_t(u, v):
    def t_feature(yp, y, i, x):
        return yp==u and y==v
    t_feature.__qualname__ = "t_f {}->{}".format(u, v)
    return t_feature

def bind_e(u, a):
    def e_feature(yp, y, i, x):
        return (i is not None) and y==u and x[i]==a
    e_feature.__qualname__ = "e_f {}->{}".format(u, a)
    return e_feature

def generate_hmm_features(tags, words):
    """Takes in 2 iterables, tags and words.
       Generate HMM features for the equivalent CRF model.
    """
    features = []
    
    # Transition features
    tags_no_start = [u for u in tags if u!="START"]
    tags_no_end = [u for u in tags if u!="END"]
    for u, v in itertools.product(tags_no_end, tags_no_start):
        # Transition from u to v.
        t_feature = bind_t(u, v)
        features.append(t_feature)
    
    # Emission features
    for u, a in itertools.product(tags, words):
        # Emission from u to a.
        e_feature = bind_e(u, a)
        features.append(e_feature)
    
    return features



#def make_vfeature_func(features):
#    feature_length = len(features)
#    def vfeature_func(yp, y, i, x):
#        """Vectorized feature function.
#           Each feature is a function in the form of f(yp, y, i, x) => 0 or 1.
#           Returns a vector of each feature applied to the input.
           
#           yp is previous state/tag
#           y is current state/tag
#           i is current index
#           x is entire sentence
#        """
#        vfeature = np.empty(feature_length)
#        for j,f in enumerate(features):
#            vfeature[j] = (f(yp, y, i, x))
#        return vfeature
#    return vfeature_func



def make_vfeature_func(tags, words):
    
    tags_no_end = [u for u in tags if u!="STOP"]
    tags_no_start = [u for u in tags if u!="START"]
    
    prev_tag_map = {tag:i*len(tags_no_start) for i, tag in enumerate(tags_no_end)}
    curr_tag_map = {tag:i for i, tag in enumerate(tags_no_start)}
    
    transition_feature_len = len(tags_no_end) * len(tags_no_start)
    
    words = list(words)
    word_map = {word:i+transition_feature_len for i, word in enumerate(words)}
    emission_feature_len = len(words)
    
    feature_length = transition_feature_len + emission_feature_len
    def vfeature_func(yp, y, i, x):
        """Vectorized feature function.
           Each feature is a function in the form of f(yp, y, i, x) => 0 or 1.
           Returns a vector of each feature applied to the input.
           
           yp is previous state/tag
           y is current state/tag
           i is current index
           x is entire sentence
        """
        
        vfeature = np.zeros(feature_length)
        
        t_indx = prev_tag_map[yp] + curr_tag_map[y] # Transition parameter index
        vfeature[t_indx] = 1
        
        if i is not None:
            e_indx = word_map[x[i]]# Emission parameter index
            vfeature[e_indx] = 1
        
        return vfeature
    return vfeature_func, feature_length