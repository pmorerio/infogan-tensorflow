import numpy.random as npr
import numpy as np
import tensorflow as tf

def sample_Z(m, n, mode='uniform'):
	if mode=='uniform':
		return npr.uniform(-1., 1., size=[m, n])
	if mode=='gaussian':
		return np.clip(npr.normal(0,0.1,(m,n)),-1,1)


def sample_cat(m,n):
    
    cat = npr.randint(0,n, size=m)
    out = np.zeros((m,n))
    out[np.arange(m),cat]=1
    return out.astype(float)
    
def sample_attribute(m):
    
    #random int 0 or 1
    out = npr.randint(0,2, size=m)
    return out.astype(float)

def lrelu(input, leak=0.2, scope='lrelu'):
    
    return tf.maximum(input, leak*input)
  
if __name__=='__main__':
	
	print('empty')

