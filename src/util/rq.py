# Taken from https://leohart.wordpress.com/2010/07/23/rq-decomposition-from-qr-decomposition/
import numpy as np
from numpy.linalg import qr
 
def rq(A):
    '''Implement rq decomposition using QR decomposition'''
    Q,R = qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T
    return R[:,::-1],Q[::-1,:]