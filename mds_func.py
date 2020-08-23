from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import array as arr

def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
                                                                                               
    """
    # Number of points                                                                        
    n = len(D)
    #print('n=',n)
 
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
    #print("Centering matrix=",H)
 
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
    #print("Original Eigenvalues= ",evals)
    #print("Original Eigenvectors= ",evecs)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    #print("idx= ",idx)
    evals = evals[idx]
    evecs = evecs[:,idx]
    #print("Changed Eigenvalues= ",evals)
    #print("Changed Eigenvectors= ",evecs)
 
    # Compute the coordinates using positive-eigenvalued components only                      
    w, = np.where(evals>0)
    #print('w=',w)
    L  = np.diag(np.sqrt(evals[w]))
    #print('L=',L)
    V  = evecs[:,w]
    #print('V=',V)
    Y  = V.dot(L)
 
    return Y, evals
