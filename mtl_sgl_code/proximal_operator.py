
import numpy as np
import time

def proximal_composition( v, ind, dim, ntasks ):
    '''
    Input:
        v: vector weights
        ind: index of the features' groups
        dim: problem dimension
        ntasks: number of tasks
    Output:
        w_new: update vector weights
    '''

    w_new = np.zeros( (dim,ntasks) )
    ngroups = ind.shape[1]-1  # number of groups

    # convert vector weights to a 2D representation (column format - Fortran)
    w2D = np.reshape(v,(dim,ntasks),order='F')

    # L21 + LG
    if ind[0,0] == -1:
        for i in xrange(0,dim):
            w_new[i,:] = epp2(w2D[i,:],ntasks,ind[2,0])

    for t in xrange(0,ntasks):
        for i in xrange(0,ngroups):
            temp = ind[:,i+1] #.astype(int)
            ids_group = np.arange(temp[0],temp[1], dtype=np.int)
            twoNorm = np.sqrt( np.dot(w_new[ids_group,t],w_new[ids_group,t]) )

            if twoNorm > temp[2]:
                w_new[ids_group,t] = w_new[ids_group,t] * (twoNorm-temp[2])/float(twoNorm)
            else:
                w_new[ids_group,t] = 0

    # reshape it back to a vector representation and return it
    return np.reshape(w_new,(dim*ntasks,), order='F')



def epp2( v, n, rho ):
    v2 = np.sqrt(np.dot(v,v))
    if rho >= v2:
        xk = np.zeros( n )
    else:
        ratio = (v2-rho)/float(v2)
        xk = v*ratio
    return xk



def compute_largest_group_norm(v,ind,dim,ntasks):
    lambda2_max = 0
    ngroups = ind.shape[1]  # number of groups
    w2D = np.reshape(v,(dim,ntasks),order='F')

    for t in xrange(0,ntasks):
        for i in xrange(0,ngroups):
            temp = ind[:,i]
            ids = np.arange(temp[0],temp[1], dtype=np.int)
            twoNorm = np.linalg.norm(w2D[ids,t]) / float( temp[2] )
            if twoNorm > lambda2_max:
                lambda2_max = twoNorm
    return lambda2_max



def proximal_average(w, ind, dim, ntasks):

    ngroups = ind.shape[1]-1  # number of groups
    w2D     = np.reshape(w,(dim,ntasks),order='F')
    w1_new  = np.zeros( w2D.shape )
    w2_new  = w2D.copy()

    # L2,1-norm penalization 
    if ind[0,0] == -1:
        for i in xrange(0,dim):  # applies l2,1-norm on matrix W
            w1_new[i,:] = epp2(w2D[i,:],ntasks,ind[2,0])

    # Group Lasso penalization
    for t in xrange(0,ntasks): # applies group lasso for each tas independently
        for i in xrange(0,ngroups):
            temp = ind[:,i+1] # +1 because there was a -1 column added as the first column
            ids_group = np.arange(temp[0],temp[1], dtype=np.int)
            twoNorm = np.sqrt( np.dot(w2D[ids_group,t],w2D[ids_group,t]) )

            # print np.linalg.norm(w2D[ids_group,t])
            if twoNorm > temp[2]:
                w2_new[ids_group,t] = w2_new[ids_group,t] * (twoNorm-temp[2])/float(twoNorm)
            else:
                w2_new[ids_group,t] = 0

    # average of L2,1 and Group lasso norms
    w_new = (w1_new+w2_new)/2.0

    return np.reshape(w_new,(dim*ntasks,), order='F')



