
import numpy.linalg as nplin
import numpy as np
import solve_fista as fista_new

_EPS_W = 1e-3
_MAXITER = 20

class MTSGL:
    """
    Multitask Sparse Group Lasso.
    """
    def __init__(self,glms):
        self.W = None
        if glms is None:
            self.ntasks = 0
            self.glms = None
        else:
            self.ntasks = len(glms)
            self.glms = glms

    def get_W(self):
        return self.W

    def predict( self, xtest ):
        yhat = list()

        for j in xrange(0,self.ntasks):
            yhat.append( self.__predict_glm( self.W[:,j], xtest, self.glms[j] ) )
        return yhat

    def __predict_glm( self, w, x, glm ):
                
        # f = 0
        nsamples = x.shape[0]
        mu = np.zeros(x.shape[0])
        for i in xrange(0,nsamples):
            if glm['glm'] == 'Gaussian':
                mu[i] = np.dot(x[i,:],w)
                if np.abs(glm['offset']) > 0:
                    mu[i] = glm['offset'] - mu[i]
                # f = f + scipy.stats.norm.logpdf( y[i], mu[i], sig )

            elif glm['glm'] == 'Poisson':
                # print 'Poisson'
                mu[i] = np.exp( np.dot(x[i,:],w) )
                if np.abs(glm['offset']) > 0:
                    mu[i] = np.maximum( glm['offset'] - mu[i], 0 ) # zero is the minimum value
                # f = f + scipy.stats.poisson.logpmf( int(y[i]), mu[i] )

            elif glm['glm'] == 'Gamma':
                # f = f + scipy.stats.gamma.logpdf( y[i], 1.0/np.dot(x[i,:],w) )
                mu[i] =  1.0/np.dot(x[i,:],w)
                
        return mu



    def fit( self, xtr, ytr, ind, r, opt_method ):

        if self.ntasks == 0:
            self.ntasks = len(ytr)

        dimension = xtr.shape[1]
        Wvec = np.random.randn( dimension*self.ntasks ) * 1e-4
        # apply offset in y's if needed    
        for k in xrange(0,self.ntasks):
            if np.abs(self.glms[k]['offset']) > 0:
                ytr[k] = self.glms[k]['offset'] - ytr[k]

        cont = 1
        while cont <= _MAXITER:
            W_old = Wvec
            Wvec= fista_new.SolveFISTA( Wvec, xtr, ytr, ind, r, self.glms, opt_method )
            diff_W = nplin.norm( Wvec - W_old )
            if diff_W < _EPS_W:
                break
            cont = cont + 1

        # print 'cont: %d'%(cont)
        self.W = np.reshape( Wvec,( dimension, self.ntasks ), order="F" )
          
