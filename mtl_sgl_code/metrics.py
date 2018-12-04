import numpy as np 
import scipy.stats as ssts
import matplotlib.pyplot as plt
import scipy.special
import math


def compute_performance_metrics( yhat, ytrue, glm, p ):
	rmse  = np.zeros( len(yhat) )
	nrmse = np.zeros( len(yhat) )
	wr = np.zeros( len(yhat) )
	adj_r_squared = np.zeros( len(yhat) )
	r_squared = np.zeros( len(yhat) )
	loglik = np.zeros( len(yhat) )
	aic = np.zeros( len(yhat) )
	bic = np.zeros( len(yhat) )
	nmse2 = np.zeros( len(yhat) )
	for k in xrange(0,len(yhat)):
		rmse[k] = np.sqrt( np.mean( np.square(yhat[k]-ytrue[k]) ) )   # roor mean squared error
		nrmse[k] = np.mean( np.square(yhat[k]-ytrue[k]) ) / float(np.var(ytrue[k]))
		wr[k] = np.corrcoef( yhat[k], ytrue[k] )[0,1]
		r_squared[k], adj_r_squared[k] = compute_adj_r_squared( yhat[k], ytrue[k], p )
		loglik[k],aic[k],bic[k] = compute_loglik_aic_bic( yhat[k], ytrue[k], glm[k], p )
		nmse2[k] = np.square(np.linalg.norm(yhat[k]-ytrue[k]))/float(np.std(ytrue[k]));
	return rmse,nrmse,wr,r_squared,adj_r_squared, loglik, aic, bic,nmse2

def compute_adj_r_squared( yhat, yobs, p=327 ):
	n = len(yhat)
	sse = np.square(yobs-yhat).sum()
	sst = np.square(yobs-np.mean(yobs)).sum()
	r_squared = 1 - (sse / float(sst))
	adj_r_squared = 1 - (1-r_squared)*((n-1)/float(n-p-1))
	return r_squared, adj_r_squared

def compute_loglik_aic_bic( yhat, yobs, glm, p ):
	nsamples = len(yhat)
	if glm['glm'] == 'Gaussian':
		# sig = np.dot( e.T, e )/float(x_tr.shape[0])  # error variance estimation
		e = yobs - yhat
		loglik = -(nsamples/2.0)*(np.log(np.dot( e.T, e )) + (1+np.log(2*math.pi/nsamples)))
	elif glm['glm'] == 'Poisson':
		scale = 1
		loglik = scale * (-yhat + yobs*np.log(yhat) - scipy.special.gammaln(yobs+1)).sum() 

	aic = -2*loglik + 2*p
	bic = -2*loglik + p*np.log(nsamples)
	return loglik,aic,bic
