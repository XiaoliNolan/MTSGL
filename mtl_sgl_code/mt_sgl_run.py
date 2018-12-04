
import numpy as np
from MTSGL import MTSGL
from metrics import compute_performance_metrics

def mt_sgl( glm_names, nruns, opt_method, rho=0.1 ):

    X = np.loadtxt('./data/X.txt')
    Y = np.loadtxt('./data/Y.txt')

    nfeatures = len(X[0])
    ntasks = len(Y[0])
    ind = [0,12,25,32,40,50]

    groups_index = np.zeros( (3,len(ind)-1) )
    for i in xrange(0,len(ind)-1):
        groups_index[:,i] = np.array( (ind[i],ind[i+1],np.sqrt(ind[i+1]-ind[i]) ) )
    #print groups_index

    if glm_names == 'gaussian':
        glms = ({'glm':'Gaussian','offset': 0},   # task 1 ADAS
                {'glm':'Gaussian','offset': 0},   # task 2 MMSE P
                {'glm':'Gaussian','offset': 0},   # task 3 TOTOA 
                {'glm':'Gaussian','offset': 0},   # task 4 T30 P
                {'glm':'Gaussian','offset': 0} )  # task 5 RECOG P  

    elif glm_names == 'poisson':
        glms = ({'glm':'Poisson','offset': 0},   # task 1
                {'glm':'Poisson','offset': 0},    # task 2
                {'glm':'Poisson','offset': 0},   # task 3
                {'glm':'Poisson','offset': 0},   # task 4
                {'glm':'Poisson','offset': 0} )  # task 5

    elif glm_names == '2g3p':
        glms = ({'glm':'Gaussian','offset': 0},   # task 1
                {'glm':'Poisson','offset': 0},    # task 2
                {'glm':'Gaussian','offset': 0},   # task 3
                {'glm':'Poisson','offset': np.max(data[:,4])},   # task 4
                {'glm':'Poisson','offset': 0} )  # task 5
        

    rmse_runs = np.zeros( (nruns,ntasks) )
    nmse_runs = np.zeros( (nruns,ntasks) )
    wr_runs = np.zeros( (nruns,ntasks) )
    r_squared_runs = np.zeros( (nruns,ntasks) )
    adj_r_squared_runs = np.zeros( (nruns,ntasks) )

    loglik_runs = np.zeros( (nruns,ntasks) )
    aic_runs = np.zeros( (nruns,ntasks) )
    bic_runs = np.zeros( (nruns,ntasks) )

    wri = np.zeros(nruns)
    nmsei = np.zeros(nruns)

    # maximum number of algorithm runs in parameter selection
    IDS_TR = 450
    best_rhos = np.zeros((nruns,1))
    for i in xrange( 0, nruns ):
    
        print "Run %d" % ( i+1 )

        y_tr = Y[:IDS_TR]
        y_ts = Y[IDS_TR:]
        
        x_tr = X[:IDS_TR]
        x_ts = X[IDS_TR:]
        
        ytrain = list()
        ytest = list()

        for k in xrange(0, ntasks):
            ytrain.append( y_tr[:,k] )
            ytest.append( y_ts[:,k] )

        reg = MTSGL(glms)
        reg.fit( x_tr, ytrain, groups_index, rho, opt_method=opt_method)

        yhat = reg.predict( x_ts )

        rmse,nrmse,wr,_,_,loglik,aic,bic,nmsek = compute_performance_metrics( yhat, ytest, glms, nfeatures )

    return rmse,wr
