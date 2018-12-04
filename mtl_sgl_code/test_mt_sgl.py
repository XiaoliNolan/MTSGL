

import mt_sgl_run

glm = 'gaussian' # glm model: 'gaussian','poisson','2g3p'
nruns =  50 # number of runs
opt_method='proximal_average' # opt methods: 'proximal_average', 'proximal_composition'

rmse,wr = mt_sgl_run.mt_sgl( glm, nruns, opt_method)
print rmse
print wr