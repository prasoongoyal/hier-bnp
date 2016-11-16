import sys
import numpy as np
from numpy.linalg import norm

beta_filename = sys.argv[1]

beta = np.loadtxt(beta_filename)

def EucDist(vec1, vec2):
  return norm(np.asarray(vec1) - np.asarray(vec2))/np.size(vec1)

np.set_printoptions(precision = 2)

avg_beta_diff = 0.0
for i in range(np.size(beta, 0)):
  for j in range(np.size(beta, 0)):
    euc_dist = EucDist(beta[i, :], beta[j, :]) 
    print ('%.3f' % euc_dist), '\t',
    avg_beta_diff += euc_dist
  print ''
print avg_beta_diff / (np.size(beta,0)**2)
