cimport cython
from libc.stdlib cimport malloc, free

def CRP(counts, double gamma):
  cdef float *counts_c
  cdef int *zeros

  counts_c = <float *> malloc(len(counts)*cython.sizeof(float))
  for i in range(len(counts)):
    counts_c[i] = counts[i]

  cdef int len_counts = len(counts)
  zeros = <int *> malloc(len_counts*cython.sizeof(int))
  cdef int num_zeros = 0
  for i in range(len_counts):
    if (counts_c[i] == 0):
      zeros[i] = 1
      num_zeros += 1
    else:
      zeros[i] = 0
  cdef float avg_prob = 0.0
  #if num_zeros > 0:
  avg_prob = gamma / max(num_zeros, 1)
  cdef float sum_prob = 0.0
  for i in range(len_counts):
    if (zeros[i] == 1):
      counts_c[i] = avg_prob
    sum_prob += counts_c[i]
  cdef float inv_sum_prob = 1.0/sum_prob
  for i in range(len_counts):
    counts_c[i] *= inv_sum_prob

  probs = []
  for i in range(len_counts):
    probs.append(counts_c[i])

  return probs

def nCRP(counts, params):
  branching_factor = params['branching_factor']
  num_levels = params['num_levels']
  num_nodes = (branching_factor ** num_levels - 1) / (branching_factor - 1)
  num_paths = branching_factor ** (num_levels - 1)
  num_internal_nodes = num_nodes - num_paths
  probs = [1.0]
  counts = counts[1:]
  #print 'len counts:', len(counts)
  while len(probs) < num_paths:
    #print 'probs', probs, len(probs), len(counts)
    #raw_input()
    p = probs[0]
    probs = probs[1:]
    probs += map(lambda x: p*x, \
        CRP(counts[:params['branching_factor']], params['gamma']))
    counts = counts[(params['branching_factor']):]
  return probs
  
