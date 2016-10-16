import sys
import numpy as np
from numpy.random import dirichlet, multinomial
from math import exp, log, factorial
from scipy.special import gamma as gamma_fn
from scipy.special import gammaln as log_gamma_fn
from copy import copy, deepcopy
import random

def path_len(path_str):
  return path_str.count('-')+1

def get0tol(path_str, level):
  nodes = path_str.split('-')
  return '-'.join(nodes[:level])

def read_data(filename):
  with open(filename) as f:
    lines = f.readlines()
    values = map(lambda x:map(eval, x.split()), lines)
    max_cols = max(map(lambda x:len(x), values))
    values = map(lambda x:x + [-1]*(max_cols - len(x)), values)
    return np.asarray(values)

def initialize_parameters(W, gamma, alpha, vocab_size):
  params = {}
  params['gamma'] = gamma
  params['alpha'] = alpha
  params['vocab_size'] = vocab_size
  params['num_levels'] = 4
  params['branching_factor'] = 5
  params['logbeta'] = log(vocab_size * gamma_fn(alpha)) - \
      log_gamma_fn(vocab_size * alpha)
  params['newpathprob'] = - vocab_size * (alpha - 1) * log(vocab_size) - \
      params['logbeta']
  (num_docs, num_words) = np.shape(W)
  params['num_docs'] = num_docs
  params['num_words'] = num_words
  params['max_iter'] = 1000
  params['temperature'] = 10.0
  return params

# Gets a flat array of counts in each cluster. Returns the probability of each
# cluster.
def CRP(counts, gamma):
  prob = [0] * len(counts)
  zeros = []
  for i in range(len(counts)):
    prob[i] = counts[i]
    if counts[i] == 0:
      zeros.append(i)
  if len(zeros) > 0:
    avg_prob_new_path = gamma / len(zeros)
    for z in zeros:
      prob[z] = avg_prob_new_path
  prob_sum = sum(prob)*1.0
  prob = map(lambda x:x/prob_sum, prob)
  return prob

# Gets an array which represents counts of nodes in the hierarchy. Returns the
# probability of each path
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

def sample_Z(W_d, Z_d, beta, n, counts, params):
  branching_factor = params['branching_factor']
  num_levels = params['num_levels']
  num_nodes = (branching_factor ** num_levels - 1) / (branching_factor - 1)
  num_paths = branching_factor ** (num_levels - 1)
  num_internal_nodes = num_nodes - num_paths
  # remove current Z from counts
  idx = Z_d[n] + num_internal_nodes
  while idx >= 0:
    counts[idx] -= 1
    idx = (idx - 1) / branching_factor
  
  prior = nCRP(counts, params)
  #print 'counts', len(counts)
  #print 'Prior', len(prior)
  #print zip(counts[-125:], prior)
  #raw_input()
  posterior = prior[:]
  for k in range(len(posterior)):
    posterior[k] *= beta[k][int(W_d[n])]
  posterior_sum = sum(posterior) * 1.0
  posterior = map(lambda x:x/posterior_sum, posterior)
  Z_dn = multinomial(1, make_distr_peaked(posterior, \
      params['temperature'])).tolist().index(1)
  return Z_dn

def computeCounts(Z_d, num_levels, branching_factor):
  num_nodes = (branching_factor ** num_levels - 1) / (branching_factor - 1)
  num_paths = branching_factor ** (num_levels - 1)
  num_internal_nodes = num_nodes - num_paths
  counts = [0] * num_paths
  for n in Z_d:
    counts[n] += 1
  counts = [0] * num_internal_nodes + counts
  for n in range(len(counts)-1, 0, -1):
    counts[(n-1)/branching_factor] += counts[n]
  return counts

def make_distr_peaked(values, temperature):
  #print temperature
  new_values = map(lambda x:x**(1.0 / temperature), values)
  if 1.0 / temperature > 500:
    # numerical issues with direct calcuations
    new_values = map(lambda x: 1.0 if (x == max(values)) else 0.0, values)
  norm = sum(new_values)
  new_values = map(lambda x:x/norm, new_values)
  #print values, new_values
  #raw_input()
  return new_values

def run_gibbs_sampler(W, Z, beta, params):
  branching_factor = params['branching_factor']
  num_levels = params['num_levels']
  num_nodes = (branching_factor ** num_levels - 1) / (branching_factor - 1)
  num_paths = branching_factor ** (num_levels - 1)
  num_internal_nodes = num_nodes - num_paths
  for iteration in range(params['max_iter']):
    # sample Z
    for d in range(len(Z)):
      # compute counts for this document
      counts = computeCounts(Z[d], params['num_levels'], params['branching_factor'])
      #print Z[d], counts
      #raw_input()
      for n in range(len(Z[0])):
        Z[d][n] = sample_Z(W[d], Z[d], beta, n, counts, params)
    # sample beta
    all_docs_counts = []
    for path in range(num_paths):
      all_docs_counts.append([params['alpha']] * params['vocab_size'])
    for d in range(len(Z)):
      for n in range(len(Z[0])):
        all_docs_counts[Z[d][n]][W[d][n]] += 1
    for path_idx in range(len(beta)):
      beta[path_idx] = dirichlet(all_docs_counts[path_idx])

def read_data(filename):
  with open(filename) as f:
    lines = f.readlines()
    values = map(lambda x:map(eval, x.split()), lines)
    max_cols = max(map(lambda x:len(x), values))
    values = map(lambda x:x + [-1]*(max_cols - len(x)), values)
    return np.asarray(values)

def main():
  #initialize parameters
  W = read_data(sys.argv[1])
  gamma = eval(sys.argv[2])
  alpha = eval(sys.argv[3])
  vocab_size = eval(sys.argv[4])

  params = initialize_parameters(W, gamma, alpha, vocab_size)
  
  # create a list of all paths and all nodes
  queue = ['0']
  all_paths = []
  while len(queue) > 0:
    q = queue[0]
    queue = queue[1:]
    if path_len(q) == params['num_levels']:
      all_paths.append(q)
    else:
      for i in range(params['branching_factor']):
        queue.append(q + '-' + str(i))
  #print all_paths
  #print len(all_paths)
  num_paths = len(all_paths)

  # initialize beta for each path
  beta = []
  for path in range(len(all_paths)):
    beta.append(dirichlet([alpha] * params['vocab_size']))

  # initialize Z
  Z = []
  for d in range(params['num_docs']):
    curr_doc = []
    for n in range(params['num_words']):
      path_idx = random.randint(0, num_paths-1)
      curr_doc.append(path_idx)
    Z.append(curr_doc)

  #print Z
  #print all_nodes_count

  best_Z, best_beta = run_gibbs_sampler(W, Z, beta, params)
  print 'best_Z', best_Z
  print 'best_beta', best_beta

if __name__=="__main__":
  main()
