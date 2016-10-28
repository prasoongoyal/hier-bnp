import sys
import numpy as np
from numpy.random import dirichlet, multinomial, exponential
from math import exp, log, factorial
from scipy.special import gamma as gamma_fn
from scipy.special import gammaln as log_gamma_fn
from copy import copy, deepcopy
import random
from joblib import Parallel, delayed

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
  params['gamma'] = float(gamma)
  params['alpha'] = float(alpha)
  params['vocab_size'] = vocab_size
  params['num_levels'] = 4
  params['branching_factor'] = 3
  params['logbeta'] = log(vocab_size * gamma_fn(alpha)) - \
      log_gamma_fn(vocab_size * alpha)
  params['newpathprob'] = - vocab_size * (alpha - 1) * log(vocab_size) - \
      params['logbeta']
  (num_docs, num_words) = np.shape(W)
  params['num_docs'] = num_docs
  params['num_words'] = num_words
  params['max_iter'] = 5
  params['iteration'] = -1
  params['temperature'] = 10.0
  params['decay_factor'] = 0.95
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

def nCRP_approx(counts, params):
  counts = map(lambda x:x + params['gamma'], counts)
  counts_sum = sum(counts)
  return map(lambda x:x/counts_sum, counts)

def sample_Zdn(W_d, Z_d, beta, counts, params, n):
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
  
  prior = nCRP_approx(counts[num_internal_nodes:], params)
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
      params['temperature']))
  #print Z_dn
  Z_dn = Z_dn.tolist().index(1)
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
  if 1.0 / temperature > 100:
    # numerical issues with direct calcuations
    new_values = map(lambda x: 1.0 if (x == max(values)) else 0.0, values)
  else:
    new_values = map(lambda x:x**(1.0 / temperature), values)
  norm = sum(new_values)
  new_values = map(lambda x:x/norm, new_values)
  #print values, new_values
  #raw_input()
  return new_values

def filter_Z(W, Z):
  try:
    idx = W.tolist().index(-1)
    return Z[:idx]
  except ValueError:
    return Z

def CRP_likelihood(freq, gamma):
  print 'In CRP ll', freq
  raw_input()
  log_prob = 0.0
  freq = filter(lambda x:x>0, freq)
  for f in freq:
    log_prob += log(factorial(freq[f] - 1))
    print freq, f, freq[f]
  log_prob += (len(freq) - 1) * log(gamma)
  for k in range(1, sum(freq)):
    log_prob -= log(k + gamma)
  return log_prob

def NCRP_likelihood(Z, params):
  branching_factor = params['branching_factor']
  num_levels = params['num_levels']
  num_nodes = (branching_factor ** num_levels - 1) / (branching_factor - 1)
  num_paths = branching_factor ** (num_levels - 1)
  num_internal_nodes = num_nodes - num_paths
  log_prob = 0.0
  counts = [0] * num_paths
  while len(counts) > 1:
    prev_level_counts = []
    while len(counts) > 0:
      curr_counts = counts[:branching_factor]
      counts = counts[branching_factor:]
      log_prob += CRP_likelihood(curr_counts, params['gamma'])
      prev_level_counts.append(sum(curr_counts))
    counts = prev_level_counts
  return log_prob

# computes log likelihood
def compute_likelihood(W, Z, beta, params):
  likelihood = 0.0
  alpha = params['alpha']
  for curr_beta in beta:
    for v in range(len(curr_beta)):
      likelihood += (alpha - 1) * log(curr_beta[v])
  for d in range(len(Z)):
    likelihood += NCRP_likelihood(filter_Z(W[d], Z[d]), params)
  for d in range(len(W)):
    for n in range(len(W[0])):
      if (W[d][n] == -1):
        continue
      likelihood += log(beta[Z[d][n]][int(W[d][n])])
  return likelihood

def sample_Z_d(W_d, Z_d, beta, params):
  # compute counts for this document
  counts = computeCounts(Z_d, params['num_levels'], \
      params['branching_factor'])
  #print Z[d], counts
  #raw_input()
  for n in range(len(Z_d)):
    Z_d[n] = sample_Zdn(W_d, Z_d, beta, counts, params, n)
  return Z_d

def sample_Z(W, Z, beta, params):
  Z_new = Parallel(n_jobs = 40)(delayed(sample_Z_d)(W[d], Z[d], beta, params) \
      for d in range(params['num_docs']))
  return Z_new

def sample_beta(num_paths, params, Z, W, beta):
  all_docs_counts = []
  for path in range(num_paths):
    all_docs_counts.append([params['alpha']] * params['vocab_size'])
  for d in range(len(Z)):
    for n in range(len(Z[0])):
      #print 'Zdn', Z[d][n]
      #print 'Wdn', W[d][n]
      all_docs_counts[Z[d][n]][W[d][n]] += 1
  beta_new = beta
  for path_idx in range(len(beta)):
    beta_new[path_idx] = dirichlet(all_docs_counts[path_idx])
  return beta_new

def run_gibbs_sampler(W, Z, beta, params):
  branching_factor = params['branching_factor']
  num_levels = params['num_levels']
  num_nodes = (branching_factor ** num_levels - 1) / (branching_factor - 1)
  num_paths = branching_factor ** (num_levels - 1)
  num_internal_nodes = num_nodes - num_paths
  best_Z = Z
  best_beta = beta
  best_likelihood = compute_likelihood(W, Z, beta, params)
  last_update_iter = 0
  for iteration in range(params['max_iter']):
    params['iteration'] += 1
    #params['temperature'] *= params['decay_factor']
    if iteration - last_update_iter > 10:
      params['temperature'] *= params['decay_factor']
      if params['temperature'] < 0.001:
        break
    print 'iter :', iteration
    # sample Z
    Z = sample_Z(W, Z, beta, params)
    print 'Z sampled'
    #for d in range(num_docs):
    #  Z[d] = sample_Z_d(W[d], Z[d], beta, params)
    #sample_all_Z()
    # sample beta
    beta = sample_beta(num_paths, params, Z, W, beta)
    print 'beta sampled'
    # sample alpha
    #eta = 1.0
    #for beta_p in beta:
    #  for beta_pv in beta_p:
    #    eta -= log(beta_pv)
    #print 'eta', eta
    #params['alpha'] = exponential(1.0/eta)
    #print 'alpha', params['alpha']
    curr_likelihood = compute_likelihood(W, Z, beta, params)
    if curr_likelihood > best_likelihood:
      best_Z = Z
      best_beta = beta
      best_likelihood = curr_likelihood
      last_update_iter = iteration
    print iteration, best_likelihood
  return best_Z, best_beta

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

  best_beta = beta
  best_Z = Z
  best_likelihood = compute_likelihood(W, Z, beta, params)
  for trial in range(1):
    new_Z, new_beta = run_gibbs_sampler(W, Z, beta, params)
    new_likelihood = compute_likelihood(W, new_Z, new_beta, params)
    if new_likelihood > best_likelihood:
      best_beta = new_beta
      best_Z = new_Z
      best_likelihood = new_likelihood

  '''
  # write output
  with open(sys.argv[5], 'w') as f:
    for d in range(len(best_Z)):
      for n in range(len(best_Z[0])):
        f.write(str(best_Z[d][n]) + ' ')
      f.write('\n')
  #print 'best_Z'
  #print 'best_beta'
  #for i, beta in enumerate(best_beta):
  #  print i, beta
  with open(sys.argv[6], 'w') as f:
    for i, beta in enumerate(best_beta):
      f.write(str(i) + '\t' + ' '.join(map(lambda x:str(x), beta)) + '\n')
  '''

if __name__=="__main__":
  np.set_printoptions(suppress=True, precision=4)
  main()
