import sys
import numpy as np
from numpy.random import dirichlet, multinomial
from math import exp, log, factorial
from scipy.special import gamma as gamma_fn
from copy import copy, deepcopy

W = np.loadtxt(sys.argv[1])
gamma = eval(sys.argv[2])
alpha = eval(sys.argv[3])
vocab_size = eval(sys.argv[4])

debug = open('debug.txt', 'w')

num_levels = 4

(num_docs, num_words) = np.shape(W)
Z = []
for d in range(num_docs):
  Z.append(['0-0-0-0'] * num_words)
beta = {}
beta['0-0-0-0'] = dirichlet([alpha] * vocab_size)

def get0tol(path_str, level):
  nodes = path_str.split('-')
  return '-'.join(nodes[:level])

def path_len(path_str):
  return path_str.count('-')+1

def get_last_segment(path_str):
  return path_str.rsplit('-', 1)[1]

paths_evaluated = {}

def generate_paths(curr_path, count_keys):
  count_keys.sort()
  try:
    existing_children_keys = paths_evaluated[str((curr_path, count_keys))]
  except KeyError:
    existing_children_keys = count_keys[:]
    curr_level = len(curr_path)
    # add new path
    try:
      max_value = max(map(lambda x:eval(get_last_segment(x)), \
                    existing_children_keys))
    except:
      max_value = -1
    existing_children_keys.append(curr_path + '-' + str(max_value + 1))
    paths_evaluated[str((curr_path, count_keys))] = existing_children_keys
  return existing_children_keys

peak_exp = 3.0

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

def compute_probs(new_paths, curr_prob, counts, gamma):
  probs = []
  for path in new_paths:
    try:
      probs.append(counts[path])
    except:
      probs.append(gamma)
  f = curr_prob / sum(probs)
  probs = map(lambda x:f*x, probs)
  return zip(new_paths, probs)

def generate_next_level_paths_2(curr_path_prob, counts, gamma):
  curr_path_len_plus_1 = path_len(curr_path_prob[0]) + 1
  counts = {k: v for k, v in counts.iteritems() if k.startswith(curr_path_prob[0]) \
            and path_len(k)==curr_path_len_plus_1}
  new_paths = generate_paths(curr_path_prob[0], counts.keys())
  paths_probs = compute_probs(new_paths, curr_path_prob[1], counts, gamma)
  return paths_probs


def generate_next_level_paths_1(curr_path_prob, counts, gamma):
  curr_path = curr_path_prob[0]
  curr_prob = curr_path_prob[1]
  ck = counts.keys()
  ck.sort()
  #debug.write(curr_path + '\t' + str(ck) + '\n')
  existing_children_keys = filter(lambda x:x.startswith(curr_path) and \
                            path_len(x)==path_len(curr_path) + 1, counts.keys())
  existing_children_values = [counts[k] for k in existing_children_keys]
  curr_level = len(curr_path)
  #add new path
  try: 
    max_value = eval(max(map(lambda x:x[-1:], existing_children_keys)))
  except:
    max_value = -1
  existing_children_keys.append(curr_path + '-' + str(max_value + 1))
  existing_children_values.append(gamma)
  norm = curr_prob / sum(existing_children_values)
  existing_children_values = map(lambda x:x*norm, existing_children_values)
  return zip(existing_children_keys, existing_children_values)

def nCRP(counts, gamma):
  paths_prob_dict = {}
  paths_prob_list = [('0', 1.0)]
  while len(paths_prob_list) > 0:
    curr_path_prob = paths_prob_list[0]
    if (path_len(curr_path_prob[0]) < num_levels):
      gnlp2 = generate_next_level_paths_2(curr_path_prob, counts, gamma)
      paths_prob_list = gnlp2 + paths_prob_list[1:]
    else:
      paths_prob_dict[curr_path_prob[0]] = curr_path_prob[1]
      paths_prob_list = paths_prob_list[1:]
  return paths_prob_dict

def NCRP_prob(Z, gamma):
  total_log_prob = 0.0
  paths_list = ['0']
  while len(paths_list) > 0:
    curr_path = paths_list[0]
    paths_list = paths_list[1:]
    if (path_len(curr_path) == num_levels):
      continue
    Z_curr = filter(lambda x:x.startswith(curr_path), Z)
    Z_curr = map(lambda x:get0tol(x, path_len(curr_path)+1), Z_curr)
    freq = {}
    freq_keys = []
    for z in Z_curr:
      try:
        freq[z] += 1
      except KeyError:
        freq[z] = 1
        freq_keys.append(z)
    for f in freq:
      total_log_prob += log(factorial(freq[f] - 1))
    total_log_prob += (len(freq) - 1) * log(gamma)
    for k in range(1, len(Z_curr)):
      total_log_prob -= log(k + gamma)
    paths_list = freq_keys + paths_list
  return total_log_prob

logbeta = log(vocab_size * gamma_fn(alpha)) - log(gamma_fn(vocab_size * alpha))
newpathdefaultprob = - vocab_size * (alpha - 1) * log(vocab_size) - logbeta
print logbeta, newpathdefaultprob

def likelihood(alpha, gamma, beta, Z, W):
  l = 0.0
  for k in beta.keys():
    for v in range(len(beta[k])):
      l += (alpha - 1) * log(beta[k][v])
  l -= logbeta * len(beta)
  comp1 = l
  num_docs = len(Z)
  num_words = len(Z[0])
  for d in range(num_docs):
    l += NCRP_prob(Z[d], gamma)
  comp2 = l - comp1
  for d in range(num_docs):
    for n in range(num_words):
      l += log(beta[Z[d][n]][int(W[d][n])])
  comp3 = l - comp2 - comp1
  #print 'likelihood components:', comp1, comp2, comp3, comp1 + comp2 + comp3
  return comp1 + comp2 + comp3

def likelihood_with_extra_paths(likelihood, num_extra_paths):
  #return likelihood + num_extra_paths*newpathdefaultprob
  orig_likelihood = likelihood
  # generate num_extra_paths dirichlet variables and compute their likelihoods
  for p in range(num_extra_paths):
    dir_p = dirichlet([alpha] * vocab_size)
    likelihood += (alpha - 1) * sum(map(lambda x:log(x), dir_p)) - logbeta
  #print orig_likelihood, num_extra_paths, likelihood
  return likelihood

def EuclideanDist(d1, d2):
  d = 0
  for i in range(len(d1)):
    d += (d1[i] - d2[i])**2
  return d**(0.5)

TOL = 10**-1

def merge_similar_paths(Z, beta):
  beta_keys = beta.keys()
  beta_values = [beta[k] for k in beta_keys]
  beta_len = len(beta)
  beta_map_to = {}
  for i in range(beta_len):
    for j in range(i + 1, beta_len):
      if j in beta_map_to:
        pass
      else:
        if EuclideanDist(beta_values[i], beta_values[j]) < TOL:
          beta_map_to[j] = i
          for d in range(len(Z)):
            for n in range(len(Z[0])):
              if (Z[d][n] == j):
                Z[d][n] = i
  #print 'merging paths: ', len(beta),
  for k in beta_map_to:
    del beta[beta_keys[k]]
  #print len(beta)
  return Z, beta

def cosine_sim(d1, d2, c1, c2):
  # normalize c1 and c2
  sum1 = sum(c1)
  c1 = map(lambda x:x/sum1, c1)
  sum2 = sum(c2)
  c2 = map(lambda x:x/sum2, c2)
  score = 0.0
  for k in range(len(d1)):
    score += d1[k] * d2[k] * c1[k] * c2[k]
  return score

def weighted_diff(d1, d2, c1, c2):
  score = 0.0
  for k in range(len(d1)):
    score += (c1[k] + c2[k]) * abs(d1[k] - d2[k])
  return score

def merge_similar_paths_with_Z(Z, beta, W):
  #return Z, beta
  beta_keys = beta.keys()
  beta_values = [beta[k] for k in beta_keys]
  beta_len = len(beta)
  beta_map_to = {}
  beta_counts = []
  for key in range(len(beta_keys)):
    beta_counts.append([0] * vocab_size)
  for d in range(len(Z)):
    for n in range(len(Z[0])):
      beta_counts[beta_keys.index(Z[d][n])][int(W[d][n])] += 1
  for i in range(beta_len):
    for j in range(i + 1, beta_len):
      if (i in beta_map_to) or (j in beta_map_to):
        pass
      else:
        wt_diff = weighted_diff(beta_values[i], beta_values[j], beta_counts[i], \
            beta_counts[j])
        if wt_diff < TOL:
          print 'merging paths : ', beta_values[i], '\n', beta_values[j], '\n', \
            beta_counts[i], '\n', beta_counts[j], '\n', wt_diff, '\n', \
            beta_keys[i], '\n', beta_keys[j]
          #raw_input()
          beta_map_to[j] = i
          for d in range(len(Z)):
            for n in range(len(Z[0])):
              if (Z[d][n] == beta_keys[j]):
                Z[d][n] = beta_keys[i]
        else:
          pass
          #print 'not merging paths : ', beta_values[i], '\n', beta_values[j], '\n', \
          #  beta_counts[i], '\n', beta_counts[j], '\n', wt_diff
  #print 'merging -- ', len(beta),
  for k in beta_map_to:
    del beta[beta_keys[k]]
  #print len(beta)
  return Z, beta

max_iters = 1000
max_likelihood = -10**10
best_Z = None
best_beta = deepcopy(beta)
temperature = 10.0
for i in range(max_iters):
  print 'i', i 
  temperature *= 0.95
  print 'T', temperature
  # sample z_dn
  for d in range(num_docs):
    # compute counts for current document
    counts = {}
    for n in range(num_words):
      for level in range(num_levels):
        try:
          counts[get0tol(Z[d][n], level+1)] += 1
        except KeyError:
          counts[get0tol(Z[d][n], level+1)] = 1
    #print 'counts', counts
    for n in range(num_words):
      # remove Z_dn from counts
      for level in range(num_levels):
        counts[get0tol(Z[d][n], level+1)] -= 1
      prior = nCRP(counts, gamma)
      #print 'prior', prior
      posterior = {}
      for k in prior.keys():
        if k not in beta.keys():
          beta[k] = dirichlet([alpha] * vocab_size)
          beta[k] = np.asarray([1.0 / vocab_size] * vocab_size)
          #print beta[k], beta_tmp
          #print type(beta[k]), type(beta_tmp)

          #raw_input()
          #print 'new_path_lk:', k, beta[k], (alpha - 1) * sum(map(lambda x:log(x), \
          #          beta[k])) - logbeta
        posterior[k] = prior[k] * beta[k][W[d][n]]
      #print 'posterior', posterior
      if i < 0:
        for k in posterior:
          posterior[k] = exp(-sum(map(eval, k.split('-'))))
      keys = posterior.keys()
      values = [posterior[k] for k in keys]
      norm = sum(values) * 1.0
      values = map(lambda x:x/norm, values)
      #print 'posterior_norm', values
      #print 'old Z', d, n, Z[d][n]
      #print make_distr_peaked(values, temperature) 
      Z[d][n] = keys[multinomial(1, make_distr_peaked(values, temperature)).tolist().index(1)]
      #print 'new Z', d, n, Z[d][n]
      #if (d == 0 and n==10):
      #  raw_input()
      # add new value of Z_nd to counts
      for level in range(num_levels):
        try:
          counts[get0tol(Z[d][n], level+1)] += 1
        except KeyError:
          counts[get0tol(Z[d][n], level+1)] = 1
      #print 'new counts', counts
  # sample beta
  beta_counts = []
  for b in range(len(beta)):
    beta_counts.append([0] * vocab_size)
  key2idx_mapping = beta.keys()
  for dd in range(num_docs):
    for n in range(num_words):
      beta_counts[key2idx_mapping.index(Z[dd][n])][int(W[dd][n])] += 1
  # remove empty paths
  for k in range(len(beta_counts)):
    if sum(beta_counts[k]) == 0:
      #print 'deleting 1 path', key2idx_mapping[k], \
      #        (alpha - 1) * sum(map(lambda x:log(x), \
      #        beta[key2idx_mapping[k]])) - logbeta
      del beta[key2idx_mapping[k]]
    else:
      pass
      #print 'not removing path', key2idx_mapping[k], \
      #        (alpha - 1) * sum(map(lambda x:log(x), \
      #        beta[key2idx_mapping[k]]))
  for k in beta.keys():
    beta[k] = dirichlet(map(lambda x:x + alpha, beta_counts[key2idx_mapping.index(k)]))
  # compute likelihood of the current configuration and update MAP estimate
  curr_likelihood = likelihood(alpha, gamma, beta, Z, W)
  #print curr_likelihood, max_likelihood, likelihood_with_extra_paths(max_likelihood, len(beta) - len(best_beta))
  if (curr_likelihood > likelihood_with_extra_paths(max_likelihood, len(beta) - len(best_beta))):
    max_likelihood = curr_likelihood
    #best_Z, best_beta = merge_similar_paths(deepcopy(Z), deepcopy(beta))
    best_Z, best_beta = merge_similar_paths_with_Z(deepcopy(Z), deepcopy(beta), W)
    print 'new best'
  print curr_likelihood
  if (i%10 == 0):    
    print best_Z
    print best_beta
    print len(best_beta)
    #raw_input()
    Z, beta = merge_similar_paths_with_Z(deepcopy(Z), deepcopy(beta), W)

print best_Z
print best_beta.keys()
print len(best_beta)
print best_beta
print max_likelihood
