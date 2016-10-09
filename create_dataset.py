import sys
from numpy.random import poisson, multinomial, dirichlet

# takes all children of current node, and either selects one of 
# the existing children or creates a new child
def CRP(gamma, existing_children_keys, existing_children_values, \
        curr_level):
  if curr_level == 0:
    return 0
  existing_children_values.append(gamma)
  normalization_const = sum(existing_children_values)*1.0
  next_value = multinomial(1, map(lambda x:x/normalization_const, \
                                  existing_children_values))
  idx = next_value.tolist().index(1)
  try:
    return existing_children_keys[idx][curr_level]
  except IndexError:
    # find max value at position idx
    max_value = -1
    for k in existing_children_keys:
      if int(k[curr_level]) > max_value:
        max_value = int(k[curr_level])
    return str(max_value + 1)

output_file = open(sys.argv[1], 'w')
word_paths_file = open(sys.argv[2], 'w')
beta_file = open(sys.argv[3], 'w')
num_doc = eval(sys.argv[4])
vocab_size = eval(sys.argv[5])
gamma = eval(sys.argv[6])
alpha = eval(sys.argv[7])

num_words_lambda = 20
num_levels = 4

beta = {}

for doc_id in range(num_doc):
  counts = {}
  num_words = num_words_lambda
  document = ''
  word_paths = ''
  for word_id in range(num_words):
    # create current word path
    path = ''
    while (len(path) < num_levels):
      existing_children_keys = filter(lambda x:x.startswith(path) and \
                                      len(x)==len(path)+1, \
                                      counts.keys())
      existing_children_values = [counts[k] for k \
                                 in existing_children_keys]
      next_level = CRP(gamma, existing_children_keys, \
                       existing_children_values, len(path))
      path = path + str(next_level)
      try:
        counts[path] += 1
      except KeyError:
        counts[path] = 1
    if path not in beta.keys():
      # sample parameters for this path
      beta[path] = dirichlet([alpha]*vocab_size)
    word = multinomial(1, beta[path]).tolist().index(1)
    if len(document) == 0:
      document = str(word)
      word_paths = path
    else:
      document += ' ' + str(word)
      word_paths += ' ' + path
  output_file.write(document + '\n')
  word_paths_file.write(word_paths + '\n')

output_file.close()
word_paths_file.close()

for k in beta.keys():
  beta_file.write(k + '\t' + str(beta[k].tolist()) + '\n')
beta_file.close()
