# Run as python levelwise_gibbs_sampling_wrapper.py corpus alpha sigma GAMMA 
# outfile_prefix num_levels branching_factor beta_outfile
# Assumes C++ code is already compiled.

import sys
import os

corpus = sys.argv[1]
alpha = sys.argv[2]
sigma = sys.argv[3]
gamma = sys.argv[4]
outfile_prefix = sys.argv[5]
num_levels = sys.argv[6]
bf = sys.argv[7]
beta_output_filename = sys.argv[8]

def plen(s):
  return len(s.split('-'))

#all_beta_file = open()

paths_queue = ['0']
os.system('ln -s %s %s' % (corpus, 'tmp_0.txt'))
while True:
  path = paths_queue[0]
  if plen(path) == num_levels:
    break
  paths_queue = paths_queue[1:]
  input_corpus = 'tmp_' + path + '.txt'
  output_prefix = 
  os.system('./a.out %s %s %s %s %s %s %s' % ('tmp_'+path+'.txt', alpha, sigma, gamma, \
      'tmp_'+path+'-'+str(b)+'.txt', 1, bf))
  
  # write beta to file
  os.system('cat %s >> %s' % ())

  '''
  os.system('./a.out %s %s %s %s tmp %s %s' % (corpus, alpha, sigma, gamma, 1, bf))
  # partition corpus
  corpus_files = []
  for b in range(eval(bf)):
    corpus_files.append(open('corpus_' + ))
  '''
