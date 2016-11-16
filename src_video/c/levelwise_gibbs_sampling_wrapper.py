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
num_levels = eval(sys.argv[6])
bf = sys.argv[7]
beta_output_filename = sys.argv[8]

def plen(s):
  return len(s.split('-'))

#all_beta_file = open()

paths_queue = ['0']
os.system('ln -s %s %s' % (corpus, 'tmp_0.txt'))
while len(paths_queue)>0:
  path = paths_queue[0]
  if plen(path) == num_levels:
    break
  #print path
  paths_queue = paths_queue[1:]
  curr_input_corpus = 'tmp_' + path + '.txt'
  curr_output_prefix = 'tmp_' + path
  cmd = ('./single_level.o %s %s %s %s %s %s' % (curr_input_corpus, alpha, sigma, gamma, \
      curr_output_prefix, bf))
  #print cmd
  #raw_input()
  os.system(cmd)
  
  # write beta to file
  os.system('cat %s >> %s' % (curr_output_prefix + '_beta.out', beta_output_filename))

  # partition corpus
  corpus_files = []
  for b in range(eval(bf)):
    corpus_files.append(open('tmp_' + path + '-' + str(b) + '.txt', 'w'))
  Z_file = open(curr_output_prefix + '_Z.out')
  W_file = open(curr_input_corpus)
  Z_list = []
  map_vidframe2path = {}
  for line in Z_file.readlines():
    line = line.strip()
    Z_list += map(eval, line.split())
  for line in W_file.readlines():
    line = line.strip()
    parts = line.split()
    vidid, frameid = parts[0], parts[1]
    map_vidframe2path[(vidid, frameid)] = Z_list[0]
    Z_list = Z_list[1:]
    
  corpus_parent = open(curr_input_corpus)
  for line in corpus_parent.readlines():
    line = line.strip()
    parts = line.split()
    vidid, frameid = parts[0], parts[1]
    branch = map_vidframe2path[(vidid,frameid)]
    corpus_files[branch].write(line + '\n')    
  for b in range(eval(bf)):
    corpus_files[b].close()
  if len(path) < num_levels:
    for b in range(eval(bf)):
      paths_queue.append(path + '-' + str(b))

  '''
  os.system('./a.out %s %s %s %s tmp %s %s' % (corpus, alpha, sigma, gamma, 1, bf))
  # partition corpus
  corpus_files = []
  for b in range(eval(bf)):
    corpus_files.append(open('corpus_' + ))
  '''
