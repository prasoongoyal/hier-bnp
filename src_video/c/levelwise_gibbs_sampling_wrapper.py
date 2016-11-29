# Run as python levelwise_gibbs_sampling_wrapper.py corpus alpha sigma GAMMA 
# outfile_prefix num_levels branching_factor beta_outfile
# Assumes C++ code is already compiled.

import sys
import os
import subprocess

num_features = 2048

corpus = sys.argv[1]
alpha = sys.argv[2]
sigma = sys.argv[3]
gamma = sys.argv[4]
tmp_prefix = sys.argv[5]
num_levels = eval(sys.argv[6])
bf = sys.argv[7]
beta_output_filename = sys.argv[8]

def plen(s):
  return len(s.split('-'))

#all_beta_file = open()

paths_queue = ['0']
os.system('ln -s %s %s' % (corpus, tmp_prefix + '_0.txt'))
while len(paths_queue)>0:
  path = paths_queue[0]
  #if plen(path) == num_levels:
    #break
  #print path
  paths_queue = paths_queue[1:]
  curr_input_corpus = tmp_prefix + '_' + path + '.txt'
  curr_output_prefix = tmp_prefix + '_' + path

  os.system('echo %s >> %s' % (path, beta_output_filename))

  cmd = 'wc -l ' + curr_input_corpus
  try:
    num_frames_in_corpus = eval(subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0].split()[0])
  except IndexError:
    num_frames_in_corpus = 0
  if (num_frames_in_corpus == 0):
    default_beta = ('-1 ' * num_features)
    os.system('echo %s >> %s' % (default_beta, beta_output_filename))
    if plen(path) < num_levels-1:
      for b in range(eval(bf)):
        paths_queue.append(path + '-' + str(b))
    continue

  cmd = ('./single_level_switch.o %s %s %s %s %s %s %s' % (curr_input_corpus, alpha, sigma, gamma, \
      curr_output_prefix, bf, ('1' if plen(path)==num_levels-1 else '0')))
  #print cmd
  #raw_input()
  os.system(cmd)
  
  # write beta to file
  os.system('cat %s >> %s' % (curr_output_prefix + '_beta.out', beta_output_filename))

  # partition corpus
  corpus_files = []
  for b in range(eval(bf)):
    corpus_files.append(open(tmp_prefix + '_' + path + '-' + str(b) + '.txt', 'w'))
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
  # delete old files
  try:
    os.system('rm -f %s' % curr_input_corpus)
  except:
    pass
  try:
    os.system('rm -f %s' % (curr_output_prefix + '_Z_out'))
  except:
    pass
  try:
    os.system('rm -f %s' % (curr_output_prefix + '_beta_out'))
  except:
    pass

  if plen(path) < num_levels-1:
    for b in range(eval(bf)):
      paths_queue.append(path + '-' + str(b))

  '''
  os.system('./a.out %s %s %s %s tmp %s %s' % (corpus, alpha, sigma, gamma, 1, bf))
  # partition corpus
  corpus_files = []
  for b in range(eval(bf)):
    corpus_files.append(open('corpus_' + ))
  '''
