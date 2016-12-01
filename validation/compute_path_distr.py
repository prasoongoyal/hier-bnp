import sys
import numpy as np

prefix = sys.argv[1]
num_levels = eval(sys.argv[2])
bf = eval(sys.argv[3])
mapping_file = sys.argv[4]
prefix_test = sys.argv[5]
num_test_videos = eval(sys.argv[6])


def plen(s):
  return len(s.split('-'))

path_queue = ['0']

#mappings = []
while len(path_queue) > 0:
  path = path_queue[0]
  path_queue = path_queue[1:]
  if plen(path) < num_levels:
    for b in range(bf):
      path_queue.append(path + '-' + str(b))
  else:
    break

mapping_vidid_to_class = {}
with open(mapping_file) as f:
  for line in f.readlines():
    (vidid, classid) = (line.strip()).split()
    mapping_vidid_to_class[eval(vidid)] = classid

classes = list(np.unique(mapping_vidid_to_class.values()))

mapping_path_class_counts = {}
for path in path_queue:
  class_counts = [0] * len(classes)
  a = np.loadtxt(prefix + path + '.txt')
  (vidids, counts) = np.unique(a[:, 0], return_counts=True)
  for (v,c) in zip(vidids, counts):
    vid_class_idx = classes.index(mapping_vidid_to_class[int(v)])
    '''
    try:
      class_count = class_counts[vid_class]
    except KeyError:
      class_count = 0
    '''
    class_counts[vid_class_idx] += c
  print path, class_counts
  mapping_path_class_counts[path] = class_counts

#print classes
#sys.exit()
#all_vid_dicts = []
#for v in range(num_test_videos):
#  all_vid_dicts.append({})
all_vid_counts = np.zeros((num_test_videos, len(classes)))

for path in path_queue:
  a = np.loadtxt(prefix_test + path + '.txt')
  (vidids, counts) = np.unique(a[:, 0], return_counts=True)
  for (v,c) in zip(vidids, counts):
    v = int(v)
    path_counts = mapping_path_class_counts[path]
    path_counts = map(lambda x:c*x, path_counts)
    #print np.shape(all_vid_counts[v, :])
    #print np.shape(np.asarray(path_counts))
    all_vid_counts[v, :] = all_vid_counts[v, :] + np.asarray(path_counts)
    #vid_class = mapping_vidid_to_class[v]
    #idx = classes.index(vid_class)
    #all_vid_counts[v][idx] += c

np.set_printoptions(threshold=np.nan)
#print all_vid_counts
for v in range(num_test_videos):
  print v, classes[np.argmax(all_vid_counts[v, :])]

