import sys

infile = sys.argv[1]
outfile = open(sys.argv[2], 'w')

for line in open(infile).readlines():
  line = line.strip()
  words = line.split()
  counts = {}
  for w in words:
    try:
      counts[w] += 1
    except KeyError:
      counts[w] = 1
  outfile.write(str(len(counts)))
  for k in counts:
    outfile.write(' ' + str(k)+':'+str(counts[k]))
  outfile.write('\n')
