

with file_ as open("/home/ioannis/data/glove/glove.840B.300d.txt", "r"):
    for line in file_:
        line = line.strip().split()
        print len(line)
        rcricn
