
with open("/home/ioannis/data/glove/glove.840B.300d.txt", "r") as file_ :
    for line in file_:
        line = line.strip().split()
        if len(line) != 301:
            print line[0]
        # print len(line)
        # rcricn
