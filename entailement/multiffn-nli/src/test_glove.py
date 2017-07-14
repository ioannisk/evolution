counter = 0
with open("/home/ioannis/data/glove/glove.840B.300d.txt", "r") as file_ :
    for line in file_:
        counter +=1
        if counter%100000==0:
            print counter
        line = line.strip().split()
        if len(line) != 301:
            print line[0]
        # print len(line)
        # rcricn
