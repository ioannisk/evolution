import numpy as np

def load_text_embeddings(path):
    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :return: a tuple (wordlist, array)
    """
    words = []

    # start from index 1 and reserve 0 for unknown
    vectors = []
    with open(path, 'r') as f:
        for line in f:
            # line = line.decode('utf-8')
            line = line.strip()
            if line == '':
                continue

            fields = line.split(' ')
            word = fields[0]
            words.append(word)
            vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            print(vector.shape)
            print(np.linalg.norm(vector))
            advfvfv
            vectors.append(vector)
    # IPython.embed()
    embeddings = np.array(vectors, dtype=np.float32)

    ## Save embeddings
    vocab_file = open("/home/ioannis/data/glove/glove-840B-vocabulary_l2.txt", "w")
    for i in words:
        vocab_file.write("{}\n".format(i))
    npy_file = open("/home/ioannis/data/glove/glove-840B_l2.npy", "w")
    np.save(npy_file, embeddings)
    # return words, embeddings

if __name__=="__main__":
    load_text_embeddings("/home/ioannis/data/glove/glove.840B.300d.txt")
