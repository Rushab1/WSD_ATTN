import itertools
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle

windowSize = 5
windowSize = windowSize * 2 + 1
wrong_sentences = pickle.load(open("wrong_sentences.pkl", "rb"))

def statistics(sent_file = "text_words.csv", sense_file = "summary_words.csv", file_flag = 1):
    if file_flag == 1:
        f = open(sent_file).read().split("\n\n")
        g = open(sense_file).read().split("\n\n")
    else:
        f = sent_file
        g = sense_file


    fdiv = {}
    fdiv_dist = {}
    for i in range(0, len(f)):
        #If directly providing input sentences
        if(file_flag == 0):
            f[i] = f[i].split(" ")
        else:
            f[i] = f[i].split("\n")

        try:
            fdiv[g[i]].append(f[i])
        except:
            fdiv[g[i]] = [f[i]]

        tmp = deepcopy(f[i])
        for j in range(0, len(tmp)):
            tmp[j] = str(j - (windowSize - 1) / 2) + "_" + tmp[j]

        try:
            fdiv_dist[g[i]].append(tmp)
        except:
            fdiv_dist[g[i]] = [tmp]


    fcount_dist = {}
    fcount = {}

    for i in set(g):
        flist = list(itertools.chain.from_iterable(fdiv[i]))
        words_to_count = (word for word in flist)
        c = Counter(words_to_count)
        fcount[i] = c

    for i in set(g):
        flist = list(itertools.chain.from_iterable(fdiv_dist[i]))
        words_to_count = (word for word in flist)
        c = Counter(words_to_count)
        fcount_dist[i] = c

    fdiv_by_position = {}
    for i in set(g):
        fdiv[i] = np.matrix(fdiv[i])

    for i in set(g):
        tmp = {}
        for j in range(0, windowSize):

            #words_to_count = (word for word in np.transpose(fdiv[i][:,j]).tolist()[0][0])
            words_to_count = (word for word in np.transpose(fdiv[i][:,j]).tolist()[0])
            c = Counter(words_to_count)
            tmp[j- (windowSize - 1) / 2 ] = c

        fdiv_by_position[i] = deepcopy(tmp)
    return fcount, fcount_dist, fdiv_by_position


def wrong_statistics(target, predicted):
    tmp = deepcopy(wrong_sentences[target][predicted])
    d = statistics(tmp, [predicted]*len(tmp), file_flag = 0)
    return d[2][predicted]


def save_statistics(save_file_name = "cooccurrence.csv", sent_file = "text_words.csv", sense_file = "summary_words.csv", file_flag = 1):
    s = statistics(sent_file, sense_file, file_flag)[2]

    save_file = open(save_file_name, "w")

    for sense in s:
        for position in s[sense].keys():
            for word in s[sense][position]:
                save_file.write(sense + " " + str(position) + " " + word + " " + str(s[sense][position][word]) + "\n")

    save_file.close()

save_statistics()
