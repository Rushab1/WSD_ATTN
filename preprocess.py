import os
import sys 
import re
import string
import argparse
from numpy import random

reload(sys)

try:
	from nltk.corpus import stopwords
	from nltk import word_tokenize
	from nltk import pos_tag
except:
	print("NLTK NOT FOUND")


def preprocess(checkword, 
               pos = "noun", 
               datafile = "~/Data/one-million-sense-tagged-instances-wn30", 
               per_split = 0.2, 
               window = 1, 
               windowSize = 5,
               add_start_end = 1,
               dominant_only = 1,
               dominant_per = 0.2, 
               remove_punctuation = 0,
               remove_stopwords = 0,
               POS_tags = 0,
               POS_RED = 0,
               senseval = False):

    sys.setdefaultencoding('utf-8')

    if not senseval:
        print("Using million-word-dataset")
        cmd = "./Preprocess_Files/view.sh " + datafile + " " + pos + " " + checkword 
        os.system(cmd)
    else:
        print("Using Senseval Dataset")
        cmd = "python senseval_preprocess.py " + checkword
        os.system(cmd)
        
    Sent = open("./Preprocess_Files/Temp/tmp", "r").read().split('\n')
    del Sent[len(Sent) - 1] # last string is empty
    Sens = open("./Preprocess_Files/Temp/tmpkey", "r").read().split()

    assert(len(Sent) == len(Sens))
    print("Total number of examples: " + str(len(Sent)))
   
    # Delete minor senses if flag positive
    if dominant_only == 1:
        SenSet = list(set(Sens))
        SenCnt = []
        for i in SenSet:
            SenCnt.append(Sens.count(i))
        
        maxCnt = max(SenCnt)
        SenRemove = []
        for i in range(0, len(SenSet)):
            if SenCnt[i] < dominant_per * maxCnt:
                print("IGNORING", SenSet[i], SenCnt[i])
                SenRemove.append(SenSet[i])

        tmpSt = []
        tmpSe = []
        for i in range(0, len(Sent)):
            if Sens[i] not in SenRemove:
                tmpSt.append(Sent[i])
                tmpSe.append(Sens[i])
        Sent = tmpSt
        Sens = tmpSe
                
    ##Use POS Tags if flag positive
    if POS_tags == 1:
        for i in range(0, len(Sent)):
            try:
                tmp = word_tokenize(Sent[i])
                tmp_tagged = pos_tag(tmp)
            except UnicodeDecodeError as u:
                Sent[i] = re.sub(r'[^\x00-\x7F]+',' ', Sent[i])
                tmp = word_tokenize(Sent[i])
                tmp_tagged = pos_tag(tmp)
            tmp = ""
            for j in range(0, len(tmp_tagged)):
                tmptag = tmp_tagged[j][1].lower()

                if POS_RED == 1:
                    if tmptag in ['$', '\'', '(', ')', ',', '.', '--', ':', ';']:
                        tmptag = 'PUNCTUATION'
                    elif tmptag in ['DT', 'EX']:
                        tmptag = 'DETERMINER'
                    elif tmptag in ['jj', 'jjr', 'jjs']:
                        tmptag = 'ADJ'
                    elif tmptag in ['nn', 'nnp', 'nnps', 'nns']:
                        tmptag = 'NOUN'
                    elif tmptag in ['prp', 'prp$']:
                        tmptag = 'PRONOUN'
                    elif tmptag in ['rb', 'rbr','rbs']:
                        tmptag = 'ADVERB'
                    elif tmptag in ['vb', 'vbd', 'vbg','vbn','vbp','vbz']:
                        tmptag = 'VERB'
                    elif tmptag in ['wdt', 'wp', 'wp$', 'wrb']:
                        tmptag = 'WH_WORD'

                tmp += " " + tmp_tagged[j][0] + "_" + tmptag
            Sent[i] = tmp

    ##Randomly shuffle sentences
    tmp = list(zip(Sent, Sens))
    random.shuffle(tmp)
    Sent[:], Sens[:] = zip(*tmp)

    ##Get set of senses
    lenst = len(Sent)
    SenSet = list(set(Sens))
    SenSet.sort()
    SenCnt = []

    ##Get count of each sense
    for i in SenSet:
        SenCnt.append(Sens.count(i))

    ##Most dominant sense
    maxCnt = max(SenCnt)

    Sent_div = []
    for i in range(0,len(SenSet)):
        Sent_div.append([])
        
    newline_regex = re.compile(r"\n[\n]*")
    exclude = set(string.punctuation)


    for i in range(0, lenst):
        Sent[i] = Sent[i].strip()

        ##remove stopwords if flag positive
        if remove_stopwords == 1:
            tmp = Sent[i].split()
            filtered_words = [word for word in tmp if word not in stopwords.words('english')]
            Sent[i] = ' '.join(filtered_words)

        ##remove punctuations if flag positive
        if remove_punctuation == 1:
            Sent[i] = ''.join(ch for ch in Sent[i] if ch not in exclude)

        ##extract window around the word if window positive
        if window == 1:
            left, _ ,right = Sent[i].lower().partition(checkword)
            n = windowSize
            left = left.split()[-n:]
            right = right.split()[:n]
            #print(right)
            #try:
            #    del right[0]
            #except:
            #    pass
           
            tmp = []
            if len(left) < windowSize and add_start_end == 1:
                for j in range(0,windowSize-len(left)):
                    tmp.append('START')
            tmp.extend(left)
            left = tmp 

            if add_start_end == 1:
                for j in range(len(right), windowSize):
                    right.append('END') 

            Sent[i] = ' '.join(left + [checkword] + right)

        ##Formatting the sentences
        Sent[i] = Sent[i].strip()
        Sent[i] = Sent[i].replace(" ", "\n")
        Sent[i] = newline_regex.sub("\n", Sent[i])
        if "\n\n" in Sent[i]:
            print(Sent[i])
        ind = SenSet.index(Sens[i])
        Sent_div[ind].append(Sent[i])
    
    ##Split Training and test sentences
    SentTest = []
    SensTest = []
    Sent = []
    Sens = []
    for i in range(0, len(SenSet)):
        numTotal = len(Sent_div[i])
        numTest = int( per_split * numTotal)
        numTrain = numTotal - numTest

        SentTest.extend(Sent_div[i][ 0: numTest ])
        for j in range(0, numTest ):
            SensTest.append(SenSet[i])

        Sent.extend(Sent_div[i][ numTest: numTotal ])
        for j in range(0, numTrain ):
            Sens.append(SenSet[i])
    
    ##Equalize the number of Senses
    lenst = len(Sent)
    for i in range(0, lenst):
        ind = SenSet.index(Sens[i])
        num = int(1.0*maxCnt/SenCnt[ind])
        for j in range(0, num):
            Sent.append(Sent[i])
            Sens.append(Sens[i])

    ##Write training set sentences
    tmp = list(zip(Sent, Sens))
    random.shuffle(tmp)
    Sent[:], Sens[:] = zip(*tmp)

    print("Number of training examples: " + str(len(Sent)))
    fileSent = open("BLSTM/text_words.csv", "w")
    fileSens = open("BLSTM/summary_words.csv", "w")
    tmp = '\n\n'.join(Sent)
    fileSent.write(tmp)
    tmp = '\n\n'.join(Sens)
    fileSens.write(tmp)

    try: 
        ##Write test set sentences
        tmp = list(zip(SentTest, SensTest))
        random.shuffle(tmp)
        SentTest[:], SensTest[:] = zip(*tmp)
        print("Number of testing examples: " + str(len(SentTest)))

        fileSentTest = open("BLSTM/test_text_words.csv", "w")
        fileSensTest = open("BLSTM/test_summary_words.csv", "w")
        tmp = '\n\n'.join(SentTest)
        fileSentTest.write(tmp)
        tmp = '\n\n'.join(SensTest)
        fileSensTest.write(tmp)
    except:
        print("Number of testing examples: 0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Preprocess data")
    parser.add_argument('word', type = str)
    parser.add_argument('type', type = str)
    parser.add_argument('-datafile', type = str)
    parser.add_argument('-split', type = float)
    parser.add_argument('-window', type = int)
    parser.add_argument('-windowSize', type = int)
    parser.add_argument('-add_start_end', type = int)
    parser.add_argument('-dominant', type = float)
    parser.add_argument('-remove_punctuation', type = int)
    parser.add_argument('-remove_stopwords', type = int)
    parser.add_argument('-POS', type = int)
    parser.add_argument('-POSRED', type = int)
    parser.add_argument('-senseval', type = int)
    args = parser.parse_args()
    
    tmp_dominant_only = 1
    if args.word == None or args.type == None:
        print(args.help)

    if args.datafile == None:
        args.datafile = "~/Data/one-million-sense-tagged-instances-wn30"
    if args.split == None:
        args.split = 0.4
    if args.window == None:
        args.window = 1
    if args.windowSize == None:
        args.windowSize = 5
    if args.add_start_end == None:
        args.add_start_end = 1 
    if args.dominant == None:
        tmp_dominant_only = 0
        args.dominant = 0.2
    if args.remove_punctuation == None:
        args.remove_punctuation = 0
    if args.remove_stopwords == None:
        args.remove_stopwords = 0
    if args.POS == None:
        args.POS = 0
    if args.POSRED == None:
        args.POSRED = 0
    if args.senseval == None:
        args.senseval = False

    print(args.dominant, args.split)
    
    preprocess(args.word, args.type, 
            per_split = args.split, 
            window = args.window, 
            windowSize = args.windowSize,
            add_start_end = args.add_start_end,
            dominant_only = tmp_dominant_only, 
            dominant_per = args.dominant, 
            remove_punctuation = args.remove_punctuation, 
            remove_stopwords = args.remove_stopwords, 
            POS_tags = args.POS, 
            POS_RED = args.POSRED, 
            senseval = args.senseval )
