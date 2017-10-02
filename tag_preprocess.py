import os
import sys 
import re
import string
import argparse
from numpy import random
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import pos_tag
from copy import deepcopy

def preprocess(checkword, pos = "noun", per_split = 0.2, uniformly_random = 1, sliceSize = 5 , dominant_only = 1, dominant_per = 0.2, remove_punctuation = 0, remove_stopwords = 0, POS_tags = 0, POS_RED = 1):
    cmd = "./Preprocess_Files/view.sh " + pos + " " + checkword 
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
                print(SenSet[i], SenCnt[i])
                SenRemove.append(SenSet[i])

        tmpSt = []
        tmpSe = []
        for i in range(0, len(Sent)):
            if Sens[i] not in SenRemove:
                tmpSt.append(Sent[i])
                tmpSe.append(Sens[i])
        Sent = tmpSt
        Sens = tmpSe
    
    if POS_tags == 1:
        TagSent = []
    else:
        TagSent = deepcopy(Sent)

    if POS_tags == 1:
        for i in range(0, len(Sent)):
            print(i)
            try:
                tmp = word_tokenize(Sent[i])
                tmp_tagged = pos_tag(tmp)
            except UnicodeDecodeError as u:
                Sent[i] = re.sub(r'[^\x00-\x7F]+',' ', Sent[i])
                tmp = word_tokenize(Sent[i])
                tmp_tagged = pos_tag(tmp)
            tmp = ""
            for j in range(0, len(tmp_tagged)):
                if checkword not in tmp_tagged[j][0]:
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

                    tmp += " " + tmptag
                else:
                    tmp += " " + checkword
            # TagSent[i] = tmp
            TagSent.append(tmp)
            # Sent[i] = tmp

    #randomly shuffle
    tmp = list(zip(Sent, Sens, TagSent))
    random.shuffle(tmp)
    Sent[:], Sens[:], TagSent[:] = zip(*tmp)

    lenst = len(Sent)
    SenSet = list(set(Sens))
    SenSet.sort()
    SenCnt = []

    for i in SenSet:
        SenCnt.append(Sens.count(i))

    maxCnt = max(SenCnt)
    
    TagSent_div = []
    Sent_div = []
    for i in range(0,len(SenSet)):
        Sent_div.append([])
        TagSent_div.append([])
        
    newline_regex = re.compile(r"\n[\n]*")
    exclude = set(string.punctuation)
    for i in range(0, lenst):
        Sent[i] = Sent[i].strip()
        if remove_stopwords == 1:
            tmp = Sent[i].split()
            filtered_words = [word for word in tmp if word not in stopwords.words('english')]
            Sent[i] = ' '.join(filtered_words)

        if remove_punctuation == 1:
            Sent[i] = ''.join(ch for ch in Sent[i] if ch not in exclude)
        if uniformly_random == 1:
            left, _ ,right = Sent[i].lower().partition(checkword)
            n = sliceSize
            left = left.split()[-n:]
            right = right.split()[:n]
           
            tmp = []
            if len(left) < sliceSize:
                for j in range(0,sliceSize-len(left)):
                    tmp.append('START')
            tmp.extend(left)
            left = tmp 

            for j in range(len(right), sliceSize):
                right.append('END') 

            Sent[i] = ' '.join(left + [checkword] + right)

            left, _ ,right = TagSent[i].lower().partition(checkword)
            n = sliceSize
            left = left.split()[-n:]
            right = right.split()[:n]
           
            tmp = []
            if len(left) < sliceSize:
                for j in range(0,sliceSize-len(left)):
                    tmp.append('START')
            tmp.extend(left)
            left = tmp 

            for j in range(len(right), sliceSize):
                right.append('END') 

            TagSent[i] = ' '.join(left + [checkword] + right)

        
        Sent[i] = Sent[i].strip()
        Sent[i] = Sent[i].replace(" ", "\n")
        Sent[i] = newline_regex.sub("\n", Sent[i])
        TagSent[i] = TagSent[i].strip()
        TagSent[i] = TagSent[i].replace(" ", "\n")
        TagSent[i] = newline_regex.sub("\n", TagSent[i])
        if "\n\n" in Sent[i]:
            print(Sent[i])
        ind = SenSet.index(Sens[i])
        Sent_div[ind].append(Sent[i])
        TagSent_div[ind].append(TagSent[i])


    SentTest = []
    TagSentTest = []
    SensTest = []
    Sent = []
    TagSent = []
    Sens = []
    for i in range(0, len(SenSet)):
        numTotal = len(Sent_div[i])
        numTest = int( per_split * numTotal)
        numTrain = numTotal - numTest

        SentTest.extend(Sent_div[i][ 0: numTest - 1 ])
        TagSentTest.extend(TagSent_div[i][ 0: numTest - 1 ])
        for j in range(0, numTest - 1):
            SensTest.append(SenSet[i])

        Sent.extend(Sent_div[i][ numTest: numTotal ])
        TagSent.extend(TagSent_div[i][ numTest: numTotal ])
        for j in range(0, numTrain ):
            Sens.append(SenSet[i])
    
    lenst = len(Sent)
    for i in range(0, lenst):
        ind = SenSet.index(Sens[i])
        num = maxCnt/SenCnt[ind]
        for j in range(0, num):
            Sent.append(Sent[i])
            TagSent.append(TagSent[i])
            Sens.append(Sens[i])

    #Write training set sentences
    tmp = list(zip(Sent, Sens, TagSent))
    random.shuffle(tmp)
    Sent[:], Sens[:], TagSent[:] = zip(*tmp)

    print("Number of training examples: " + str(len(Sent)))
    fileSent = open("TAG_BLSTM/text_words.csv", "w")
    fileSens = open("TAG_BLSTM/summary_words.csv", "w")
    fileTagSent = open("TAG_BLSTM/text_words_tags.csv", "w")
    tmp = '\n\n'.join(Sent)
    fileSent.write(tmp)
    tmp = '\n\n'.join(TagSent)
    fileTagSent.write(tmp)
    tmp = '\n\n'.join(Sens)
    fileSens.write(tmp)

    #Write test set sentences
    try:
        tmp = list(zip(SentTest, SensTest, TagSentTest))
        random.shuffle(tmp)
        SentTest[:], SensTest[:], TagSentTest[:] = zip(*tmp)
        print("Number of testing examples: " + str(len(SentTest)))
    except:
        print("Number of testing examples: 0")

    fileSentTest = open("TAG_BLSTM/test_text_words.csv", "w")
    fileSensTest = open("TAG_BLSTM/test_summary_words.csv", "w")
    fileTagSentTest = open("TAG_BLSTM/test_text_words_tags.csv", "w")
    tmp = '\n\n'.join(SentTest)
    fileSentTest.write(tmp)
    tmp = '\n\n'.join(TagSentTest)
    fileTagSentTest.write(tmp)
    tmp = '\n\n'.join(SensTest)
    fileSensTest.write(tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Preprocess data")
    parser.add_argument('word', type = str)
    parser.add_argument('type', type = str)
    parser.add_argument('-split', type = float)
    parser.add_argument('-uniform', type = int)
    parser.add_argument('-dominant', type = float)
    parser.add_argument('-remove_punctuation', type = int)
    parser.add_argument('-remove_stopwords', type = int)
    parser.add_argument('-POS', type = int)
    parser.add_argument('-POSRED', type = int)
    args = parser.parse_args()
    
    if args.word == None or args.type == None:
        print(args.help)




    elif args.split == None and args.uniform == None and args.dominant == None and args.remove_punctuation == None and args.remove_stopwords == None and args.POS == None:
        preprocess(args.word, args.type)



    elif args.uniform == None and args.dominant == None and args.remove_punctuation == None and args.remove_stopwords == None and args.POS == None:
        preprocess(args.word, args.type, args.split)




    elif args.uniform == None and args.dominant == None and args.remove_punctuation == None and args.remove_stopwords == None and args.POS == None:
        preprocess(args.word, args.type, args.split)




    #currently in use
    elif args.split == None and args.uniform == None and args.dominant == None and args.remove_punctuation == None and args.remove_stopwords == None and args.POS != None:
        preprocess(args.word, args.type, POS_tags = args.POS)





    #currently in use__2
    elif args.split == None and args.uniform == None and args.dominant == None and args.remove_punctuation == None and args.remove_stopwords == None and args.POS != None and args.POSRED != None:
        preprocess(args.word, args.type, POS_tags = args.POS, POS_RED = args.POS_RED)
    


    #currently in use__3
    elif args.split != None and args.uniform == None and args.dominant == None and args.remove_punctuation == None and args.remove_stopwords == None and args.POS != None and args.POSRED != None:
        print("CURRENT 3")
        preprocess(args.word, args.type, per_split = float(args.split), POS_tags = args.POS, POS_RED = args.POSRED)




    elif args.split != None and args.uniform != None and args.dominant != None and args.remove_punctuation != None and args.remove_stopwords != None and args.POS != None:
        preprocess(args.word, args.type, args.split, 1, args.uniform, 1, args.dominant, args.remove_punctuation, args.remove_stopwords, args.POS)
