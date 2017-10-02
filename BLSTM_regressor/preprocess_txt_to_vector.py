import os
import random
import sys
import re
import numpy as np
import string
reload(sys)
uniform = 0

def preprocess(checkword = 'all',per_split = 0.4, checkWindow = 1, sliceSize = 5):
    sys.setdefaultencoding('utf-8')

    txt=open('text_words.csv',"w")
    smy=open('summary_words.csv',"w")

    os.system("echo \"\" > text_words.csv && echo \"\" >  summary_words.csv")

    f = open('../Sentences.txt','r')
    Sentences = f.read()
    f.close()

    f = open('../Senses.txt','r')
    Senses = f.read()
    f.close()
    
#######################################################################3

    Sentences_Split = Sentences.split('\n')
    Senses_Split = Senses.split('\n')
    
    tmp = list(zip(Sentences_Split,Senses_Split))
    random.shuffle(tmp)
    
    Sentences_Split[:], Senses_Split[:] = zip(*tmp)

#######################################################################3
    _tmpst = []
    _tmpse = []

    if checkword != 'all':
        for i in range(0,len(Sentences_Split)):
            if checkword == 'line' and Senses_Split[i].lower().strip() in ['division','formation','cord','text','product','phone']:
            # if checkword in Sentences_Split[i].lower():
                _tmpst.append(Sentences_Split[i])
                _tmpse.append(Senses_Split[i])
            elif checkword == 'interest' and Senses_Split[i].lower().strip() in ['interest1','interest2','interest3','interest4','interest5','interest6']:
                _tmpst.append(Sentences_Split[i])
                _tmpse.append(Senses_Split[i])
            elif checkword == 'hard' and Senses_Split[i].lower().strip() in ['hard1','hard2','hard3']:
                _tmpst.append(Sentences_Split[i])
                _tmpse.append(Senses_Split[i])
            elif checkword == 'hardlower' and Senses_Split[i].lower().strip() in ['hard2','hard3']:
                _tmpst.append(Sentences_Split[i])
                _tmpse.append(Senses_Split[i])
            elif checkword == 'linelower' and Senses_Split[i].lower().strip() in ['text', 'phone', 'formation', 'cord', 'division']:
                _tmpst.append(Sentences_Split[i])
                _tmpse.append(Senses_Split[i])
            elif checkword == 'serve' and Senses_Split[i].lower().strip() in ['serve2', 'serve6', 'serve10', 'serve12']:
                _tmpst.append(Sentences_Split[i])
                _tmpse.append(Senses_Split[i])



    Sentences_Split = _tmpst
    Senses_Split = _tmpse 

    #Extract context window around required word if checkWindow = 1
    if checkWindow == 1:
        for i in range(0, len(Sentences_Split)):
            # replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
            # Sentences_Split[i] = Sentences_Split[i].translate(replace_punctuation)
            left, _ , right = Sentences_Split[i].lower().partition(checkword)

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

            s = ' '.join(left + [checkword] + right)
            Sentences_Split[i] = s


    lenst = len(Sentences_Split)
    
#Equally distribute test and training data among all senses, 40%-60%  for hard1 and hard2 and hard3, instead of global division, done for uniformity
    if uniform == 1:
        print("Original # of Sentence",len(Sentences_Split))
        p = list(set(Senses_Split))
        p.sort()
        len_p = len(p)

        divSt = []
        divSe = []
        for i in range(0,len_p):
            divSt.append([])
            divSe.append([])

        lenst_perm = np.random.permutation(lenst)
        for i in lenst_perm:
            j = p.index(Senses_Split[i])    
            print(j, Senses_Split[i])
            divSt[j].append(Sentences_Split[i])
            divSe[j].append(Senses_Split[i])

        Sentences_Split = []
        Sentences_Split_test = []
        Senses_Split = []
        Senses_Split_test = []
    
        for i in range(0, len_p):
            tmp = divSt[i]
            Sentences_Split_test.extend(tmp[0:int(per_split*len(tmp))])
            Sentences_Split.extend(tmp[int(per_split*len(tmp[i])):len(tmp)])

            tmp = divSe[i]
            Senses_Split_test.extend(tmp[0:int(per_split*len(tmp))])

            Senses_Split.extend(tmp[int(per_split*len(tmp[i])): len(tmp)])

        tmp = list(zip(Sentences_Split_test,Senses_Split_test))
        random.shuffle(tmp)
        Sentences_Split_test[:], Senses_Split_test[:] = zip(*tmp)

        tmp = list(zip(Sentences_Split,Senses_Split))
        random.shuffle(tmp)
        Sentences_Split[:], Senses_Split[:] = zip(*tmp)
    else:
        Sentences_Split_test = Sentences_Split[0:int(per_split*lenst)]
        Senses_Split_test = Senses_Split[0:int(per_split*lenst)]

        Sentences_Split  = Sentences_Split[int(per_split*lenst):lenst]
        Senses_Split = Senses_Split[int(per_split*lenst):lenst]

#######################################################################

    for i in range(0,lenst):
        if Senses_Split[i].lower().strip() in ['phone','cord','formation','division','text']:
            for j in range(0,10):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() in ['hard3']:
            for j in range(0,7):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() in ['hard2']:
            for j in range(0,4):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() in ['hard1']:
            for j in range(0,1):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() in ['interest3','interest4']:
            for j in range(0,5):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() in ['interest1','interest5']:
            for j in range(0,2):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'interest2':
            for j in range(0,30):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'interest6':
            for j in range(0,2):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'product':
            for j in range(0,4):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'serve2':
            for j in range(0,3):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'serve6':
            for j in range(0,3):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'serve10':
            for j in range(0,1):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'serve12':
            for j in range(0,2):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])

    Sentences_Split = Sentences_Split[0:len(Sentences_Split) - 1] 
    Senses_Split = Senses_Split[0:len(Senses_Split) - 1]

#######################################################################3

    tmp = list(zip(Sentences_Split,Senses_Split))
    random.shuffle(tmp)
    
    Sentences_Split[:], Senses_Split[:] = zip(*tmp)
    

    for i in range(0,len(Sentences_Split)):
        Senses_Split[i] = Senses_Split[i].replace("\r","")
        if Senses_Split[i] == "":
            Senses_Split[i] = "some_unknown_symbol"
            # print("some unknown symbol replaced")

    for i in range(0,len(Sentences_Split_test)):
        Senses_Split_test[i] = Senses_Split_test[i].replace("\r","")
        if Senses_Split_test[i] == "":
            Senses_Split_test[i] = "some_unknown_symbol"
            # print("some unknown symbol replaced")
    

    #Changes and writing TRAINING data
    for i in range(0,len(Sentences_Split )):
        Sentences_Split[i] = re.sub(' [ ]*','&&&',Sentences_Split[i])



    # Sentences_Split = Sentences_Split[1:50]
    # Senses_Split = Senses_Split[1:50]

    Sentences = '@@@'.join(Sentences_Split)
    Senses = '@@@'.join(Senses_Split)

    Sentences = re.sub('@@@[@@@]*','\n\n',Sentences)
    Sentences = re.sub('&&[&&]*','\n',Sentences)
    Sentences = re.sub('\n\n\n[\n]*','\n\n',Sentences)
    
    if Sentences[0] in [' ', '\n']:
        Sentences = Sentences[1:len(Sentences)]

    print(len(Sentences_Split), len(Senses_Split),per_split)
    Senses = re.sub('@@@[@@@]*','\n\n',Senses)
    txt.write(Sentences)
    smy.write(Senses)
    txt.close()
    smy.close()
    
    Sentences2 = []
    Senses2 = []
    for i in range(0,len(Sentences_Split)):
        if Senses_Split[i] in ['HARD2', 'HARD3', 'cord', 'formation', 'phone', 'text', 'division', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'SERVE2', 'SERVE6']:
            Senses2.append(Senses_Split[i])
            Sentences2.append(Sentences_Split[i])

    print(len(Sentences2))
    Sentences = '@@@'.join(Sentences2)
    Senses = '@@@'.join(Senses2)

    Sentences = re.sub('@@@[@@@]*','\n\n',Sentences)
    Sentences = re.sub('&&[&&]*','\n',Sentences)
    Sentences = re.sub('\n\n\n[\n]*','\n\n',Sentences)
    # print(Sentences[0])
    if Sentences[0] in [' ', '\n']:
        Sentences = Sentences[1:len(Sentences)]
    Senses = re.sub('@@@[@@@]*','\n\n',Senses)

    txt = open('text_words2.csv','w')
    smy = open('summary_words2.csv','w')
    txt.write(Sentences)
    smy.write(Senses)
    txt.close()
    smy.close()

#################################################################################
    
    txt = open('test_text_words2.csv','w')
    smy = open('test_summary_words2.csv','w')

    Sentences2 = []
    Senses2 = [] 

    #Changes and writing TEST data
    for i in range(0,len(Sentences_Split_test)):
        if Senses_Split_test[i] in ['HARD2', 'HARD3', 'cord', 'formation', 'phone', 'text', 'division', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'SERVE2', 'SERVE6']:
            Senses2.append(Senses_Split_test[i])
            Sentences2.append(Sentences_Split_test[i])
    for i in range(0,len(Sentences2)):
        Sentences2[i] = re.sub(' [ ]*','&&&',Sentences2[i])
    Sentences = '@@@'.join(Sentences2)
    Senses = '@@@'.join(Senses2)

    Sentences = re.sub('@@@[@@@]*','\n\n',Sentences)
    Sentences = re.sub('&&[&&]*','\n',Sentences)
    Sentences = re.sub('\n\n\n[\n]*','\n\n',Sentences)
    # print(Sentences)
    if Sentences[0] in [' ', '\n']:
        Sentences = Sentences[1:len(Sentences)]

    Senses = re.sub('@@@[@@@]*','\n\n',Senses)
    txt.write(Sentences)
    smy.write(Senses)
    txt.close()
    smy.close()

    txt = open('test_text_words.csv','w')
    smy = open('test_summary_words.csv','w')

    #Changes and writing TEST2 data
    for i in range(0,len(Sentences_Split_test)):
        Sentences_Split_test[i] = re.sub(' [ ]*','&&&',Sentences_Split_test[i])
    Sentences = '@@@'.join(Sentences_Split_test)
    Senses = '@@@'.join(Senses_Split_test)

    Sentences = re.sub('@@@[@@@]*','\n\n',Sentences)
    Sentences = re.sub('&&[&&]*','\n',Sentences)
    Sentences = re.sub('\n\n\n[\n]*','\n\n',Sentences)
    if Sentences[0] in [' ', '\n']:
        Sentences = Sentences[1:len(Sentences)]

    Senses = re.sub('@@@[@@@]*','\n\n',Senses)
    txt.write(Sentences)
    smy.write(Senses)
    txt.close()
    smy.close()

if __name__=="__main__":
    if len(sys.argv) == 2 :
        a=preprocess(sys.argv[1])
    elif len(sys.argv) == 3:
        a=preprocess(sys.argv[1], float(sys.argv[2]))
    elif len(sys.argv) == 5:
        a=preprocess(sys.argv[1], float(sys.argv[2]), int(sys.argv[3], ), int(sys.argv[4]))
    else:
        print("Wrong input format: Using 100 files")
        preprocess('all')


os.system("cp summary_words.csv summary_words_train.csv")
