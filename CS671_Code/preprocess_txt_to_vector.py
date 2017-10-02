import os
import random
import sys
import re
import numpy as np
reload(sys)

def preprocess(checkword = 'all',per_split = 0.05):
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

    Sentences_Split = _tmpst
    Senses_Split = _tmpse 
    
    lenst = len(Sentences_Split)
    
    Sentences_Split_test = Sentences_Split[0:int(per_split*lenst)]
    Senses_Split_test = Senses_Split[0:int(per_split*lenst)]

    Sentences_Split  = Sentences_Split[int(per_split*lenst):lenst]
    Senses_Split = Senses_Split[int(per_split*lenst):lenst]

    print(len(Sentences_Split), len(Senses_Split),per_split)
#######################################################################

    for i in range(0,lenst):
        if Senses_Split[i].lower().strip() in ['hard2','hard3','interest1','interest5','phone','cord','formation','division','text']:
            for j in range(0,10):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() in ['hard1']:
            for j in range(0,2):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() in ['interest3','interest4']:
            for j in range(0,40):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'interest2':
            for j in range(0,200):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower().strip() == 'product':
            for j in range(0,4):
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
    
    print(len(Sentences_Split),len(Sentences_Split_test))

    #Changes and writing TRAINING data
    for i in range(0,len(Sentences_Split )):
        Sentences_Split[i] = re.sub(' [ ]*','&&&',Sentences_Split[i])
    Sentences = '@@@'.join(Sentences_Split)
    Senses = '@@@'.join(Senses_Split)

    Sentences = re.sub('@@@[@@@]*','\n\n',Sentences)
    Sentences = re.sub('&&[&&]*','\n',Sentences)
    Sentences = re.sub('\n\n\n[\n]*','\n\n',Sentences)
    Sentences = Sentences[1:len(Sentences)]

    Senses = re.sub('@@@[@@@]*','\n\n',Senses)
    txt.write(Sentences)
    smy.write(Senses)
    txt.close()
    smy.close()

    txt = open('test_text_words.csv','w')
    smy = open('test_summary_words.csv','w')

    #Changes and writing TEST data
    for i in range(0,len(Sentences_Split_test)):
        Sentences_Split_test[i] = re.sub(' [ ]*','&&&',Sentences_Split_test[i])
    Sentences = '@@@'.join(Sentences_Split_test)
    Senses = '@@@'.join(Senses_Split_test)

    Sentences = re.sub('@@@[@@@]*','\n\n',Sentences)
    Sentences = re.sub('&&[&&]*','\n',Sentences)
    Sentences = re.sub('\n\n\n[\n]*','\n\n',Sentences)
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
    else:
        print("Wrong input format: Using 100 files")
        preprocess('all')
