import os
import random
import sys
import re
import numpy as np
reload(sys)

def preprocess(numTrain, numTest,checkword = 'all'):
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
    


    Sentences_Split = Sentences.split('\n')
    Senses_Split = Senses.split('\n')
    
    lenst = len(Sentences_Split)
    lense = len(Senses_Split)

    for i in range(0,lenst):
        if Senses_Split[i].lower().strip() in ['hard2','hard3','interest1','interest5','phone','cord','formation','division','text']:
            for j in range(0,10):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower() in ['interest3','interest4']:
            for j in range(0,40):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])
        elif Senses_Split[i].lower() == 'interest2':
            for j in range(0,200):
                Sentences_Split.append(Sentences_Split[i])
                Senses_Split.append(Senses_Split[i])

    Sentences_Split = Sentences_Split[0:len(Sentences_Split) - 1] 
    Senses_Split = Senses_Split[0:len(Senses_Split) - 1]

    tmp = list(zip(Sentences_Split,Senses_Split))
    random.shuffle(tmp)
    
    Sentences_Split[:], Senses_Split[:] = zip(*tmp)
    

    for i in range(0,len(Sentences_Split)):
        Senses_Split[i] = Senses_Split[i].replace("\r","")
        if Senses_Split[i] == "":
            Senses_Split[i] = "some_unknown_symbol"
            # print("some unknown symbol replaced")
    
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

    Senses_Split_train = Senses_Split[0:numTrain]
    Sentences_Split_train = Sentences_Split[0:numTrain]

    Senses_Split_test = Senses_Split[numTrain:numTrain + numTest]
    Sentences_Split_test = Sentences_Split[numTrain:numTrain + numTest]

    _tmp1 = set(zip(Sentences_Split_train,Senses_Split_train))
    _tmp2 = set(zip(Sentences_Split_test,Senses_Split_test))
    # print(len(Sentences_Split_test))
    _tmp = np.array(list(set(_tmp2).difference(_tmp1)))
    Sentences_Split_test = list(_tmp[:,0])
    Senses_Split_test = list(_tmp[:,1])
    print(len(Sentences_Split_train),len(Sentences_Split_test))

    #Changes and writing TRAINING data
    for i in range(0,len(Sentences_Split_train)):
        Sentences_Split_train[i] = re.sub(' [ ]*','&&&',Sentences_Split_train[i])
    Sentences = '@@@'.join(Sentences_Split_train)
    Senses = '@@@'.join(Senses_Split_train)

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
    if len(sys.argv) == 3 :
        a=preprocess(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 4:
        a=preprocess(int(sys.argv[1]), int(sys.argv[2]),sys.argv[3])
    else:
        print("Wrong input format: Using 100 files")
        preprocess(100,100)
