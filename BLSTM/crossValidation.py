import os
try:
	import matplotlib.pyplot as plt
except:
	print("matplotlib.pyplot NOT FOUND")
import sys
import numpy as np
import argparse
from numpy import random
from copy import deepcopy

txt = open('text_words.csv', 'r').read().split('\n\n')
sen = open('summary_words.csv', 'r').read().split('\n\n')

def crossValidate(txt, sen, k = 5, numRounds = -1, glove = False, glove_file = "../Glove.t7", LSTM_type = "birectional"):
    if numRounds == -1:
        numRounds = k

    assert(len(txt) == len(sen))
    lentxt= len(txt)

    textset = []
    for i in range(0, lentxt):
        textset.append((txt[i], sen[i]))

    textset = list(set(textset))

    lentxt = len(textset)
    txt = []
    sen = []
    for i in range(0, lentxt):
        txt.append(textset[i][0])
        sen.append(textset[i][1])

    slicedTxt = []
    slicedSen = []

    for i in range(0, k):
        slicedTxt.append(txt[int(1.0 * i * lentxt / k) : int(1.0 * ( i + 1 ) * lentxt / k) ])
        slicedSen.append(sen[int(1.0 * i * lentxt / k) : int(1.0 * ( i + 1 ) * lentxt / k) ])

    open('./testResults', 'w').close()
    open('./validResults', 'w').close()
    open('./avgValidResults', 'w').close()
    open('./confusionMatrix', 'w').close()

    for sliceNo in range(0, numRounds):
        selectSlice(slicedTxt, slicedSen, sliceNo, k)
        f = open('validResults' , 'a')
        f.write('\n')
        f.close()

        cmd = "th train.lua -eval True -confusionMatrix True -glove " + str(bool(glove)) + " -glove_file " + glove_file + " -LSTM_type " + LSTM_type
        os.system(cmd)
        print("Evaluating...")

        trainAcc = open('trainAcc').read().split()
        f = open('testResults', 'a')

        # for model in os.listdir('Models'):
            # cmd = "th eval.lua -model Models/" + model + " -testfile cv"
            # print(cmd)
            # os.system(cmd)
            # print("____________________")
            # cmd = "python quantify_results.py --label " + model + " --file validResults"
            # os.system(cmd)

            #One vs all
            # cmd = 'th genSentVec.lua -model Models/' + model + ' -set train'
            # os.system(cmd)
            # cmd = 'th genSentVec.lua -model Models/' + model + ' -set test'
            # os.system(cmd)
            # cmd = "python OVR.py " + model
            # os.system(cmd)

    os.system("sed -i 's/.*el//g' validResults && sed -i 's/_.* / /g' validResults")
    os.system("sed -i 's/.*el//g' OVRResultsLR && sed -i 's/_.* / /g' OVRResultsLR")#LogisticRegression
    os.system("sed -i 's/.*el//g' OVRResultsSVC && sed -i 's/_.* / /g' OVRResultsSVC")#Support Vector Classifier
    formatResults(numRounds, 'validResults', 'avgValidResults')
    # testAccuracy()

def selectSlice(slicedTxt, slicedSen, sliceNo, k):
    lenslice = len(slicedTxt[0])
    txtTrain = []
    senTrain = []
    txtStrValid = ""
    senStrValid = ""
    txtStrTest = ""
    senStrTest = ""

    for i in range(0, k):
        if i == sliceNo:
            tmp1 = deepcopy(slicedTxt[i])
            tmp2 = deepcopy(slicedSen[i])
            tmp = list(zip(tmp1, tmp2))
            random.shuffle(tmp)
            tmp1[:], tmp2[:] = zip(*tmp)
            del tmp
            #within test data - 50%test, 50%Validation
            for j in range(0, lenslice):
                # if j < 0.5 * lenslice:
                    txtStrTest += tmp1[j] + "\n\n"
                    senStrTest += tmp2[j] + "\n\n"
                # else:
                    # txtStrValid += tmp1[j] + "\n\n"
                    # senStrValid += tmp2[j] + "\n\n"
        else:
            tmp1 = deepcopy(slicedTxt[i])
            tmp2 = deepcopy(slicedSen[i])
            tmp = list(zip(tmp1, tmp2))
            random.shuffle(tmp)
            tmp1[:], tmp2[:] = zip(*tmp)
            del tmp

            for j in range(0 , lenslice):
                if j < 0.1 * lenslice:
                    txtStrValid += tmp1[j] + "\n\n"
                    senStrValid += tmp2[j] + "\n\n"
                else:
                    txtTrain.append(tmp1[j])
                    senTrain.append(tmp2[j])
    senSet = list(set(senTrain))

    print("Training Examples = " + str(len(set(txtTrain))) + "/" + str(lenslice*k))
    print("Testing Examples = " + str(len(txtStrTest.split("\n\n"))) + "/" + str(lenslice*k))
    print("Validation Examples = " + str(len(set(txtStrValid.split("\n\n")))) + "/" + str(lenslice*k))

    senCnt = [0]* len(senSet)
    for i in range(0, len(senSet)):
        senCnt[i] = senTrain.count(senSet[i])

    maxCnt = max(senCnt)
    senCnt_tmp = deepcopy(senCnt)

    for i in range(0, len(txtTrain)):
        ind = senSet.index(senTrain[i])
        if senCnt[ind] < maxCnt:
            for j in range(0, int(np.ceil(maxCnt/senCnt_tmp[ind]))):
                if senCnt[ind] >= maxCnt:
                    break
                txtTrain.append(txtTrain[i])
                senTrain.append(senTrain[i])
                senCnt[ind] += 1


    #No need to randomize the training data as it is randomized separately at the beginning of every epoch

    txtStrTrain = ""
    senStrTrain = ""
    for i in range(0, len(txtTrain)):
        txtStrTrain += txtTrain[i] + '\n\n'
        senStrTrain += senTrain[i] + '\n\n'

    txtStrValid = txtStrValid[:-1]
    senStrValid = senStrValid[:-1]
    txtStrTest = txtStrTest[:-1]
    senStrTest = senStrTest[:-1]
    txtStrTrain = txtStrTrain[:-1]
    senStrTrain = senStrTrain[:-1]

    f = open('./cv_text_words.csv', 'w')
    f.write(txtStrTrain)
    f = open('./cv_summary_words.csv', 'w')
    f.write(senStrTrain)
    f = open('./cv_text_words_valid.csv', 'w')
    f.write(txtStrValid)
    f = open('./cv_summary_words_valid.csv', 'w')
    f.write(senStrValid)
    f.close()
    f = open('./cv_text_words_test.csv', 'w')
    f.write(txtStrTest)
    f = open('./cv_summary_words_test.csv', 'w')
    f.write(senStrTest)

def testAccuracy():
    open('./testResults', 'w').close()
    f = open('avgValidResults', 'r').read().split('\n')
    maxAcc = 0
    argmax = ""
    for i in range(0, len(f)):
        tmp = f[i].split()
        if tmp != []:
            if float(tmp[1]) > maxAcc:
                maxAcc = float(tmp[1])
                argmax = tmp[0]

    model = "model" + argmax
    cmd = "th eval.lua -model Models/" + model + " -testfile test"
    os.system(cmd)
    cmd = "python quantify_results.py --label " + model + " --file testResults"
    os.system(cmd)
    os.system("sed -i 's/.*el//g' testResults && sed -i 's/_.* / /g' testResults")
    testAcc = open('testResults').read().strip()
    plot('avgValidResults', plottitle = 'BLSTM: Accuracy vs Epoch\nTest Accuracy: ' + testAcc)

def formatResults(k, filename, outputfilename, plottitle = "crossValidation Accuracy vs Epochs"):
    accList = open(filename, 'r').read().split("\n")
    accDict = {}
    for i in accList:
        tmp = i.split()
        if tmp == []:
            pass
        else:
            try:
                accDict[int(tmp[0])] += float(tmp[1])
            except:
                accDict[int(tmp[0])] = float(tmp[1])

    f = open(outputfilename, 'w')
    for i in accDict:
        f.write(str(i) + " " + str(accDict[i]/k) + "\n")
    f.close()

def plot(filename1, filename2 = -1, plottitle = 'BLSTM: Accuracy vs Epochs'):
    f = open(filename1, 'r').read().split('\n')
    x1 = []
    y1 = []
    for i in range(0, len(f)):
        if f[i] != []:
            tmp = f[i].split()
            if tmp != []:
                x1.append(int(tmp[0]))
                y1.append(float(tmp[1]))
    print(x1,y1)
    plt.figure(1)
    plt.plot(x1, y1, 'r-' )
    plt.title(plottitle)

    if filename2 != -1:
        f = open(filename2, 'r').read().split('\n')
        x2 = []
        y2 = []
        for i in range(0, len(f)):
            if f[i] != []:
                tmp = f[i].split()
                if tmp != []:
                    x2.append(int(tmp[0]))
                    y2.append(float(tmp[1]))
        print(x2, y2, f)
        plt.plot(x2,y2,'r-')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "cross-validation parameters")
    parser.add_argument('-type', type = str, default = "CV", nargs = '?')
    parser.add_argument('-k', type = int, default = 10, nargs = '?')
    parser.add_argument('-numRounds', type = int)
    parser.add_argument('-glove', type = int, default = 0, nargs = '?')
    parser.add_argument('-glove_file', type = str, default = "../Glove.t7", nargs = '?')
    parser.add_argument('-LSTM_type', type = str, default = "bidirectional", nargs = '?')
    args = parser.parse_args()

    if args.type == "NOCV":
        print("NOCV: Copying training data to cv_textwords.csv and cv_summary_words.csv")
        os.system('cp text_words.csv ./cv_text_words.csv')
        os.system('cp summary_words.csv ./cv_summary_words.csv')
    elif args.type == "FORMAT":
        print("FORMAT")
        formatResults(5)
    elif args.type == "CV":
        if args.numRounds == None:
            args.numRounds = args.k
        print("CV: " + str(args.k) + "-Fold Cross-Validation with " + str(args.numRounds) + " rounds")
        print("Using glove vectors: " + str(bool(args.glove)))
        print("LSTM Type: " + args.LSTM_type)
        crossValidate(txt, sen, k = args.k, numRounds = args.numRounds, glove = args.glove, glove_file = args.glove_file, LSTM_type = args.LSTM_type)
