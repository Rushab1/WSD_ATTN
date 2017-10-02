import re 
import numpy as np
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Preprocess data")
    parser.add_argument('--file', type = str)
    parser.add_argument('--label', type = str)
    parser.add_argument('--confusionMatrix', type = str)
    args = parser.parse_args()

f = open('target_summary.txt')
target = f.read()
f.close()

target = re.sub(' <eos>','',target)
target = target.split('\n')

f = open('predicted_summary.txt')
pred = f.read().split('\n')
f.close()

f = open('evalText.txt')
evalTxt = f.read().split('\n')
f.close()

assert(len(target) == len(pred))

accCount = 0
unk0 = 0 #unknown word count for wrong predictions
unk1 = 1 #unknown word count for correct predictions
unkAvg0 = 0
unkAvg1 = 0
tot0 = 0 #Number of wrong predictions
tot1 = 1 #Number of correct predictions

if target[len(target) - 1] == '':
    del target[len(target) - 1]
if pred[len(pred) - 1] == '':
    del pred[len(pred) - 1]

senseSet = list(set(target))
numSenses = len(senseSet)
confusionMatrix = np.zeros([numSenses, numSenses])

for i in range(0,len(target)):
    indTarget = senseSet.index(target[i])
    indPred = senseSet.index(pred[i])
    confusionMatrix[indTarget][indPred] += 1

    if target[i] == pred[i]:
        tot1 += 1
        unkCount = evalTxt[i].split().count('<unknown>')
        unkAvg1 += unkCount
        if unkCount >=4:
            unk1 += 1
        # print(str(i)+":"+ pred[i] + "-" +  target[i] + "-1")
        accCount += 1
    else:
        tot0 += 1
        unkCount = evalTxt[i].split().count('<unknown>')
        unkAvg0 += unkCount
        if unkCount >=4:
            unk0 += 1
        # print(str(i)+":" + pred[i] + "-" +  target[i] + "-0 : " + str(unkCount))
        # print('*' + target[i] + '* : *' + pred[i] + '*')


accuracy = str(1.0*accCount/(len(target))*100 )
try:
    if args.file != None:
        writeFile = open(args.file, "a")
    # else:
        # writeFile = open("tmpresults", "a")
        if args.label != None:
            writeFile.write(args.label + " " +accuracy+"\n")
        else:
            writeFile.write(accuracy + "\n")
        writeFile.close()
    print("Accuracy" + " = " + accuracy)
except:
    print("Accuracy" + " = " + accuracy)
print("\n")
print("Number of wrong predictions : " + str(tot0) + "\nSentences with unknown words >= 4 : " + str(unk0) + "\nMean number of wrong words persentence : " + str(1.0*unkAvg0/tot0))


print("\n")
print("Number of correct predictions : " + str(tot1) + "\nSentences with unknown words >= 4 : " + str(unk1) + "\nMean number of correct words persentence : " + str(1.0*unkAvg1/tot1))

print(senseSet)
print(confusionMatrix)

n = confusionMatrix.shape[0]
F1 = 0
for i in range(0, n):
    precision = confusionMatrix[i][i]/sum(confusionMatrix[:,i])
    recall = confusionMatrix[i][i]/sum(confusionMatrix[i])
    F1 +=  2*precision*recall/(precision + recall)

F1 /= n

print("F1 SCORE: " + str(F1))

if args.confusionMatrix != None:
    f = open("confusionMatrix", "a")
    f.write(args.confusionMatrix)
    f.write(str(confusionMatrix) + " \n F1 SCORE: " + str(F1) + "\n\n")
    f.close()

