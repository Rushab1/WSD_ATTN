import re 
import sys

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
unk0 = 0
unk1 = 1
unkAvg0 = 0
unkAvg1 = 0
tot0 = 0
tot1 = 1
for i in range(0,len(target)-1):
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

correction =0
# print("\n===>"+str(len(target)))

accuracy = str(1.0*accCount/(len(target)-1-correction)*100 )
try:
<<<<<<< HEAD
    if args.file != None:
        writeFile = open(args.file, "a")
    # else:
        # writeFile = open("tmpresults", "a")

        if args.label != None:
            writeFile.write(args.label + " " + accuracy + "\n")
        else:
            writeFile.write(accuracy + "\n")
        writeFile.close()

    print("Accuracy = " + accuracy)
except:
    print("Accuracy" + " = " + accuracy)
print("\n")

if tot0 != 0:
    print("Number of wrong predictions : " + str(tot0) + "\nSentences with unknown words >= 4 : " + str(unk0) + "\nMean number of wrong words persentence : " + str(1.0*unkAvg0/tot0))
else:
    print("No wrong predictions")
=======
    writeFile = open("tmpresults", "a")
    writeFile.write(sys.argv[1] + " " +accuracy+"\n")
    writeFile.close()
    # print(sys.argv[1] + " = " + accuracy)
except:
    print("Accuracy" + " = " + accuracy)
print("\n")
print("Number of wrong predictions : " + str(tot0) + "\nSentences with unknown words >= 4 : " + str(unk0) + "\nMean number of wrong words persentence : " + str(1.0*unkAvg0/tot0))

>>>>>>> parent of d1025b3... Temporary

print("\n")
print("Number of correct predictions : " + str(tot1) + "\nSentences with unknown words >= 4 : " + str(unk1) + "\nMean number of correct words persentence : " + str(1.0*unkAvg1/tot1))
