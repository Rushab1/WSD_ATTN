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

# print('=====>' + str(len(target)))
assert(len(target) == len(pred))

accCount = 0
for i in range(0,len(target)-1):
    if target[i] == pred[i]:
        accCount += 1
    else:
        print('*' + target[i] + '* : *' + pred[i] + '*')

correction =0
print(sys.argv[1] + " = " + str(1.0*accCount/(len(target)-1-correction)*100 ))
