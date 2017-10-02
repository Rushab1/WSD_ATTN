import sys

Sent = open('./Sentences.txt', 'r').read().split('\n')
Sens = open('./Senses.txt', 'r').read().split('\r\n')

sentStr = ""
sensStr = ""

wordList = {
        'hard': ['HARD1', 'HARD2', 'HARD3'],
        'interest': ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'interest6'],
        'line': ['text', 'phone', 'product', 'formation', 'division', 'cord'],
        'serve': ['SERVE2', 'SERVE6', 'SERVE10', 'SERVE12']
        }

checkword = sys.argv[1]
checklist = wordList[checkword]
for i in range(0, len(Sens)):
    if Sens[i] in checklist:
        sentStr += Sent[i] + "\n"
        sensStr += Sens[i] + "\n"

sentFile = open('./Preprocess_Files/Temp/tmp', 'w')
sensFile = open('./Preprocess_Files/Temp/tmpkey', 'w')

sentFile.write(sentStr)
sensFile.write(sensStr)

sentFile.close()
sensFile.close()
