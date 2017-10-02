import sys

try:
    filename = sys.argv[1]
except:
    filename = 'text_words'

textTag = open(filename + '.csv').read().split('\n')

text = []
tag = []


lenT = len(textTag)

for i in range(0, lenT):
    try:
        tmp = textTag[i].split('_')
        text.append(tmp[0])
        tag.append(tmp[1])
    except:
        tag.append("")
    
opStr = '\n'.join(text)
f= open(filename + '.csv', 'w')
f.write(opStr)
f.close()

opStr = '\n'.join(tag)
f= open(filename + '_tags.csv', 'w')
f.write(opStr)
f.close()
