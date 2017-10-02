def couple(text_file, tag_file):
    text = open(text_file).read().split('\n')
    tags = open(tag_file).read().split('\n')

    textTag = []

    assert(len(text) == len(tags))

    lenT = len(text)

    for i in range(0, lenT):
        if text[i] != "" and tags[i] != "":
            textTag.append(text[i] + "_"  + tags[i])
        elif text[i] != "" and tags[i] == "":
            textTag.append(text[i])
        else:
            textTag.append("")

    opStr = '\n'.join(textTag)

    f= open(text_file, 'w')
    f.write(opStr)

try:
    couple('./text_words.csv', './text_words_tags.csv')
except:
    print("No such file: text_words.csv")

try:
    couple('./test_text_words.csv', './test_text_words_tags.csv')
except:
    print("No such file: test_text_words.csv")

