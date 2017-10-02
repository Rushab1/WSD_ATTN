from sklearn.manifold import TSNE
import gensim
import csv
from gensim.models import word2vec, Word2Vec

Sent = open('text_words.csv', 'r').read().lower().split("\n\n")
Sens = open('summary_words.csv', 'r').read().lower().split("\n\n")

#Assume word at center of local window
tmp = Sent[0].split("\n")
checkword = tmp[len(tmp)/2]

wordvec_size = 100

for i in range(0, len(Sent)):
    Sent[i] = Sent[i].replace(checkword, Sens[i])
    Sent[i] = Sent[i].split("\n")


model=Word2Vec(Sent,sg=1,size=wordvec_size,window=5,min_count=0,workers=3,iter=10,negative=1)
model.delete_temporary_training_data(replace_word_vectors_with_normalized = True)

tsne = TSNE(n_components = 5, random_state = 0 )
m = tsne.fit_transform(model.wv.syn0)

SensVectors = []
senSet = list(set(Sens))

for  i in senSet:
    for j in range(0, len(model.wv.syn0)):
        if model.wv.syn0[j,:] == model[i]:
            SensVectors.append(m[j,:])

with open('senseVectors.csv', 'w') as csvfile:
    writer= csv.writer(csvfile, delimiter = ',')
    for i in SensVectors:
        writer.writerow(i)

f =  open('senseOrder.csv', 'w')
for i in senSet:
    f.write(i + "\n")
f.close()
