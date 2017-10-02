from numpy import genfromtxt
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
from sklearn.svm import SVC

try:
    prefix = sys.args[1]
except:
    prefix = ""

#clear files
# open("OVRResultsLR", "w").close()
# open("OVRResultsSVC", "w").close()

flr = open("OVRResultsLR", "a")
fsv = open("OVRResultsSVC", "a")
=======
>>>>>>> parent of d1025b3... Temporary

xtrain = genfromtxt("./train_sent.csv", delimiter= ',')
ytrain = genfromtxt("./train_sense.csv", delimiter= ',')

xtest = genfromtxt("./test_sent.csv", delimiter= ',')
ytest = genfromtxt("./test_sense.csv", delimiter= ',')

l = LogisticRegression(max_iter = 1000, multi_class = 'ovr')
l.fit(xtrain, ytrain)
print(l.score(xtrain, ytrain), l.score(xtest, ytest) )
