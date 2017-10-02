import numpy as np

a = np.genfromtxt('tmpresults', delimiter = ' ')

b = np.transpose(np.sort(np.transpose(a),1))

np.savetxt('tmpresults', b, delimiter = ' ')
