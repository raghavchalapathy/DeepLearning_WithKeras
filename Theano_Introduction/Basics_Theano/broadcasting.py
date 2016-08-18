__author__ = 'raghav'

import numpy as np
import theano
from theano import tensor as T


# Numpy example of broadcasting
xval = np.array([[1, 2, 3], [4, 5, 6]])
bval = np.array([[10, 20, 30]])
print xval.shape
print bval.shape
print xval + bval


xval = np.array([[0.1], [0.2]])
bval = np.array([[1,2,3]])
print xval.shape
print bval.shape
print xval + bval

# Theano equivalent declaring the variables as broadcastable
x = T.dmatrix('x')
b = theano.shared(bval, broadcastable=(True,False))
z = x + b
f = theano.function([x], z)

# print f(xval)

# Another example of broadcastable feature of theano
r = T.row('r')
# print(r.broadcastable)

c = T.col('c')
# print(c.broadcastable)


f = theano.function([r, c], r + c)
print(f([[1, 2, 3]], [[.1], [.2]]))