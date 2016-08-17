__author__ = 'raghav'

import theano

from theano import tensor as T
from theano.printing import debugprint

x = T.vector('x')
W = T.matrix('W')
b = T.vector('b')
dot = T.dot(x, W)
out = T.nnet.sigmoid(dot + b)

f = theano.function(inputs=[x, W], outputs=dot)

debugprint(f)

from theano.printing import pydotprint
pydotprint(f, outfile='/Users/raghav/Documents/Uni/SummerSchool2015-DeepLearning/Intro_Theano/pydotprint_f.png')
g = theano.function([x, W, b], out)
h = theano.function([x, W, b], [dot, out])
i = theano.function([x, W, b], [dot + b, out])
debugprint(g)


import numpy as np

np.random.seed(42)

W_val = np.random.randn(2,2)
x_val = np.random.rand(2)

print W_val[0]
print x_val
b_val = np.ones(2)
print b_val

print f(x_val, W_val)
