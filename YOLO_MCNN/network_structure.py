# python library
import numpy as np

# chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import reporter


# networks
class MCNN1(chainer.Chain):

    def __init__(self, train= True):
        super(MCNN1, self).__init__(
            conv1 = L.Convolution2D(3, 6, 3, stride=1),
            conv2 = L.Convolution2D(6, 6, 3, pad=1),
            conv3 = L.Convolution2D(6, 6, 3, pad=1),
            l1 = L.Linear(486, 432),
            l2 = L.Linear(432, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


class MCNN2(chainer.Chain):

    def __init__(self, train= True):
        super(MCNN2, self).__init__(
            conv1 = L.Convolution2D(3, 6, 3, stride=1),
            conv2 = L.Convolution2D(6, 6, 3, pad=1),
            conv3 = L.Convolution2D(6, 6, 3, pad=1),
            l1 = L.Linear(864, 432),
            l2 = L.Linear(432, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


class MCNN3(chainer.Chain):

    def __init__(self, train= True):
        super(MCNN3, self).__init__(
            conv1 = L.Convolution2D(3, 3, 5, stride=1),
            conv2 = L.Convolution2D(3, 3, 5, pad=1),
            conv3 = L.Convolution2D(3, 3, 5, pad=1),
            l1 = L.Linear(768, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.l1(h))
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

class MCNN32(chainer.Chain):

    def __init__(self, train= True):
        super(MCNN32, self).__init__(
            conv1 = L.Convolution2D(3, 3, 5, stride=1),
            conv2 = L.Convolution2D(3, 3, 5, pad=2),
            conv3 = L.Convolution2D(3, 3, 5, pad=2),
            l1 = L.Linear(972, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        # print "conv1"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool1"
        # print h.shape
        h = F.relu(self.conv2(h))
        # print "conv2"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool2"
        # print h.shape
        h = F.relu(self.conv3(h))
        # print "conv3"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool3"
        # print h.shape
        h = F.relu(self.l1(h))
        # print "l1"
        # print h.shape
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


class MCNN4(chainer.Chain):

    def __init__(self, train= True):
        super(MCNN4, self).__init__(
            conv1 = L.Convolution2D(3, 3, 5),
            conv2 = L.Convolution2D(3, 3, 5),
            conv3 = L.Convolution2D(3, 3, 5),
            l1 = L.Linear(243, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.l1(h))
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


#model5
class CNN_thibault2(chainer.Chain):

    def __init__(self, train= True):
        super(CNN_thibault2, self).__init__(
            conv1 = L.Convolution2D(3, 6, 3, stride=1),
            conv2 = L.Convolution2D(6, 6, 3, pad=1),
            l1 = L.Linear(864, 432),
            l2 = L.Linear(432, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


class MCNN32_grayed(chainer.Chain):

    def __init__(self, train= True):
        super(MCNN32_grayed, self).__init__(
            conv1 = L.Convolution2D(1, 3, 5, stride=1),
            conv2 = L.Convolution2D(3, 3, 5, pad=2),
            conv3 = L.Convolution2D(3, 3, 5, pad=2),
            l1 = L.Linear(972, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        # print "conv1"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool1"
        # print h.shape
        h = F.relu(self.conv2(h))
        # print "conv2"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool2"
        # print h.shape
        h = F.relu(self.conv3(h))
        # print "conv3"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool3"
        # print h.shape
        h = F.relu(self.l1(h))
        # print "l1"
        # print h.shape
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

class MCNN33(chainer.Chain):

    def __init__(self, train= True):
        super(MCNN33, self).__init__(
            conv1 = L.Convolution2D(3, 3, 5),
            conv2 = L.Convolution2D(3, 3, 5),
            conv3 = L.Convolution2D(3, 3, 5),
            l1 = L.Linear(675, 100),
            lo = L.Linear(100, 2),
            bn1 = L.BatchNormalization(6),
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.h = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        # print "conv1"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool1"
        # print h.shape
        h = F.relu(self.conv2(h))
        # print "conv2"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool2"
        # print h.shape
        h = F.relu(self.conv3(h))
        # print "conv3"
        # print h.shape
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        # print "pool3"
        # print h.shape
        h = F.relu(self.l1(h))
        # print "l1"
        # print h.shape
        y = self.lo(h)
        return y

    def __call__(self, x, t):
        self.clear()
        y = self.forward(x)
        self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = accuracy.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
