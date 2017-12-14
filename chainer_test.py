# -*- coding: UTF-8 -*-
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import data

#NNの構成
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

#predict関数 推論結果を返す
def predict(model, x_data):
    x = Variable(x_data.astype(np.float32))
    y = model.predictor(x)
    return np.argmax(y.data, axis = 1)


batchsize = 100
datasize = 60000
N = 10000

#データの準備
mnist = data.load_mnist_data()
x_all = mnist['data'].astype(np.float32) / 255
y_all = mnist['target'].astype(np.int32)
x_train, x_test = np.split(x_all, [datasize])
y_train, y_test = np.split(y_all, [datasize])

#モデル構築
model = L.Classifier(MLP())
optimizer = optimizers.Adam()
optimizer.setup(model)

#学習　datasize/batchsize回繰り返す
for epoch in range(20):
    print('epoch % d' % epoch)
    indexes = np.random.permutation(datasize)
    sum_loss, sum_accuracy = 0, 0
    for i in range(0, datasize, batchsize):
        x = Variable(np.asarray(x_train[indexes[i : i + batchsize]]))
        t = Variable(np.asarray(y_train[indexes[i : i + batchsize]]))
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * batchsize
        sum_accuracy += float(model.accuracy.data) * batchsize
    print('train mean loss={}, accuracy={}'.format(sum_loss / datasize, sum_accuracy / datasize))


    sum_loss, sum_accuracy = 0, 0
    for i in range(0, N, batchsize):
        x = Variable(np.asarray(x_test[i : i + batchsize]),volatile='on')
        t = Variable(np.asarray(y_test[i : i + batchsize]),volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * batchsize
        sum_accuracy += float(model.accuracy.data) * batchsize
    print('test mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))

#予測＆モデルの保存
p_test = np.empty((0, 784), float)
p_test = np.append(p_test, np.array([x_test[0]]), axis=0)
p_test = np.append(p_test, np.array([x_test[1]]), axis=0)

print(p_test)
print(predict(model, p_test))
print(y_test)

serializers.save_hdf5('myMLP.model',model)

