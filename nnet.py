import numpy as np
import scipy.io.arff as arff
import sys
import random
from random import shuffle

def takeInput(inFile):
    data, metadata = arff.loadarff(inFile)
    num_feat = len(metadata.names()) -1
    num_sample = len(data)
    feature_info = metadata[metadata.names()[num_feat]]
    feature, Y = [], []
    for m in range(num_sample):
        featureVector = []
        for n in range(num_feat):
            this_feature_info = metadata[metadata.names()[n]]
            if this_feature_info[0] == 'numeric':
                featureVector.append(data[m][n])
            else:
                featureVector.append(metadata[metadata.names()[n]][1].index(data[m][n]))
        feature.append(featureVector)
        Y.append(feature_info[1].index(data[m][num_feat]))
    return feature, Y, metadata, num_feat, num_sample


def Normalize(data, metadata, num_features, num_instances):
    feature_mean, feature_std = np.empty(num_features), np.empty(num_features)
    for n in range(num_features):
        feature_info = metadata[metadata.names()[n]]
        if feature_info[0] == 'numeric':
            vals = np.empty(num_instances)     
            for m in range(num_instances):
                vals[m] = data[m][n]
            feature_mean[n] = np.mean(vals)
            feature_std[n] = np.std(vals)
            for m in range(num_instances):
                data[m][n] = 1.0 * (data[m][n] - feature_mean[n]) / feature_std[n]
    return data


def initWeights(h, row):
    weight = []     
    weight.append(np.random.uniform(-0.01, 0.01, (h, row)))
    h = h + 1
    weight.append(np.random.uniform(-0.01, 0.01, h))  #+1for bias
    return weight


def sig(input):
    return np.divide(1,(np.add(1,np.exp(-input))), dtype = np.float64)

def Error(y, y_hat):
    return -y * np.log(y_hat) - (1-y) * np.log(1-y_hat)


def gradient(feature, Y, O, weight, grad_1, grad_2, hiddenLayer):
    delta_o = Y - O
    delta_h = delta_o * weight * hiddenLayer * (1 - hiddenLayer)
    grad_2 = delta_o * hiddenLayer
    delta_h = delta_h[1:]
    grad_1 = np.outer(delta_h, feature)
    return grad_1, grad_2

def update_weight(l, tmp_w, w):
    w += l * tmp_w
    return w

def backprop(feature, Y, wts, l, t):
    grad = wts[:]
    error, counts = 0,0
    orderedIdx= range(len(Y))
    shuffle(orderedIdx)
    for m in orderedIdx:
        rawInput = np.array(feature[m])
        hid = sig(np.dot(wts[0], feature[m]))
        np.insert(hid, 0, 1.0)
        bias = np.ones((1,))
        hid = np.hstack((bias,hid))
        output = sig(np.dot(hid, wts[1]))
        grad[0], grad[1] = gradient(rawInput, Y[m], output, wts[1], grad[0], grad[1], hid)
        error += Error(Y[m], output)
        if (output > 0.5 and Y[m] == 1):
            counts += 1
        elif (output < 0.5 and Y[m] == 0):
            counts += 1
        wts[1] = update_weight(l, grad[1], wts[1]) 
        wts[0] = update_weight(l, grad[0], wts[0]) 
    print ('%d\t%.9f\t%d\t%d' % (t+1, error, counts, len(Y) - counts))

    return wts, error, counts


def trainning(train, train_res, h, l, e, printOutput = True):
    wts = initWeights(h, len(train[0]))
    for e in range(e):
        wts, error, counts = backprop(train, train_res, wts, l, e)
    return wts


def classify(test, true_Y, wts):
    TP, TN, FP, FN = 0, 0, 0, 0
    bias = np.ones((1,))
    for m in range(len(true_Y)):
        hid = sig(np.dot(wts[0], test[m]))
        hid = np.hstack((bias,hid))
        output = sig(np.dot(hid, wts[1]))
        if true_Y[m] == 0:
            if output < 0.5:
                TN +=1
                predict = 0
            else:
                FP +=1
                predict = 1
        elif true_Y[m] == 1:
            if output > 0.5:
                TP +=1
                predict = 1
            else:
                FN +=1
                predict = 0
        print ('%.9f\t%d\t%d' % (output, predict, true_Y[m]))

    P = 1.0*TP/(TP+FP)
    R = 1.0*TP/(TP+FN)
    F1 = 1.0*2*(P*R)/(P+R)
    print ('%d\t%d' % ((TP+TN), (FP+FN)))
    print ('%.12f' % F1)


args = [arg for arg in sys.argv]

l = float(str(sys.argv[1]))
h = int(str(sys.argv[2]))
e = int(str(sys.argv[3]))
fname_train = str(sys.argv[4])
fname_test = str(sys.argv[5])
train, train_res, metadata,num_feat, num_sample = takeInput(fname_train)
train = Normalize(train, metadata, num_feat, num_sample)
m,n = np.shape(train)
bias = np.ones((m,1))
train = np.hstack((bias,train))
test, true_Y, metadata, num_feat, num_sample= takeInput(fname_test)
test = Normalize(test, metadata, num_feat, num_sample)
m,n = np.shape(test)
bias = np.ones((m,1))
test = np.hstack((bias,test))
weight = trainning(train, train_res, h, l, e)
classify(test, true_Y, weight)









