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


def Normalize(data, metadata, num_feat, num_sample):
    feature_mean, feature_std = np.zeros(num_feat), np.zeros(num_feat)
    for n in range(num_feat):
        feature_info = metadata[metadata.names()[n]]
        if feature_info[0] == 'numeric':
            vals = np.zeros(num_sample)     
            for m in range(num_sample):
                vals[m] = data[m][n]
            feature_mean[n] = np.mean(vals)
            feature_std[n] = np.std(vals)
            for m in range(num_sample):
                data[m][n] = 1.0 * (data[m][n] - feature_mean[n]) / feature_std[n]
    return data, feature_mean, feature_std

def Normalize2(data, metadata, num_feat, num_sample,feature_mean, feature_std):
    for n in range(num_feat):
        feature_info = metadata[metadata.names()[n]]
        if feature_info[0] == 'numeric':
            for m in range(num_sample):
                data[m][n] = 1.0 * (data[m][n] - feature_mean[n]) / feature_std[n]
    return data

def genW(row):
    weight = np.random.uniform(-0.01, 0.01, row)
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


def backprop(X, Y, weight, l, t):
    orderedIdx= range(len(Y))
    shuffle(orderedIdx)
    error, counts = 0,0

    for m in orderedIdx:
        rawInput = np.array(X[m])
        output = sig(np.dot(X[m], weight))
        delta = Y[m] - output
        weight_gradient = np.multiply(delta, rawInput)
        weight += l*weight_gradient
        error += Error(Y[m], output)
        if (output > 0.5 and Y[m] == 1):
            counts += 1
        elif (output < 0.5 and Y[m] == 0):
            counts += 1
    print ('%d\t%.9f\t%d\t%d' % (t+1, error, counts, len(Y) - counts))
    return weight, error, counts


def trainning(train, train_res, l, e, printOutput = True):
    weight = genW(len(train[0]))
    for e in range(e):
        weight, error, counts = backprop(train, train_res, weight, l, e)
    return weight



def classify(test, true_Y, weight):
    TP, TN, FP, FN = 0, 0, 0, 0
    bias = np.ones((1,))
    for m in range(len(true_Y)):
        output = sig(np.dot(test[m], weight))
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
e = int(str(sys.argv[2]))
fname_train = str(sys.argv[3])
fname_test = str(sys.argv[4])
train, train_res, metadata,num_feat, num_sample = takeInput(fname_train)
train, mean, std = Normalize(train, metadata, num_feat, num_sample)
m,n = np.shape(train)
bias = np.ones((m,1))
train = np.hstack((bias,train))
test, true_Y, metadata, num_feat, num_sample= takeInput(fname_test)
test = Normalize2(test, metadata, num_feat, num_sample, mean, std)
m,n = np.shape(test)
bias = np.ones((m,1))
test = np.hstack((bias,test))
weight = trainning(train, train_res, l, e)
classify(test, true_Y, weight)









