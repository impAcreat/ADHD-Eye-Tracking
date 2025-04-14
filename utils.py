from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def load_data(filename):
    mat = loadmat(filename)
    
    mdata = mat['ETdata']
    mtype = mdata.dtype
    ndata = {n: mdata[n][0,0] for n in mtype.names}

    data_raw = ndata['pos']
    screenRes = ndata['screenRes']
    
    pdata = pd.DataFrame(data_raw)
    wRes = screenRes[0][0]
    hRes = screenRes[0][1]
    # print(f"-- screenRes: {wRes}, {hRes}")

    x=pdata.iloc[:, 3:5].values
    y=pdata.iloc[:, 5].values
    
    x[:, 0] = np.array(x[:, 0])/1.0
    x[:, 1] = np.array(x[:, 1])/0.75
  
    print("File",filename,"opened")
    return x ,y


def calc_xy_velocity(data):
    velX = [] #x values difference
    velY = [] #y values difference 

    for i in range(len(data) - 1):
      velX.append(float(data[i+1,0]) - float(data[i,0]) )
      velY.append(float(data[i+1,1]) - float(data[i,1]) )
    velX = np.array(velX)
    velY = np.array(velY)
    velocity = np.vstack([velX,velY]).T
    return velocity


def make_sequences(samples, labels, sequence_dim = 100, sequence_lag = 1, sequence_attributes = 2):
    nsamples = []
    nlabels = [] 
    for i in range(0,samples.shape[0]-sequence_dim,sequence_lag):
            nsample = np.zeros((sequence_dim,sequence_attributes))
            for j in range(i,i+sequence_dim):
                nsample[j-i,0] = samples[j,0]
                nsample[j-i,1] = samples[j,1]
            nlabel = labels[i+sequence_dim//2]
            nsamples.append(nsample)
            nlabels.append(nlabel)
        
    samples = np.array(nsamples)
    labels = np.array(nlabels)
    return samples,labels 

def binary_label(labels):
    lb = LabelBinarizer()
    lb.fit(labels)
    labels = lb.transform(labels)
    return labels
    

def preprocess(samples, labels, sequence_dim = 100, sequence_lag = 1, sequence_attributes = 2):
    samples = calc_xy_velocity(samples)

    # analyze_difference(samples)

    samples, labels = make_sequences(samples, labels, sequence_dim, sequence_lag, sequence_attributes)
    labels = binary_label(labels)
    return samples, labels

def analyze_difference(samples):
    x_dif = np.abs(samples[:, 0]).mean()
    y_dif = np.abs(samples[:, 1]).mean()
    print(f"-- difference x: {x_dif}, y: {y_dif}")
