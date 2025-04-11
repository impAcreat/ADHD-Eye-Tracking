from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D, Conv2D, BatchNormalization



def build_model(inputShape):
    model = Sequential()
    model.add(Conv1D(32, 3,input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    #model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    #model.add(Dense(3, activation='softmax'))

    model.summary()
    
    return model