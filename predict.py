import tensorflow as tf
import numpy as np
import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    eye_data = data['eye_data']
    id = data['id']
    label = data['label']
    eye_data = np.array(eye_data)
    eye_data = eye_data.astype(np.float32)
    return eye_data, id, label

class Predict():
    def __init__(self):
        self.model = tf.keras.models.load_model('checkpoint/model.h5')
        self.sequence_dim = 100
        self.sequence_lag = 1
        self.sequence_attributes = 2

    def predict(self, data):
        print(f"-- original data shape: {data.shape}")
        ## scale data
        data[:, 0] = data[:, 0] * 1024.0
        data[:, 1] = data[:, 1] * 1024.0

        data = self.make_sequences(data)
        print(f"-- sequenced data shape: {data.shape}")
        result = self.model.predict(data)
        result = np.argmax(result, axis=1)
        print(f"-- predict result shape: {result.shape}")
        return result
    
    def make_sequences(self, samples):
        nsamples = []
        for i in range(0, samples.shape[0] - self.sequence_dim, self.sequence_lag):
            nsample = np.zeros((self.sequence_dim, self.sequence_attributes))
            for j in range(i, i + self.sequence_dim):
                nsample[j - i, 0] = samples[j, 0]
                nsample[j - i, 1] = samples[j, 1]
            nsamples.append(nsample)
        
        samples = np.array(nsamples)
        return samples
    
def main():
    file_path = 'test.json'
    eye_data, id, label = load_data(file_path)
    
    predict_model = Predict()
    result = predict_model.predict(eye_data)

    ## analyze result
    unique_values, counts = np.unique(result, return_counts=True)

    for value, count in zip(unique_values, counts):
        print(f"值 {value} 出现次数: {count}")

    frequencies = counts / len(result) * 100
    print("频率 (%):", dict(zip(unique_values, frequencies)))

    
if __name__ == "__main__":
    main()
