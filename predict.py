import tensorflow as tf
import numpy as np
import json
import os

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    eye_data = data['eye_data']
    id = data['id']
    label = data['label']
    eye_data = np.array(eye_data)
    eye_data = eye_data.astype(np.float32)
    return eye_data, id, label

def analyze_difference(samples):
    x_dif = np.abs(samples[:, 0]).mean()
    y_dif = np.abs(samples[:, 1]).mean()
    print(f"-- difference x: {x_dif}, y: {y_dif}")

class Predict():
    def __init__(self):
        self.model = tf.keras.models.load_model('checkpoint/model.h5')
        self.sequence_dim = 100
        self.sequence_lag = 1
        self.sequence_attributes = 2

    def predict(self, data):
        print(f"-- original data shape: {data.shape}")
        ## scale data
        data[:, 0] = data[:, 0] * 102.4
        data[:, 1] = data[:, 1] * 102.4

        data = self.calc_xy_velocity(data)
        analyze_difference(data)
        # print(f"-- velocity data shape: {data.shape}")
        data = self.make_sequences(data)
        # print(f"-- sequenced data shape: {data.shape}")
        result = self.model.predict(data)
        result = np.argmax(result, axis=1)
        # print(f"-- predict result shape: {result.shape}")
        return result
    
    def calc_xy_velocity(self, data):
        velX = [] #x values difference
        velY = [] #y values difference 

        for i in range(len(data) - 1):
            velX.append(float(data[i+1,0]) - float(data[i,0]) )
            velY.append(float(data[i+1,1]) - float(data[i,1]) )
        velX = np.array(velX)
        velY = np.array(velY)
        velocity = np.vstack([velX,velY]).T
        return velocity

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
    predict_model = Predict()
    folder_path = 'ourdata'
    file_paths = os.listdir(folder_path)
    with open('files/file_list.json', 'w') as f:
        json.dump(file_paths, f, indent=4)

    file_paths = [os.path.join(folder_path, file_path) for file_path in file_paths]

    predict_result = []
    index = 0
    for file_path in file_paths:
        eye_data, id, label = load_data(file_path)
    
        result = predict_model.predict(eye_data)

        ## analyze result
        unique_values, counts = np.unique(result, return_counts=True)
        frequencies = counts / len(result) * 100

        ## 
        predict_result.append({
            'id': id,
            'label': label,
            'counts': counts.tolist(),
            'frequencies': frequencies.tolist(),
            'result': result.tolist()
        })

        print(f"- {file_path} done")

        if index%50 == 0:
            with open('files/result.json', 'w') as f:
                json.dump(predict_result, f, indent=4)
        index += 1
        # break

    with open('files/result.json', 'w') as f:
        json.dump(predict_result, f, indent=4)
    print(f"-- result saved to result.json")

    
if __name__ == "__main__":
    main()
