from utils import load_data, preprocess
from model import build_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt


def load_process_filelist(file_list):
    inputs = []
    labels = []
    
    for file in file_list:
        x, y = load_data(file)
        x, y = preprocess(x, y)
        
        inputs.extend(x)
        labels.extend(y)
       
    inputs = np.array(inputs)
    labels = np.array(labels) 
    
    print(f"-- inputs shape: {inputs.shape}")
    print(f"-- labels shape: {labels.shape}")    
    
    return inputs, labels

def main():
    file_list = ['data/UH33_img_vy_labelled_MN.mat','data/UH47_img_Europe_labelled_MN.mat','data/UH47_img_Europe_labelled_RA.mat','data/UH21_img_Rome_labelled_MN.mat', 'data/UH27_img_vy_labelled_MN.mat']
    test_file_list = ['data/TH34_img_Europe_labelled_MN.mat', 'data/UH21_img_Rome_labelled_RA.mat']
    
    ## load data and preprocess
    inputs, labels = load_process_filelist(file_list)
    test_inputs, test_labels = load_process_filelist(test_file_list)
    
    ## load model
    model = build_model(inputs.shape[1:])
    
    model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
    
    ## train and test
    EPOCHS=20
    BATCH=100
    model.fit(inputs, labels, batch_size=BATCH, epochs=EPOCHS
                ,validation_data=(test_inputs, test_labels)
                )

    print("Training")
    cnnResults = model.predict(inputs)
    print(confusion_matrix(labels.argmax(axis=1), cnnResults.argmax(axis=1)))
    print(classification_report(labels.argmax(axis=1), cnnResults.argmax(axis=1)))
    print("CNN Accuracy: {:.2f}".format(accuracy_score(labels.argmax(axis=1), cnnResults.argmax(axis=1))))
    print("Cohen's Kappa {:.2f}".format(cohen_kappa_score(labels.argmax(axis=1), cnnResults.argmax(axis=1))))

    print("Test")
    cnnResults = model.predict(test_inputs)
    print(confusion_matrix(test_labels.argmax(axis=1), cnnResults.argmax(axis=1)))
    print(classification_report(test_labels.argmax(axis=1), cnnResults.argmax(axis=1)))
    CM=(confusion_matrix(test_labels.argmax(axis=1), cnnResults.argmax(axis=1)))
    print("CNN Accuracy: {:.2f}".format(accuracy_score(test_labels.argmax(axis=1), cnnResults.argmax(axis=1))))
    print("Cohen's Kappa {:.2f}".format(cohen_kappa_score(test_labels.argmax(axis=1), cnnResults.argmax(axis=1))))

    model.save("checkpoint/model.h5")

    ## visualize confusion matrix
    import seaborn as sns
    cm_normalized=np.round(CM/np.sum(CM, axis=1).reshape(-1, 1), 2)
    print(cm_normalized)
    sns.heatmap(cm_normalized, cmap='Blues', annot=True, cbar_kws={"orientation": "vertical", "label": "color bar"}, xticklabels=['fix','sac','pso'], yticklabels=['fix', 'sac', 'pso'])
    plt.xlabel("Predicted value")
    plt.ylabel("Actual value")
    plt.show()

if __name__ == "__main__":
    main()