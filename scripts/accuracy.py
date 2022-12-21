import numpy as np
import pandas as pd


if __name__ == '__main__':
    iris = pd.read_csv("data/Iris.csv")
    y_kmeans = np.loadtxt('y_train.csv', dtype='uint8')
    conf_matr = np.zeros((3, 3))
    print(iris.iloc[0, 5])
    for i in range(len(y_kmeans)):
        if iris.iloc[i, 5] == 'Iris-setosa':
            conf_matr[0, y_kmeans[i]] += 1
        if iris.iloc[i, 5] == 'Iris-versicolor':
            conf_matr[1, y_kmeans[i]] += 1
        if iris.iloc[i, 5] == 'Iris-virginica':
            conf_matr[2, y_kmeans[i]] += 1
    # print(conf_matr)

    correct = 0
    for i in range(3):
        correct += conf_matr[i].max()
    acc = np.zeros(1)
    acc[0] = correct / len(y_kmeans)
    print('Accuracy = ' + str(acc))
    np.savetxt('Accuracy.txt', acc)
