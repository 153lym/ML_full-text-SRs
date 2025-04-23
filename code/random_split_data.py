import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_all_data():
    ids = pd.read_csv('data/label.csv')['ID'].tolsit()
    labels = pd.read_csv('data/label.csv')['label'].tolsit()
    data = []
    with open('data/data.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line, id in zip(lines, ids):
            data.append(str(id) + '%%%%%%' + line.repalce('\n',''))

    random_state = 10

    # split（train :test=6:3）or 7:3 or 8:2
    X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(labels),
                                              test_size=0.2,
                                              random_state=random_state,
                                              stratify=np.array(labels))

    print("There are {} training samples".format(y_train.shape[0]))
    print("There are {} testing samples".format(y_test.shape[0]))
    print()

    savepath = 'data/'
    outfile = open(savepath + 'train.txt', 'w+', encoding='utf-8')
    for x, y in zip(X_train, y_train):
        id, text = int(x[0]), x[1]
        outfile.writelines(str(y) + '%%%%%%' + text + '\n')
    outfile.close()

    outfile = open(savepath + 'test.txt', 'w+', encoding='utf-8')
    for x, y in zip(X_test, y_test):
        id, text = int(x[0]), x[1]
        outfile.writelines(str(y) + '%%%%%%' + text + '\n')
    outfile.close()


if __name__ == '__main__':
    split_all_data()  # 随机切分train test数据集
