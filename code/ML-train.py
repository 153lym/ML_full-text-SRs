
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import pickle

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn import ensemble


def read_file(file):
    f = open(file, 'r', encoding="utf-8")
    data, y = [], []
    for line in f.readlines():
        line = line.replace('"', '').replace('\n', '').split('%%%%%%')
        text = str(line[1])
        label = int(line[0].strip())
        data.append(text)
        y.append(label)
    return data, y


def processing_data(raw_data):
    # 分句
    sentences = sent_tokenize(raw_data)
    # 分词
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

    # 去除停用词、标点符号和数字
    stop_words = set(stopwords.words('english'))

    filtered_sentences = []
    for sentence in tokenized_sentences:
        filtered_sentence = [word for word in sentence if word.isalpha() and word not in stop_words]
        if len(filtered_sentence) == 0:
            continue
        else:
            filtered_sentences.append(filtered_sentence)
    return filtered_sentences


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def get_tfidf(content):
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(content)
    ## f1 = open('get_feature_names_out.txt', 'w+',encoding='utf-8')
    ## for vf in vectorizer.get_feature_names_out():
    ##     f1.write(vf + '\n')  # 特征名称
    #
    dn = []
    for key, value in vectorizer.vocabulary_.items():
        dn.append([value, key])
    # f1.close()
    pd.DataFrame(dn).to_csv('vocabulary_title_abs_20231107.csv')

    tfidftransformer = TfidfTransformer()
    tfidf = tfidftransformer.fit_transform(features)  # 先转换成词频矩阵，再计算TFIDF值

    return tfidf, vectorizer, tfidftransformer


def get_stop_text(data):
    str_list = []
    # 获得每篇文献分词+去除停用词之后的text
    for td in tqdm(data):
        td_list = processing_data(td)
        td_str = ''
        for tl in td_list:
            tl_str = ' '.join(tl) + ' '
            td_str += tl_str
        str_list.append(td_str)
    return str_list


def get_data_feature(file_paths):
    f = open('saved_model/tfidftransformer.pickle', 'rb')
    tfidftransformer = pickle.load(f)
    f.close()

    f = open('saved_model/vectorizer.pickle', 'rb')
    vectorizer = pickle.load(f)
    f.close()

    test_data, test_y = read_file(file_paths)
    test_str_list = get_stop_text(test_data)
    # 提取文本特征
    test_tfidf = tfidftransformer.transform(vectorizer.transform(test_str_list))
    return test_tfidf, test_y


# nbayes
def nbayes_model(X_train, y_train):
    clf = GaussianNB()

    scoring = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc']
    scores = cross_validate(clf, X_train, y_train, cv=5, scoring=scoring, return_estimator=True)
    print(scores)
    tmp = []
    score_arr = []
    for sco in scoring:
        arr = scores['test_' + sco]
        score_arr.append(arr)
        mean_arr = np.mean(arr)
        # 计算标准差
        std = np.std(arr)
        # 计算95%置信度
        confidence = 1.96 * std / np.sqrt(len(arr))
        tmp.append(str(round(mean_arr, 4)) + '±' + str(round(confidence, 4)))

    score_arr.append(tmp)
    # 保存5折交叉验证的模型
    for i, est in enumerate(scores['estimator']):
        f = open('saved_model/nbayes/nbayes_fold_%d.pickle' % i, 'wb')
        pickle.dump(est, f)
        f.close()

    index_name = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc', 'average_score']
    pd.DataFrame(score_arr, index=index_name).to_csv('results/nbayes/nbayes_5folds.csv',
                                                     header=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'None'])


# GBDT
def gradient_boosting_model(X_train, y_train, X_test, y_test):
    gb_clf = ensemble.GradientBoostingClassifier(n_estimators=1000)

    scoring = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc']
    scores = cross_validate(gb_clf, X_train, y_train, cv=5, scoring=scoring, return_estimator=True)
    print(scores)
    tmp = []
    score_arr = []
    for sco in scoring:
        arr = scores['test_' + sco]
        score_arr.append(arr)
        mean_arr = np.mean(arr)
        # 计算标准差
        std = np.std(arr)
        # 计算95%置信度
        confidence = 1.96 * std / np.sqrt(len(arr))
        tmp.append(str(round(mean_arr, 4)) + '±' + str(round(confidence, 4)))
    score_arr.append(tmp)
    # 保存5折交叉验证的模型
    for i, est in enumerate(scores['estimator']):
        f = open('saved_model/GBDT/GBDT_fold_%d_20231206.pickle' % (i+1), 'wb')
        pickle.dump(est, f)
        f.close()

    index_name = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc', 'average_score']
    pd.DataFrame(score_arr, index=index_name).to_csv('results/GBDT/GBDT_5folds.csv',
                                                     header=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'None'])



# LR
def lr_model(X_train, y_train):
    # clf = LogisticRegression(solver ='liblinear')
    clf = LogisticRegressionCV(cv=5, random_state=2021)

    scoring = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc']
    scores = cross_validate(clf, X_train, y_train, cv=5, scoring=scoring, return_estimator=True)
    print(scores)
    tmp = []
    score_arr = []
    for sco in scoring:
        arr = scores['test_' + sco]
        score_arr.append(arr)
        mean_arr = np.mean(arr)
        # 计算标准差
        std = np.std(arr)
        # 计算95%置信度
        confidence = 1.96 * std / np.sqrt(len(arr))
        tmp.append(str(round(mean_arr, 4)) + '±' + str(round(confidence, 4)))

    score_arr.append(tmp)
    # 保存5折交叉验证的模型
    for i, est in enumerate(scores['estimator']):
        f = open('saved_model/LR/LR_fold_%d.pickle' % i, 'wb')
        pickle.dump(est, f)
        f.close()

    index_name = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc', 'average_score']
    pd.DataFrame(score_arr, index=index_name).to_csv('results/LR/LR_5folds.csv',
                                                     header=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'None'])

# NN
def NN_model(X_train, y_train):
    clf = MLPClassifier(random_state=2022, validation_fraction=0.2,max_iter=1000)

    scoring = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc']
    scores = cross_validate(clf, X_train, y_train, cv=5, scoring=scoring, return_estimator=True)
    print(scores)
    tmp = []
    score_arr = []
    for sco in scoring:
        arr = scores['test_' + sco]
        score_arr.append(arr)
        mean_arr = np.mean(arr)
        # 计算标准差
        std = np.std(arr)
        # 计算95%置信度
        confidence = 1.96 * std / np.sqrt(len(arr))
        tmp.append(str(round(mean_arr, 4)) + '±' + str(round(confidence, 4)))

    score_arr.append(tmp)
    # 保存5折交叉验证的模型
    for i, est in enumerate(scores['estimator']):
        f = open('saved_model/NN/NN_fold_%d.pickle' % i, 'wb')
        pickle.dump(est, f)
        f.close()

    index_name = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc', 'average_score']
    pd.DataFrame(score_arr, index=index_name).to_csv('results/NN/NN_5folds.csv',
                                                     header=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'None'])
# RF
def rf_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=1000)

    scoring = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc']
    scores = cross_validate(clf, X_train, y_train, cv=5, scoring=scoring, return_estimator=True)
    print(scores)
    tmp = []
    score_arr = []
    for sco in scoring:
        arr = scores['test_' + sco]
        score_arr.append(arr)
        mean_arr = np.mean(arr)
        # 计算标准差
        std = np.std(arr)
        # 计算95%置信度
        confidence = 1.96 * std / np.sqrt(len(arr))
        tmp.append(str(round(mean_arr, 4)) + '±' + str(round(confidence, 4)))

    score_arr.append(tmp)
    # 保存5折交叉验证的模型
    for i, est in enumerate(scores['estimator']):
        f = open('saved_model/RF/RF_fold_%d.pickle' % i, 'wb')
        pickle.dump(est, f)
        f.close()

    index_name = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc', 'average_score']
    pd.DataFrame(score_arr, index=index_name).to_csv('results/RF/RF_5folds.csv',
                                                     header=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'None'])

# SVM
def svm_model(X_train, y_train):
    clf = SVC(C=10.0, probability=True,kernel='linear')

    scoring = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc']
    scores = cross_validate(clf, X_train, y_train, cv=5, scoring=scoring, return_estimator=True)
    print(scores)
    tmp = []
    score_arr = []
    for sco in scoring:
        arr = scores['test_' + sco]
        score_arr.append(arr)
        mean_arr = np.mean(arr)
        # 计算标准差
        std = np.std(arr)
        # 计算95%置信度
        confidence = 1.96 * std / np.sqrt(len(arr))
        tmp.append(str(round(mean_arr, 4)) + '±' + str(round(confidence, 4)))

    score_arr.append(tmp)
    # 保存5折交叉验证的模型
    for i, est in enumerate(scores['estimator']):
        f = open('saved_model/SVM/SVM_fold_%d.pickle' % i, 'wb')
        pickle.dump(est, f)
        f.close()

    index_name = ['precision', 'recall', 'f1', 'accuracy', 'average_precision', 'roc_auc', 'average_score']
    pd.DataFrame(score_arr, index=index_name).to_csv('results/SVM/SVM_5folds.csv',
                                                     header=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'None'])



if __name__ == '__main__':
    file_paths = 'data/'

    model_name = ['GBDT','LR','nbayes','NN','RF','SVM']
    for mna in model_name:
        if mna == 'GBDT':
            train_tfidf, train_y = get_data_feature(file_paths + '/train.txt')
            gradient_boosting_model(train_tfidf.toarray(), np.array(train_y))

        if mna == 'NB':
            train_tfidf, train_y = get_data_feature(file_paths + '/train.txt')
            nbayes_model(train_tfidf.toarray(), np.array(train_y))

        if mna == 'LR':
            train_tfidf, train_y = get_data_feature(file_paths + '/train.txt')
            lr_model(train_tfidf.toarray(), np.array(train_y))

        if mna == 'NN':
            train_tfidf, train_y = get_data_feature(file_paths + '/train.txt')
            NN_model(train_tfidf.toarray(), np.array(train_y))

        if mna == 'RF':
            train_tfidf, train_y = get_data_feature(file_paths + '/train.txt')
            rf_model(train_tfidf.toarray(), np.array(train_y))

        if mna == 'SVM':
            train_tfidf, train_y = get_data_feature(file_paths + '/train.txt')
            svm_model(train_tfidf.toarray(), np.array(train_y))