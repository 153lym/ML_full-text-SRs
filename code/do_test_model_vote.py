import pickle

import numpy as np
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, \
    classification_report, accuracy_score

import time
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
    dn = []
    for key, value in vectorizer.vocabulary_.items():
        dn.append([value, key])
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

def get_metric(y_true, y_pred, ):
    Accuracy_ = accuracy_score(y_true, y_pred)
    Recall_ = recall_score(y_true, y_pred)
    Precision_ = precision_score(y_true, y_pred)
    F1_ = f1_score(y_true, y_pred)
    #
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(tn, fp, fn, tp )

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    cr = classification_report(y_true, y_pred, digits=4)

    return Accuracy_, Recall_, Precision_, F1_, sensitivity, specificity, cr


def do_test(X_test, y_test,model, y_test_all):
    for num in [0, 1, 2, 3, 4]: # 5 folds
        f = open('saved_model/%s/%s_fold_%d.pickle' % (model,model,num), 'rb')
        clf = pickle.load(f)
        f.close()

        y_test_pred = clf.predict(X_test)
        y_test_prob = clf.predict_proba(X_test)[:, 1]
        y_test_all.append(y_test_pred)

    return y_test_all, y_test_prob


if __name__ == '__main__':
    file_paths = 'data/'  # 全文数据

    print('loading test tfidf features......')
    test_tfidf, test_y = get_data_feature(file_paths + '/test.txt')

    model = ['GBDT','LR','nbayes','NN','RF','SVM']

    import datetime
    start_time = datetime.datetime.now()
    print(start_time)
    vote_num = 3
    for m in model:
        y_test_all = []
        print('doing test......')
        y_test_all, y_test_pred = do_test(test_tfidf.toarray(),
                np.array(test_y),
                m,
                y_test_all)
        end_time = datetime.datetime.now()
        splee_time = end_time-start_time
        end_time = datetime.datetime.now()
        print(end_time)
        print("耗时: {}秒".format(end_time - start_time))

        y_test_concat = []
        for i in range(len(y_test_pred)):
            a = 0
            if y_test_all[0][i] == 1:
                a += 1
            if y_test_all[1][i] == 1:
                a += 1
            if y_test_all[2][i] == 1:
                a += 1
            if y_test_all[3][i] == 1:
                a += 1
            if y_test_all[4][i] == 1:
                a += 1
            if a >= vote_num:
                y_test_concat.append(1)
            else:
                y_test_concat.append(0)
        Accuracy_test, Recall_test, Precision_test, F1_test, sen_test, spec_test, cr \
            = get_metric(np.array(test_y), y_test_concat)
        #
        print('Accuracy,  Recall, Precision, F1, Sensitivity, Specificity')
        print(str([Accuracy_test, Recall_test, Precision_test, F1_test, sen_test, spec_test]) + '\n')
        outfile = open(f'results/model_vote/{m}_vote_{vote_num}_test.txt', 'w+', encoding='utf-8')
        score_name = ['Accuracy', 'Recall', 'Precision', 'F1', 'Sensitivity', 'Specificity']
        scores_ing = [Accuracy_test, Recall_test, Precision_test, F1_test, sen_test, spec_test]
        for a, b in zip(score_name, scores_ing):
            outfile.writelines(a + ': ' + str(b) + '\n')
        outfile.writelines('\n' + str(cr) + '\n')
        outfile.close()
