import pickle

import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm


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


def get_pred(file):
    data = []
    with open(file, 'r', encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace('"', '').replace('\n', '')
            text = str(line)
            data.append(text)
    return data

def get_pred_feature(file):
    f = open('saved_model/tfidftransformer.pickle', 'rb')
    tfidftransformer = pickle.load(f)
    f.close()

    f = open('saved_model/vectorizer.pickle', 'rb')
    vectorizer = pickle.load(f)
    f.close()

    data_test = get_pred(file)
    str_list = get_stop_text(data_test)
    # 提取文本特征
    tfidf = tfidftransformer.transform(vectorizer.transform(str_list))

    return tfidf

def do_pred_concat(X_pred, vote_num, model):
    y_pred_all = []

    for num in [0, 1, 2, 3, 4]:
        f = open('saved_model/%s/%s_fold_%d.pickle' % (model, model, num), 'rb')
        clf = pickle.load(f)
        f.close()

        y_pred_pred = clf.predict(X_pred)
        y_pred_all.append(y_pred_pred)

    y_pred_concat = []
    for i in range(len(y_pred_pred)):
        a = 0
        if y_pred_all[0][i] == 1:
            a += 1
        if y_pred_all[1][i] == 1:
            a += 1
        if y_pred_all[2][i] == 1:
            a += 1
        if y_pred_all[3][i] == 1:
            a += 1
        if y_pred_all[4][i] == 1:
            a += 1
        if a >= vote_num:
            y_pred_concat.append(1)
        else:
            y_pred_concat.append(0)

    pd.DataFrame(y_pred_concat).to_excel( f'results/{model}_vote_{vote_num}_full.xlsx')

def do_imp( model, num):
    f = open('saved_model/%s/%s_fold_%d.pickle' % (model, model, num), 'rb')
    clf = pickle.load(f)
    f.close()

    if model in ['LR']:
        im = clf.coef_[0]
    elif model in ['nbayes']:
        im = clf.epsilon_
    else:
        im = clf.feature_importances_
    ind = [i for i in range(len(im))]
    imdict = dict(zip(ind, im))
    dn = []
    for key, value in imdict.items():
        dn.append([value, key])
    pd.DataFrame(dn).to_excel(f'results/{model}_importance.xlsx')

if __name__ == '__main__':
    print('loading pred tfidf features......')
    pred_tfidf = get_pred_feature('data/pred.txt')

    model = ['GBDT','LR','nbayes','NN','RF','SVM','xgboost']
    vote_num = 3
    for m in model:
        print('doing pred......')
        do_pred_concat(pred_tfidf.toarray(), vote_num, m)

    best_fold = [1,0,2,4,3,1]
    for m,n in zip(model,best_fold):
        print('doing importance......')
        do_imp(m,n)