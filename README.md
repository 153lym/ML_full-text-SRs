# ML_full-text-SRs

# 1. PDF to txt
reference code [pdf2txt.py](code/pdf2txt.py)

# 2. random split train and test data
load label data, and pdf2txt data (they are same size) \
reference code [random_split_data.py](code/random_split_data.py)

# 3. do train
reference code [ML-train.py](code/ML-train.py) \
model: ['GBDT','LR','nbayes','NN','RF','SVM']

# 4. do test, use vote
reference code [do_test_model_vote.py](code/do_test_model_vote.py)[ML-train.py](code/ML-train.py)

# 5. do predeciton
reference code [do_prediction.py](code/do_prediction.py) \
can do prediction and get the feature importances