# !pip install simpletransformers

import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import numpy as np
from functools import partial
import sklearn 
from scipy import stats
pd.set_option('display.max_colwidth', None)

# Set the columns of dataset
topics_col = 'label'
text_col = 'text'

# Randomly split data, then load data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Training
output_name = 'outputs/roberta'
model = MultiLabelClassificationModel('roberta', 'roberta-base',
                                      use_cuda = True, 
                                      cuda_device = 0,
                                      args={'learning_rate': 3e-5, 
                                            'num_train_epochs': 3,
                                            'max_seq_length': 512,
                                            'overwrite_output_dir': True,
                                            'output_dir': output_name +'/',
                                            "n_gpu": 1,
                                            'reprocess_input_data': True})

model.train_model(train_df)

# Evaluation
eval_results, model_outputs, wrong_predictions = model.eval_model(test_df, verbose=True)


with open(output_name + '/fullreport.txt','w') as f:
        f.write(str(eval_results))

