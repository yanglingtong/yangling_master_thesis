import csv
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,f1_score
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import numpy as np
from datasets import load_dataset,concatenate_datasets, Dataset
from transformers import RobertaTokenizerFast, Trainer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import RobertaForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import Trainer
from transformers import TrainingArguments
from statistics import mean
import torch 
import matplotlib.pyplot as plt
import itertools
from ray import tune
import os
from transformers import set_seed
set_seed(42)
from transformers import logging
logging.set_verbosity_warning()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ast



data_path = '../../../dataset/input_framing.csv'
ds = load_dataset("csv", data_files=data_path)
checkpoint = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenization(example):
    return tokenizer(example['text'], truncation=True, max_length = 256, padding = True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def plot_confusion_matrix(cm,m, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    
    plt.figure(figsize=(8*1.5, 6*1.5))  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_'+ m+ '.png')









def get_perfomance(m):
    y_list = []
    pre_list = []
    test_idx_list = []
    kf_1 = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_idx, test_idx in kf_1.split(ds['train']):
        test_idx_list.append(test_idx)
        
    for n in range(1,11):
        model_path =  "../../../models/data_aug/framing/"+m+"/bm_" + str(n)
        model = RobertaForSequenceClassification.from_pretrained(model_path, problem_type= "single_label_classification", num_labels=15).to("cuda")

        test = ds['train'].select(test_idx_list[n-1])
        cols = ds["train"].column_names
        cols.remove("labels")
        ds_test = test.map(tokenization, batched=True, remove_columns=cols)
        ds_test.set_format("torch")
        test_trainer = Trainer(model)
        raw_pred, _, _ = test_trainer.predict(ds_test)
        y_pred = list(np.argmax(raw_pred, axis=1))



        # cm = confusion_matrix(test['labels'], y_pred)
        # plot_confusion_matrix(cm, classes = unique_labels(y_list), title = "NB Classifier on Test Set")
        # print(classification_report(test['labels'], y_pred))

        y_list = y_list + test['labels']
        pre_list = pre_list + y_pred
            



    cm = confusion_matrix(y_list, pre_list)
    plot_confusion_matrix(cm,m, classes = unique_labels(y_list), title = "NB Classifier on Test Set")
    print("The result is below " + m)
    print(classification_report(y_list, pre_list,digits=3))


def get_all():
    all_model_path = ["swap_0.3_0.7","syn_0.7_0.3","bt_0.3_0.7","all_0.5_0.5"]
    #all_model_path = ["all_0.5_0.5"]
    for m in all_model_path:
        get_perfomance(m)

get_all()
