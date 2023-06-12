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
import sys
sys.path.insert(1, '../')
from data_aug_techFunctions import *




data_path = '../../../dataset/input_framing.csv' 
best_para_path = '../../paper_replication/framing/best_para.txt'
ds_1 = load_dataset("csv", data_files=data_path)
cols = ds_1["train"].column_names
cols.remove("text")
cols.remove("labels")
ds_1 = ds_1.map(remove_columns=cols)
checkpoint = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


with open(best_para_path) as f:
    best_para_list = f.readlines()
best_para = ast.literal_eval(best_para_list[0])


def file_update(d):
    with open('best_para.txt', 'a') as f:
      f.write(str(d) + '\n')
    f.close()

def tokenization(example):
    return tokenizer(example['text'], truncation=True, max_length = 256, padding = True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def model_init():
    return RobertaForSequenceClassification.from_pretrained(checkpoint, problem_type= "single_label_classification",num_labels=15).to('cuda')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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




def my_objective(metrics):
    return metrics["eval_accuracy"]



def model_selection( output_dir,aug_label_ratio, aug_to):
    kf_1 = KFold(n_splits=10, random_state=42, shuffle=True)
    kf_2 = KFold(n_splits=9, random_state=42, shuffle=True)
    nCross = 0
    for train_idx, test_idx in kf_1.split(ds_1['train']):
        nCross = nCross + 1
        #if nCross >= 6:
        train_set_1 = ds_1['train'].select(train_idx)

        for train_idx, val_idx in kf_2.split(train_set_1):
            val = train_set_1.select(val_idx)
                 
            train_1 = train_set_1.select(train_idx)
            train_1 = pd.DataFrame(train_1)
            train_2 =  balance_swap_flexible(train_1,r=0.1,aug_label_ratio=aug_label_ratio, aug_to = aug_to)
            train = pd.concat([train_1,train_2],axis=0,ignore_index=True)
            #train = train.drop_duplicates()
            train = Dataset.from_pandas(train)
          
            # tokenize and format input data
            ds_train = train.map(tokenization, batched=True)
            ds_val = val.map(tokenization, batched=True)
            ds_train.set_format("torch")
            ds_val.set_format("torch")

            args = TrainingArguments(
                    output_dir=output_dir,
                    evaluation_strategy="steps",
                    eval_steps=500,
                    learning_rate = best_para['learning_rate'],
                    per_device_train_batch_size=best_para['per_device_train_batch_size'],
                    warmup_steps = best_para['warmup_steps'],
                    per_device_eval_batch_size=64,
                    num_train_epochs=100,
                    gradient_accumulation_steps=1,
                    weight_decay = 5e-7,
                    adam_epsilon= 1e-8,
                    max_grad_norm=1.0,
                    seed=42,
                    #gradient_checkpointing=True,
                    fp16=True,
                    load_best_model_at_end=True,
                    metric_for_best_model = 'eval_accuracy',
                    )



            trainer = Trainer(
                    args=args,
                    compute_metrics=compute_metrics,
                    train_dataset=ds_train, 
                    eval_dataset=ds_val,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
                    model_init=model_init
                )
    
    
            print(str(nCross) + ' Cross Validation')
            trainer.train()
            trainer.save_model(output_dir + 'bm_' + str(nCross))
            break



def main():
    aug_label_ratio_list = [0.1,0.3,0.5,0.7,0.9]
    for aug_label_ratio in aug_label_ratio_list:
        folder_path = 'swap_' + str(aug_label_ratio) + '_' + str(round(1-aug_label_ratio,1)) + '/'
        output_dir='../../../models/data_aug/framing/' + folder_path
        model_selection(output_dir, aug_label_ratio, aug_to= 1-aug_label_ratio)
    


if __name__ == "__main__":
    main()

