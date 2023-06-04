import csv
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,f1_score
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import numpy as np
from datasets import load_dataset
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"




data_path = '../../../dataset/input_framing.csv'
output_dir='../../../models/paper_rep/framing/'
ds = load_dataset("csv", data_files=data_path)
checkpoint = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 

def file_update(d):
    with open('best_para.txt', 'a') as f:
      f.write(str(d) + '\n')
    f.close()

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


def model_init():
    return RobertaForSequenceClassification.from_pretrained(checkpoint, problem_type= "single_label_classification",num_labels=15)


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


def ray_hp_space(trial):
    return {
        "learning_rate": tune.choice([4e-5,6e-5]),
        "per_device_train_batch_size": tune.choice([32,64,128]),
        "warmup_steps": tune.choice([0,0.3,0.6,0.9]),
    }


def my_objective(metrics):
    return metrics["eval_accuracy"]


def model_selection(n_1,n_2):
    kf_1 = KFold(n_splits=n_1, random_state=42, shuffle=True)
    kf_2 = KFold(n_splits=n_2, random_state=42, shuffle=True)

    for train_idx, test_idx in kf_1.split(ds['train']):
        train_set = ds['train'].select(train_idx)
       
        for train_idx, val_idx in kf_2.split(train_set):
            train = train_set.select(train_idx)
            val = train_set.select(val_idx)
    
            # tokenize and format input data
            cols = ds["train"].column_names
            cols.remove("labels")
            ds_train = train.map(tokenization, batched=True, remove_columns=cols)
            ds_val = val.map(tokenization, batched=True, remove_columns=cols)
            ds_train.set_format("torch")
            ds_val.set_format("torch")
            
            #model = RobertaForSequenceClassification.from_pretrained(checkpoint, problem_type= "single_label_classification", num_labels=32)
            
            args = TrainingArguments(
                    output_dir=output_dir,
                    evaluation_strategy="steps",
                    eval_steps=500,
                    per_device_train_batch_size=64,
                    per_device_eval_batch_size=64,
                    num_train_epochs=100,
                    gradient_accumulation_steps=1,
                    weight_decay = 5e-7,
                    adam_epsilon= 1e-8,
                    max_grad_norm=1.0,
                    seed=42,
                    # gradient_accumulation_steps=32,
                    # gradient_checkpointing=True,
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

            best_trial = trainer.hyperparameter_search(
                            direction="maximize",
                            backend="ray",
                            hp_space=ray_hp_space,
                            n_trials=5,
                            compute_objective=my_objective,
                            resources_per_trial={"cpu": 20, "gpu": 1}
                        )

            file_update(best_trial.hyperparameters)
            for n, v in best_trial.hyperparameters.items():
                setattr(trainer.args, n, v)
            

            print('best trial is below: ')
            print(best_trial)
            trainer.train()
            trainer.save_model(output_dir + 'best_model/bm_1')
            
            break
        break




def main():
    n_1= 10
    n_2= 9
    model_selection(n_1,n_2)

if __name__ == "__main__":
    main()