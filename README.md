# yangling_master_thesis

# Introduction

Textual data has been widely recognized as a crucial source in political science research
for examining party stances, media frames of political matters, and general
news topics. The application of Natural Language Processing (NLP) is receiving
significant attention in political science due to its ability to conduct large-scale
analysis and offer a more neutral evaluation. However, the limited, sparse, and
unbalanced data frequently challenges the development of high-performance classification
models for political text analysis. Therefore, this thesis aims to investigate
approaches to adding new data with augmentation techniques to improve the
transformer-based model understanding ability in political science. Random swap,
synonym replacement, back-translation, and all combined techniques are used as
data augmentation methods for three tasks: policy preferences detection, frame
categorization, and topic classification. In the paper, a novel technique for adding
additional data is presented. The strategy used in this thesis balances the dataset to
a pre-determined level by only adding new data to rare classes instead of adding the
same amount of instances to each label as most other studies do. The result demonstrates
slight improvements in three classification tasks, with accuracy gains for the
complete datasets of 1.0%, 0.5%, and 0.3%. Some rare classes show a reduction
in recall. The reason could be that the model overfitted the training data because
the augmentation methods might have changed the training data’s distributions and
aggressively altered the sentence’s identity.


# How to use the codes
You can find datasets in the "data" folder, codes in the "codes" folder, and the results in "result" folder.
