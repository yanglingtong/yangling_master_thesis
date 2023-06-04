from numpy import ogrid
import pandas as pd
import random
from random import shuffle
import torch
import fairseq
import concurrent.futures
import nltk
import math
from nltk.corpus import wordnet 
random.seed(1)
random_state=1


#nltk.download('wordnet')
#nltk.download('omw-1.4')



############################################################################### back translation
# Load the back-translation models



en_de_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt', tokenizer='moses', bpe='fastbpe')
de_en_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt', tokenizer='moses', bpe='fastbpe')
if str(en_de_model.device) == 'cpu' or str(de_en_model.device) == 'cpu':
  # Move the model to a GPU device
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  en_de_model.to(device)
  de_en_model.to(device)
  print(f"Model moved to device: {device}")
else:
  print(f"Model is already on device: {en_de_model.device}")
  print(f"Model is already on device: {de_en_model.device}")
# Define batch size and number of parallel workers
batch_size = 128


def convert_batch(sentences):
    input_tokens = [en_de_model.encode(sentence) for sentence in sentences]
    # Group input sentences into batches
    batches = [input_tokens[i:i+batch_size] for i in range(0, len(input_tokens), batch_size)]
    return batches

# Function to perform back-translation on a batch of sentences
def back_translate(batch,max_len= 256,nbest =4):
    # Generate the translations in the target language
    german_batch = en_de_model.generate(batch,max_len_b = max_len,beam = nbest, nbest=nbest, mask_type='user_defined',skip_invalid_size_inputs=True)
    # Decode the translations in the target language
    german_sentences = [en_de_model.decode(t[i]['tokens']) for t in german_batch for i in range(len(t))]

    # Encode the translations in the target language
    german_tokens = [de_en_model.encode(s) for s in german_sentences]

    # Generate the back-translations in the source language
    english_back = []
    for tokens in german_tokens:
        english_batch = de_en_model.generate(tokens, max_len_b = max_len,beam = nbest,nbest=nbest, mask_type='user_defined',skip_invalid_size_inputs=True)
        english_sentences = [de_en_model.decode(t['tokens']) for t in english_batch]
        english_back.extend(english_sentences)

    return english_back

# Perform back-translation and store the translations

def back_translate_para(batches,max_len = 256,nbest = 4):
    translations = []
    for batch in batches:
        batch_translations = back_translate(batch,max_len,nbest)
        translations.extend(batch_translations)
    return translations


print('translation model loaded')
################################################################################# random swap
# try different n

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

################################################################################## word replacement 


random.seed(1)
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']


      
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
  synonyms = set()
  for syn in wordnet.synsets(word): 
    for l in syn.lemmas():
      #print(l)
      synonym = l.name().replace("_", " ").replace("-", " ").lower()
      synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
      synonyms.add(synonym)
  if word in synonyms:
    synonyms.remove(word)
  return list(synonyms)

# hh = ['History is on the line for both sides','the best footballer in the world']
# gg = ['History', 'is', 'on', 'the', 'line', 'for', 'both', 'sides']
#print(synonym_replacement(gg, 3))
#import torch
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device)
# need to split the text first 


################################################################ how to use the function
#df = pd.read_csv(data_path)[['text','labels']]

def backT_aug(df,max_len = 256,nbest=4):
  df_text = df['text']
  df_labels = df[['labels']]
  original_texts = df_text.tolist()
  original_texts = list(map(lambda x: x[:max_len], original_texts))
  batches= convert_batch(original_texts)
  aug_text_t = back_translate_para(batches,max_len,nbest)

  df_labels = df_labels.loc[df_labels.index.repeat(nbest*nbest)]
  df_labels['text'] = aug_text_t
  df_labels = df_labels.reset_index(drop=True)
	#new = pd.concat([df, df_labels], axis=0)
	#new = df_all.drop_duplicates()
	#new.to_csv("test.csv", index=False)
  return df_labels 



def swap_aug(df,r):
	s_text_list = []
	df_text = df['text']
	df_labels = df[['labels']]
	original_texts = df_text.tolist()
	for i in original_texts:
		i = i.split()
		n = math.ceil(len(i) * r)
		i = random_swap(i, n)
		aug_text_s = ' '.join(i)
		
		s_text_list.append(aug_text_s)

	df_labels['text'] = s_text_list
	
	#new = pd.concat([df, df_labels], axis=0)
	#new = df_labels.drop_duplicates()
	return df_labels
	#new.to_csv("test.csv", index=False)


def synonym_aug(df,r):
    s_text_list = []
    df_text = df['text']
    df_labels = df[['labels']]
    original_texts = df_text.tolist()
    for i in original_texts:
        i = i.split()
        n =  math.ceil(len(i) * r)
        i = synonym_replacement(i, n)
        aug_text_s = ' '.join(i)
     
        s_text_list.append(aug_text_s)
    df_labels['text'] = s_text_list
	#new = pd.concat([df, df_labels], axis=0)
	#new = df_all.drop_duplicates(
    return df_labels 
	#new.to_csv("test.csv", index=False)



# data = {'text': ['I want to go home.', 'I love my mom', 'where is your dad', 'let"s go to Beijing'],'labels': ['0', '1', '2', '3']}
# df = pd.DataFrame.from_dict(data)
# swap_aug(df,4)

################################################################################
# back translation flexible 
def balance_backT_flexible(df,max_len = 256,nbest= 4, aug_label_ratio=0.7, aug_to = 0.3):
    """ aug_to has to be smaller or equal to 1-aug_label_ratio """
    df_aug = pd.DataFrame([])
    label_dict = dict(df['labels'].value_counts())
    label_list =  list(label_dict.keys())
    label_aug_length = round(len(label_list)*aug_label_ratio) 
    number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)]]
    aug_label_list = label_list[-label_aug_length:]
    first_label_list =  label_list[round(len(label_list)*aug_to)]
    print(label_list)
    print("augment label list:")
    print(aug_label_list)

          
    if first_label_list == label_list[-1] or first_label_list == aug_label_list[1]:
        number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)-1]]
        print("augment to label " , label_list[round(len(label_list)*aug_to)-1])     
    else:
        print("augment to label " , label_list[round(len(label_list)*aug_to)])
    
    print("augment number to ",number_aug_to)


    for label in aug_label_list:
        df_small = pd.DataFrame([])
        df_label = df[df['labels'] == label]
        length_small = len(df_label)
        while len(df_small) < number_aug_to:      
            df_new = backT_aug(df_label,max_len,nbest)
            df_small = pd.concat([df_small,df_new],axis=0,ignore_index=True)
            df_small.drop_duplicates()
            df_small = df_small[~df_small.text.isin(df_label.text)]
        df_new = df_small.sample(n=number_aug_to-length_small,random_state=random_state)
        df_aug = pd.concat([df_new,df_aug],axis=0,ignore_index=True)


    return df_aug


#################################################################################
def balance_syn_flexible(df,r=0.1,aug_label_ratio=0.7, aug_to = 0.3):
    """ aug_to has to be smaller or equal to 1-aug_label_ratio """
    """ aug_to has to be smaller or equal to 1-aug_label_ratio """
    df_aug = pd.DataFrame([])
    label_dict = dict(df['labels'].value_counts())
    label_list =  list(label_dict.keys())
    label_aug_length = round(len(label_list)*aug_label_ratio) 
    number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)]]
    aug_label_list = label_list[-label_aug_length:]
    first_label_list =  label_list[round(len(label_list)*aug_to)]
    print(label_list)
    print("augment label list:")
    print(aug_label_list)

          
    if first_label_list == label_list[-1] or first_label_list == aug_label_list[1]:
        number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)-1]]
        print("augment to label " , label_list[round(len(label_list)*aug_to)-1])     
    else:
        print("augment to label " , label_list[round(len(label_list)*aug_to)])
    
    print("augment number to ",number_aug_to)


    for label in aug_label_list:
        df_small = pd.DataFrame([])
        df_label = df[df['labels'] == label]
        length_small = len(df_label)
        while len(df_small) < number_aug_to:      
            df_new = synonym_aug(df_label,r)
            df_small = pd.concat([df_small,df_new],axis=0,ignore_index=True)
            df_small.drop_duplicates()
            df_small = df_small[~df_small.text.isin(df_label.text)]
        df_new = df_small.sample(n=number_aug_to-length_small,random_state=random_state)
        df_aug = pd.concat([df_new,df_aug],axis=0,ignore_index=True)


    return df_aug
#################################################################################
# aug random centences, n_sen = 16

def balance_swap(df,r=0.1,n_sen=16):
    df_aug = pd.DataFrame([])
    label_list = list(df['labels'].unique())
    most = int(df.labels.mode())
    label_list.remove(most)

    for label in label_list:
        df_label = df[df['labels'] == label]
        if sum(df['labels'] == label) >= 16:
            df_random = df_label.sample(n=n_sen,random_state=random_state)
            df_new = swap_aug(df_random,r)
            
            
        else:
            df_small = pd.DataFrame([])
            while len(df_small) < 16:      
                df_new = swap_aug(df_label,r)
                df_small = pd.concat([df_small,df_new],axis=0,ignore_index=True)
                df_small.drop_duplicates()
                df_small = df_small[~df_small.text.isin(df_label.text)]
            df_new = df_small.sample(n=16,random_state=random_state)
        df_aug = pd.concat([df_new,df_aug],axis=0,ignore_index=True)
    
        
    return df_aug

# all the same 
def balance_swap_toMost(df,r=0.1):
    df_aug = pd.DataFrame([])
    label_list = list(df['labels'].unique())
    most = int(df.labels.mode())
    label_list.remove(most)
    length = len(df[df['labels'] == most])

    for label in label_list:
        df_small = pd.DataFrame([])
        df_label = df[df['labels'] == label]
        length_small = len(df_label)
        while len(df_small) < length:      
            df_new = swap_aug(df_label,r)
            df_small = pd.concat([df_small,df_new],axis=0,ignore_index=True)
            df_small.drop_duplicates()
            df_small = df_small[~df_small.text.isin(df_label.text)]          
        df_new = df_small.sample(n=length-length_small,random_state=random_state)        
        df_aug = pd.concat([df_new,df_aug],axis=0,ignore_index=True)

    return df_aug

# flexibel balance
# only part of the labels to be balanced
# how much to balance
def balance_swap_flexible(df,r=0.1,aug_label_ratio=0.7, aug_to = 0.3):
    """ aug_to has to be smaller or equal to 1-aug_label_ratio """
    """ aug_to has to be smaller or equal to 1-aug_label_ratio """
    df_aug = pd.DataFrame([])
    label_dict = dict(df['labels'].value_counts())
    label_list =  list(label_dict.keys())
    label_aug_length = round(len(label_list)*aug_label_ratio) 
    number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)]]
    aug_label_list = label_list[-label_aug_length:]
    first_label_list =  label_list[round(len(label_list)*aug_to)]
    print(label_list)
    print("augment label list:")
    print(aug_label_list)

          
    if first_label_list == label_list[-1] or first_label_list == aug_label_list[1]:
        number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)-1]]
        print("augment to label " , label_list[round(len(label_list)*aug_to)-1])     
    else:
        print("augment to label " , label_list[round(len(label_list)*aug_to)])
    
    print("augment number to ",number_aug_to)


    for label in aug_label_list:
        df_small = pd.DataFrame([])
        df_label = df[df['labels'] == label]
        length_small = len(df_label)
        while len(df_small) < number_aug_to:      
            df_new = swap_aug(df_label,r)
            df_small = pd.concat([df_small,df_new],axis=0,ignore_index=True)
            df_small.drop_duplicates()          
        df_new = df_small.sample(n=number_aug_to-length_small,random_state=random_state)
        df_aug = pd.concat([df_new,df_aug],axis=0,ignore_index=True)


    return df_aug

###################################################### all methods combined
def balance_swap_flexible(df,r=0.1,aug_label_ratio=0.7, aug_to = 0.3):
    """ aug_to has to be smaller or equal to 1-aug_label_ratio """
    df_aug = pd.DataFrame([])
    label_dict = dict(df['labels'].value_counts())
    label_list =  list(label_dict.keys())
    label_aug_length = round(len(label_list)*aug_label_ratio) 
    number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)]]
    aug_label_list = label_list[-label_aug_length:]
    first_label_list =  label_list[round(len(label_list)*aug_to)]
    print(label_list)
    print("augment label list:")
    print(aug_label_list)

          
    if first_label_list == label_list[-1] or first_label_list == aug_label_list[1]:
        number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)-1]]
        print("augment to label " , label_list[round(len(label_list)*aug_to)-1])     
    else:
        print("augment to label " , label_list[round(len(label_list)*aug_to)])
    
    print("augment number to ",number_aug_to)


    for label in aug_label_list:
        df_small = pd.DataFrame([])
        df_label = df[df['labels'] == label]
        length_small = len(df_label)
        while len(df_small) < number_aug_to:      
            df_new = swap_aug(df_label,r)
            df_small = pd.concat([df_small,df_new],axis=0,ignore_index=True)
            df_small.drop_duplicates()          
        df_new = df_small.sample(n=number_aug_to-length_small,random_state=random_state)
        df_aug = pd.concat([df_new,df_aug],axis=0,ignore_index=True)


    return df_aug
############################################################################ all methods
def balance_allMethods_flexible(df,r=0.1,max_len = 256,nbest = 2,aug_label_ratio=0.7, aug_to = 0.3):
    """ aug_to has to be smaller or equal to 1-aug_label_ratio """
    df_aug = pd.DataFrame([])
    label_dict = dict(df['labels'].value_counts())
    label_list =  list(label_dict.keys())
    label_aug_length = round(len(label_list)*aug_label_ratio) 
    number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)]]
    aug_label_list = label_list[-label_aug_length:]
    first_label_list =  label_list[round(len(label_list)*aug_to)]
    print(label_list)
    print("augment label list:")
    print(aug_label_list)

          
    if first_label_list == label_list[-1] or first_label_list == aug_label_list[1]:
        number_aug_to = label_dict[label_list[round(len(label_list)*aug_to)-1]]
        print("augment to label " , label_list[round(len(label_list)*aug_to)-1])     
    else:
        print("augment to label " , label_list[round(len(label_list)*aug_to)])
    
    print("augment number to ",number_aug_to)


    for label in aug_label_list:
        df_small = pd.DataFrame([])
        df_label = df[df['labels'] == label]
        length_small = len(df_label)
        length_aug_needed = number_aug_to-length_small
        while len(df_small) < length_aug_needed:
            df_new_bt = backT_aug(df_label,max_len = max_len,nbest=nbest)
            df_new_bt["method"] = 'bt'
            
            df_new_swap = swap_aug(df_label,r)
            df_new_swap["method"] = 'swap'
            
                    
            df_new_syn = synonym_aug(df_label,r)
            df_new_syn["method"] = 'syn'
            
            df_small = pd.concat([df_small,df_new_swap, df_new_syn, df_new_bt],axis=0,ignore_index=True)
            df_small.drop_duplicates()
            df_small = df_small[~df_small.text.isin(df_label.text)]
            df_small = df_small.groupby('method').apply(lambda x: x.sample(min(len(x),math.ceil(length_aug_needed/3)),random_state=random_state)).reset_index(drop=True)
        
        df_new = df_small.sample(n=length_aug_needed,random_state=random_state)
        df_aug = pd.concat([df_new,df_aug],axis=0,ignore_index=True)


    return df_aug
