# %%
import transformers
from transformers import AdamW, RobertaConfig, RobertaForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import AutoTokenizer

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import ConcatDataset
from torch import nn
from torch.nn import functional as F

from pytorch_pretrained_bert import BertModel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import pandas as pd
import numpy as np
import time, datetime, random, glob, os, sys, joblib, argparse, json
from tqdm import tqdm
from collections import Counter
from statistics import mean
import numpy as np
from copy import deepcopy



# %%
import os, sys, re, string

# %%
# ==== text process ====
def strip_all_entities(text):
    # ==== error 1 ====: isn't -> isn t
    # entity_prefixes = ['@']
    # for separator in string.punctuation:
    #     if separator not in entity_prefixes:
    #         text = text.replace(separator, ' ')
    # words = []
    # for word in text.split():
    #     word = word.strip()
    #     if word:
    #         if word[0] not in entity_prefixes:
    #             words.append(word)
    # return ' '.join(words)
    user_mention_patter = r"(?:\@|https?\://)\S+"
    tweet_text = re.sub(user_mention_patter, "", text)
    return tweet_text

# it fails on this tweet: call @Susan @My 5g via @full @tre
def remove_leading_usernames(tweet):
    """
        Remove all user handles at the beginning of the tweet.
        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """
    regex_str = '^[\s.]*@[A-Za-z0-9_]+\s+'

    original = tweet
    change = re.sub(regex_str, '', original)

    while original != change:
        original = change
        change = re.sub(regex_str, '', original)

    return change

def process_tweet(tweet):
    """
        Preprocess tweet. Remove URLs, leading user handles, retweet indicators, emojis,
        and unnecessary white space, and remove the pound sign from hashtags. Return preprocessed
        tweet in lowercase.
        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """

    #Remove www.* or https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))\s+','',tweet)
    tweet = re.sub('\s+((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    #Remove RTs
    tweet = re.sub('^RT @[A-Za-z0-9_]+: ', '', tweet)
    # Incorrect apostraphe
    tweet = re.sub(r"’", "'", tweet)
    #Remove @username
    # solution 1 by Caleb:
    # tweet = remove_leading_usernames(tweet)
    # solution 2 by bing with some improvement:
    tweet = strip_all_entities(tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace ampersands
    tweet = re.sub(r' &amp; ', ' and ', tweet)
    tweet = re.sub(r'&amp;', '&', tweet)
    #Remove emojis
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet.lower().strip()

# %%
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# %%
criterion = nn.CrossEntropyLoss()

def model_eval(model, prediction_dataloader):
    model.eval()
    true_labels = []
    predictions = []
    total_loss = 0
    
    for batch in tqdm(prediction_dataloader):
        with torch.no_grad():
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids.to(DEVICE), attention_mask=b_input_mask.to(DEVICE)) # , labels = b_labels.to(DEVICE)

            b_proba = outputs[0]
            loss = criterion(b_proba, b_labels.to(DEVICE))
            total_loss = total_loss + loss.item()
            
            proba = b_proba.detach().cpu().numpy()
            label_ids = b_labels.numpy()
            
            true_labels.append(label_ids)
            predictions.append(proba)
    
    flat_predictions = np.concatenate(predictions, axis=0)
    y_pred = np.argmax(flat_predictions, axis=1).flatten()
    y_true = np.concatenate(true_labels, axis=0)
    res = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    return total_loss / len(prediction_dataloader), res

# %%
if torch.cuda.is_available():
    device = "cuda:1"
    DEVICE = device
    n_gpu = 1

# %%


# %%
df = pd.read_csv('./reply_id_label_release_with_text.csv')

# %%
df.head(1)

# %%
Counter(df['counterreply_label'])

# %%
print(df.columns)
print(df.shape)

# %%


# %%
df['tweet'] = df['tweet'].apply(process_tweet)
df['reply'] = df['reply'].apply(process_tweet)

# %%
tokenizer = AutoTokenizer.from_pretrained("roberta-base", do_lower_case=True) 

# %%
Y_COLUMN = 'counterreply_label'

# %%


# %%
def get_tensor_dataset(df):
    encode = tokenizer(list(df['tweet']), list(df['reply']), 
                            padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    seq = torch.tensor(encode['input_ids'])
    mask = torch.tensor(encode['attention_mask'])
    y = torch.tensor(df[Y_COLUMN].tolist())    
    return TensorDataset(seq, mask, y), y

# %%
# K-fold cross validation with hyperparameter search, train/val/test
from transformers.utils import logging
logging.set_verbosity(40)  # suppress warnings

data, y = get_tensor_dataset(df)

k_folds = 10
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_ids = []
for fold, (nontrain_ids, test_ids) in enumerate(kfold.split(data, y)):
    train_ids, val_ids = train_test_split(nontrain_ids, test_size=1/9,
                                          random_state=42,
                                stratify=[x for i, x in enumerate(y) if i in nontrain_ids])
    train_ids.sort()
    val_ids.sort()
    fold_ids += [(train_ids, val_ids, test_ids)]

results = {}

for batch_size in [6, 8]:
    results[batch_size] = {}
    
    # , 1e-4
    for learning_rate in [ 1e-5]:
        results[batch_size][learning_rate] = {"precision": [], "recall": [], "fscore": []}
        
        print('--------------------------------')
        print("batch_size =", batch_size, " learning_rate =", learning_rate)

        for i, (train_ids, val_ids, test_ids) in enumerate(fold_ids):            
            print('--------------------------------')
            print(f'-------- process {i}-th fold -------')

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            train_dataloader = DataLoader(data, sampler=train_subsampler, batch_size=batch_size)
            val_dataloader = DataLoader(data, sampler=val_subsampler, batch_size=batch_size)
            test_dataloader = DataLoader(data, sampler=test_subsampler, batch_size=batch_size)
            
            model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base', 
                num_labels=2, 
                return_dict=False, # not available in transformer-v2
                output_hidden_states=False
            )


            model.to(device)

            epsilon = 1e-7
            optimizer = AdamW(model.parameters(),
                          lr = learning_rate,
                          eps = epsilon,
                          no_deprecation_warning=True)

            train_loss = []
            val_loss, val_precision, val_recall, val_fscore = [], [], [], []
            test_precision, test_recall, test_fscore = [], [], []
            
            EPOCHS = 5
            for epoch in range(EPOCHS):

                #print(f'==== process epoch: {epoch} ====')
                model.train()

                total_loss = 0

                for step,batch in enumerate(train_dataloader):
                    batch = [r.to(device) for r in batch]
                    # input_id,attention_mask,token_type_id,y = batch
                    input_id,attention_mask, y = batch
                    optimizer.zero_grad()
                    pair_token_ids = input_id.to(device)
                    mask_ids = attention_mask.to(device)
                    # seg_ids = token_type_id.to(device)
                    labels = y.to(device)
                    model.zero_grad()
                    # (_, logits) = model(pair_token_ids,attention_mask = mask_ids, token_type_ids = seg_ids, labels = labels)
                    (_, logits) = model(pair_token_ids,attention_mask = mask_ids, labels = labels)
                    loss = criterion(logits, labels) # applied labels already
                    total_loss = total_loss + loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                train_loss_epoch = total_loss / len(train_dataloader)
                train_loss += [f'{train_loss_epoch:.3f}']
                #print("Average loss =", avg_loss_epoch)
                #print('======== during evaluation ========')

                # Compile validation set statistics.
                val_loss_epoch, res = model_eval(model, val_dataloader)
                val_loss += [f'{val_loss_epoch:.3f}']
                val_precision += [res[0]]
                val_recall += [res[1]]
                val_fscore += [res[2]]
                
                # Compile test set statistics.
                _, res = model_eval(model, test_dataloader)
                test_precision += [res[0]]
                test_recall += [res[1]]
                test_fscore += [res[2]]
            
            # Select best epoch for this fold
            # best_epoch = np.argmin(val_loss) + 1
            # we use the evaluation metrics for performance monitoring
            best_epoch = np.argmax(val_fscore) + 1

            
            # Compile results for this fold
            results[batch_size][learning_rate]["precision"] += [test_precision[best_epoch - 1]]
            results[batch_size][learning_rate]["recall"] += [test_recall[best_epoch - 1]]
            results[batch_size][learning_rate]["fscore"] += [test_fscore[best_epoch - 1]]
            
            print('Training loss       =', ' '.join(train_loss))
            print('Val loss            =', ' '.join(val_loss))
            print('Best epoch (BE)     =', best_epoch)
            # print('Val precision  @ BE =', f'{val_precision[best_epoch - 1]:.3f}')
            # print('Val recall     @ BE =', f'{val_recall[best_epoch - 1]:.3f}')
            # print('Val f1-score   @ BE =', f'{val_fscore[best_epoch - 1]:.3f}')
            # print('Test precision @ BE =', f'{test_precision[best_epoch - 1]:.3f}')
            # print('Test recall    @ BE =', f'{test_recall[best_epoch - 1]:.3f}')
            # print('Test f1-score  @ BE =', f'{test_fscore[best_epoch - 1]:.3f}')
            
        # Summarize this hyperparameter setting
        results[batch_size][learning_rate]["precision"] = mean(results[batch_size][learning_rate]["precision"])
        results[batch_size][learning_rate]["recall"] = mean(results[batch_size][learning_rate]["recall"])
        results[batch_size][learning_rate]["fscore"] = mean(results[batch_size][learning_rate]["fscore"])
        #print("For this setting, the summary results are:")
        print("    precision =", f'{results[batch_size][learning_rate]["precision"]:.3f}')
        print("    recall    =", f'{results[batch_size][learning_rate]["recall"]:.3f}')
        print("    f1-score  =", f'{results[batch_size][learning_rate]["fscore"]:.3f}')

# %% [markdown]
# Train model for specified hyperparameters, save model to file

# %%



