import numpy as np
import torch
import nltk
import re 
import string
import json
import random
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def preprocess(data, select_long = True, length_threshold = 0):
    """ this function integrates the preprocessing steps of raw dataset """
    # split into sentences
    data = [preprocess_text(text, select_long=select_long) for text in data]
    # filter out any potential markers (. ! ?) for sentence ending at the front and end of the sentences
    data = [line.strip('"').strip('“').strip('”').strip('.').strip('?').strip('!').strip(' ') for multiline in data for line in multiline]
    # remove duplicates
    data = list(set(data))
    
    # remove the sentences shorter than length threshold
    df = pd.DataFrame({'data':data})
    df['length'] = [len(sent.split(' ')) for sent in df.data]
    df.drop(df[df['length'] <= length_threshold].index, inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    data = df.data.values
    
    print('Got {} sentences'.format(len(data)))

    return data


def preprocess_text(text, select_long=True):
    """ this function removes the website hyperlink, brackets, specific symbols and non-proper short sentences in the raw text, and splits text into sentences """
    url_reg  = r'[a-z]*[:]+\S+'
    text = re.sub('Category','Category ',text)
    
    # remove website hyperlink (http://)
    text   = re.sub(url_reg, '', text)
    
    # sentence tokenization
    sentences = sent_tokenize(text)
        
    output = []
    short_sentence = []
    for sentence in sentences:
        
        # remove '\n'
        sentence = re.sub('\n+', '',sentence)
        
        # remove brackets
        sentence = re.sub("[\(\[].*?[\)\]]", '', sentence)
        
        # remove specific symbols
        flag = 0
        for word in sentence.split():
            q = re.match(r'[a-zA-Z][^a-zA-Z0-9]+', word)
            p = re.match(r'[^a-zA-Z0-9]+', word)
            if q or p:
                flag = 1
        
        if flag:
            continue
        
        if 'References' not in sentence and 'Category' not in sentence and ':' not in sentence and sentence !='':
            
            if select_long == False:
                sentence = ' '.join(sentence.split())
                output.append(sentence)
            # if select_long = True: select sentences contatining more than 5 tokens
            elif len(word_tokenize(sentence.translate(str.maketrans('', '', string.punctuation)))) > 5:
                sentence = ' '.join(sentence.split())
                output.append(sentence)
                
    if '' in output:
           output.remove('')    
    return output


def clean_punctuation(text):
    """ this function removes the punctuations in the sentence """
    words = word_tokenize(text)
    for i, word in enumerate(words):
        if word in string.punctuation:
            words[i] = ''
    if '' in words:
        words.remove('')
    return TreebankWordDetokenizer().detokenize(words).lower()

def generate_negative_samples(dataset):
    """ this function generates the negative samples"""
        
    # generate 40% negative samples by cutting sentences, this type of samples correspond to False Sentence Boundary in the report
    neg_samples_random_cut = []
    for i in range(int(len(dataset) * 0.4) // 3):
        random_cut_sent = random_cut(dataset[3*i], dataset[3*i+1], dataset[3*i+2])
        if random_cut_sent != None:
            neg_samples_random_cut += random_cut_sent
        print('\r-------- {}/{} negative samples generated ----------'.format(3*(i+1), len(dataset)), flush=True, end='')
    # remove empty sample
    i = 0
    data_sorted = sorted(neg_samples_random_cut)
    while data_sorted[i] == '':
        i += 1
    neg_samples_random_cut = data_sorted[i:]
    
    # generate 20% negative samples by removing words, this type of samples correspond to Missing Words in the report
    neg_samples_random_missing = []
    for i in range(int(len(dataset) * 0.4), int(len(dataset) * 0.6)):
        if random_missing(dataset[i]) != None:
            neg_samples_random_missing += random_missing(dataset[i])
        print('\r-------- {}/{} negative samples generated ----------'.format(i, len(dataset)), flush=True, end='')

    # generate 20% negative samples by replacing words, this type of samples correspond to False Word Recognition in the report
    neg_samples_random_replace = []
    for i in range(int(len(dataset) * 0.6) // 2, int(len(dataset) * 0.8) // 2):
        if random_replace(dataset[2*i], dataset[2*i+1]) != None:
            neg_samples_random_replace += random_replace(dataset[2*i], dataset[2*i+1])
        print('\r-------- {}/{} negative samples generated ----------'.format(2*i+1, len(dataset)), flush=True, end='')

    # generate 20% negative samples by repeating words, this type of samples correspond to Repeating Words in the report
    neg_samples_random_repeat = []
    for i in range(int(len(dataset) * 0.8), len(dataset)):
        if random_repeat(dataset[i]) != None:
            neg_samples_random_repeat += random_repeat(dataset[i])
        print('\r-------- {}/{} negative samples generated ----------'.format(i+1, len(dataset)), flush=True, end='')
    print('\n')
    
    neg_samples = neg_samples_random_cut + neg_samples_random_missing + neg_samples_random_replace + neg_samples_random_repeat
    neg_labels = len(neg_samples_random_cut)*['random cut'] + len(neg_samples_random_missing)*['random missing'] + \
                 len(neg_samples_random_replace)*['random replace'] + len(neg_samples_random_repeat)*['random repeat']
    
    return neg_samples, neg_labels

def random_missing(sent):
    """ this function randomly removes words in the middle of the sentence, either continously or discretely. """
    sent = nltk.word_tokenize(clean_punctuation(sent))
    # choose the number of missing words (between 1-4)
    if len(sent) <= 1:
        return None
    elif len(sent) <= 3:
        n_missing_words = 1
    else:
        n_missing_words = np.random.randint(2, min(len(sent)-1, 5))
        
    # randomly choose continous or dicrete missing
    continous_missing = (np.random.rand() >= 0.5)
    # remove words
    if continous_missing:
        missing_start = np.random.randint(0, len(sent) - n_missing_words)
        sent = sent[0:missing_start] + sent[missing_start + n_missing_words:]
    else:
        for i in range(n_missing_words):
            sent.pop(np.random.randint(0, len(sent)))
    sent = TreebankWordDetokenizer().detokenize(sent)
    
    return [sent]
    
def random_repeat(sent):
    """ this function randomly repeats words in the middle of the sentence. """
    sent = nltk.word_tokenize(clean_punctuation(sent))
    # choose the number of repeated words (between 1-3)
    if len(sent) <= 1:
        return None
    elif len(sent) <= 3:
        n_repeat_words = 1
    else:
        n_repeat_words = np.random.randint(1, min(len(sent)-1, 4))
    # repeat words
    for i in range(n_repeat_words):
        repeated_words_idx = np.random.randint(0, len(sent))
        sent.insert(repeated_words_idx, sent[repeated_words_idx])
    sent = TreebankWordDetokenizer().detokenize(sent)
    
    return [sent]    
    
def random_replace(sent_1, sent_2):
    """ this function randomly replaces words in the middle of the sentence. (the replacement is done by exchanging between two sentences) """
    sent_1, sent_2 = nltk.word_tokenize(clean_punctuation(sent_1)), nltk.word_tokenize(clean_punctuation(sent_2))
    # choose the number of replaced words (between 1-3)
    if len(sent_1) <= 1 or len(sent_2) <= 1:
        return None
    elif len(sent_1) <= 3 or len(sent_2) <= 3:
        n_replace_words = 1
    else:
        n_replace_words = np.random.randint(1, min(len(sent_1)-1, len(sent_2)-1,  4))
    sent_1_replace_idx = random.sample(range(len(sent_1)), n_replace_words)
    sent_2_replace_idx = random.sample(range(len(sent_2)), n_replace_words)
    
    # exchange the replaced words between two sentences
    for (idx_1, idx_2) in zip(sent_1_replace_idx, sent_2_replace_idx):
        temp = sent_1[idx_1]
        sent_1[idx_1] = sent_2[idx_2]
        sent_2[idx_2] = temp
    sent_1, sent_2 = TreebankWordDetokenizer().detokenize(sent_1), TreebankWordDetokenizer().detokenize(sent_2)
    
    return sent_1, sent_2    
    
def random_cut(sent_1, sent_2, sent_3):
    """ this function randomly cuts 3 sentences into 2~4 sentences """
    sent_1, sent_2, sent_3 = nltk.word_tokenize(sent_1), nltk.word_tokenize(sent_2), nltk.word_tokenize(sent_3)
    real_cut = {len(sent_1), len(sent_1) + len(sent_2)} # the real sentence splitting places
    joined_sentence = sent_1 + sent_2 + sent_3
    
    # randomly choose the number of cuts
    num_cut = np.random.randint(1, 4)
    
    # for num_cut == 1, find a cutting place in the middle of the sentences (not too near the beginning or the end)
    if num_cut == 1:
        for i in range(10): 
            cut_point = np.random.randint(len(sent_1)//2, (len(sent_1) + len(sent_2) + len(sent_3)//2))
            cut_point = [cut_point]
            
            # check whether the cuts are suitable cuts (the definition of suitable cuts are explained in the 'check_cut' function)
            try:
                check_cut(joined_sentence, cut_point)
            except:
                continue

            if (set(cut_point) & real_cut == set()) and check_cut(joined_sentence, cut_point):
                break
            # try for at most 10 times to find suitable cutting places, if not return None (sometimes there are no suitable places to cut the sentences)
            if i == 9:
                return None

    # for num_cut == 2, find one cutting place near the beginning of the second sentence, one near the end of the second sentence
    if num_cut == 2:
        for i in range(10):
            cut_point_1 = np.random.randint(len(sent_1)//2, (len(sent_1) + len(sent_2)//2))
            cut_point_2 = np.random.randint((len(sent_1) + len(sent_2)//2), (len(sent_1) + len(sent_2) + len(sent_3)//2))
            cut_point = [cut_point_1, cut_point_2]

            try:
                check_cut(joined_sentence, cut_point)
            except:
                continue

            if (set(cut_point) & real_cut == set()) and check_cut(joined_sentence, cut_point):
                break
            if i == 9:
                return None
    
    # for num_cut == 3, find one cut near the beginning of the second sentence, one cut near the middle, one cut near the beginning of the third sentence
    if num_cut == 3:
        for i in range(10):
            cut_point_1 = np.random.randint(max(len(sent_1)//2, 1), len(sent_1)+2)
            cut_point_2 = np.random.randint(max(len(sent_1)-2, 1), \
                                            min(len(sent_1) + len(sent_2)+2, len(sent_1) + len(sent_2) + len(sent_3) - 1))
            cut_point_3 = np.random.randint(max(len(sent_1) + len(sent_2)-2, 1), \
                                            min(len(sent_1) + len(sent_2) + len(sent_3)//2+2, len(sent_1) + len(sent_2) + len(sent_3) - 1))
            cut_point = [cut_point_1, cut_point_2, cut_point_3]

            try:
                check_cut(joined_sentence, cut_point)
            except:
                continue

            if (set(cut_point) & real_cut == set()) and check_cut(joined_sentence, cut_point):
                break
            if i == 9:
                return None
    
    cut_point.insert(0, 0)
    cut_point.insert(len(cut_point), len(joined_sentence))
    
    # cut the sentences using the generated cutting places
    cutted_sentence = [clean_punctuation(' '.join(joined_sentence[cut_point[i]:cut_point[i+1]])) for i in range(len(cut_point)-1)]
    
    return cutted_sentence


def check_cut(joined_sentence, cut_point):
    """ this function checks whether proposed cut points are 'suitable' cut points (not near punctuations or typical words for starting a sentence)"""
    
    tokens_aftercut = set([joined_sentence[cut+1] for cut in cut_point])
    tokens_beforecut = set([joined_sentence[cut] for cut in cut_point])
    
    # the cut is not a suitable cut if followed immediately by the following words or punctuations
    if (tokens_aftercut & ({'and', 'that', 'with', 'which', 'where', 'who', 
                          'whose', 'whom', 'when', 'because', 'but', 'for',
                          'then', 'during', 'except', 'if', 'or', 'after'} | set(string.punctuation)) == set()) \
    and (tokens_beforecut & ({'this', 'that'} | set(string.punctuation)) == set()):  # the cut is not a suitable cut if it is folloing the following words or punctuations

        return True
    else:
        return False

def pad_to_max_length(embedding, max_length=100):
    """ this function pads the embedding (batch_size, sequence_length, embedding_length) to maximal length (in dim=1)"""
    if embedding.size(1) < max_length:
        embedding = torch.cat([embedding, torch.zeros((embedding.size(0), max_length-embedding.size(1), embedding.size(2)), dtype=int)], dim=1)
    
    return embedding