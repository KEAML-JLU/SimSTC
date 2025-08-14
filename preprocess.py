import json
import numpy as np
import pickle as pkl
import math
import nltk
from tqdm import tqdm
import os

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import torch
from nlpaug.util import Action
import json
from tqdm import tqdm

import nlpaug.augmenter.word as naw
import argparse
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import re
import random
def clean_str(string,use=True):
    if not use: return string

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_stopwords(filepath='./stopwords_en.txt'):
    stopwords = set()
    with open(filepath, 'r') as f:
        for line in f:
            swd = line.strip()
            stopwords.add(swd)
    print(len(stopwords))
    return stopwords

def tf_idf_transform(inputs, mapping=None, sparse=False):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from scipy.sparse import coo_matrix
    vectorizer = CountVectorizer(vocabulary=mapping)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(inputs))
    weight = tf_idf.toarray()
    return weight if not sparse else coo_matrix(weight)

def PMI(inputs, mapping, window_size, sparse):
    W_ij = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
    W_i = np.zeros([len(mapping)], dtype=np.float64)
    W_count = 0
    for one in inputs:
        word_list = one.split(' ')
        if len(word_list) - window_size < 0:
            window_num = 1
        else:
            window_num = len(word_list) - window_size + 1
        for i in range(window_num):
            W_count += 1
            context = list(set(word_list[i:i + window_size]))
            while '' in context:
                context.remove('')
            for j in range(len(context)):
                W_i[mapping[context[j]]] += 1
                for k in range(j + 1, len(context)):
                    W_ij[mapping[context[j]], mapping[context[k]]] += 1
                    W_ij[mapping[context[k]], mapping[context[j]]] += 1
    if sparse:
        rows = []
        columns = []
        data = []
        for i in range(len(mapping)):
            rows.append(i)
            columns.append(i)
            data.append(1)
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i]
            for j in tmp:
                value = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if value > 0:
                    rows.append(i)
                    columns.append(j)
                    data.append(value)
                    rows.append(j)
                    columns.append(i)
                    data.append(value)
        PMI_adj = coo_matrix((data, (rows, columns)), shape=(len(mapping), len(mapping)))
    else:
        PMI_adj = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
        for i in range(len(mapping)):
            PMI_adj[i, i] = 1  
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i] 
            # for j in range(i + 1, len(mapping)):
            for j in tmp:
                pmi = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if pmi > 0:
                    PMI_adj[i, j] = pmi
                    PMI_adj[j, i] = pmi
    return PMI_adj

def text_aug(src_file, dataset_name, aug_type):
    #src_file = base_dir + dataset_name + '_Train.json'
    tgt_file = './aug_{}_text/{}_{}_split.json'.format(aug_type, dataset_name, aug_type)#aug_base_dir + dataset_name +'_Train_aug.json'
    
    origin_data = json.load(open(src_file, 'r', encoding='utf-8'))['train']
    origin_data_test = json.load(open(src_file, 'r', encoding='utf-8'))['test']
    
    print('running ', dataset_name, '.......')
    lines_in = [item['text'] for item in origin_data.values()]
    lines_in_test = [item['text'] for item in origin_data_test.values()]
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    batch_size = 128
    
    if aug_type == "context":
        augmenter = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", \
            aug_min=1, aug_p=0.2, device="cuda")
    elif aug_type == "wordnet":
        augmenter = naw.SynonymAug(aug_src='wordnet', aug_p=0.4)
    else:
        augmenter = naw.RandomWordAug()
    
    """
    for i in tqdm(range(0, len(lines_in), batch_size)):
        lines_batch = lines_in[i:i+batch_size]
        for j, p in enumerate(lines_batch):
            aug_text = random.choice(augmenter.augment(data=p, n=10))
            origin_data[str(i+j)]['aug_text'] = aug_text
        del lines_batch
        del aug_text

    for i in tqdm(range(0, len(lines_in_test), batch_size)):
        lines_batch = lines_in_test[i:i+batch_size]
        for j, p in enumerate(lines_batch):
            aug_text = random.choice(augmenter.augment(data=p, n=10))
            origin_data_test[str(i+j)]['aug_text'] = aug_text
        del lines_batch
        del aug_text
    """
    for i in tqdm(range(0, len(lines_in), batch_size)):
        lines_batch = lines_in[i:i+batch_size]
        aug_text = augmenter.augment(data=lines_batch)
        for j, p in enumerate(aug_text):
            origin_data[str(i+j)]['aug_text'] = p
        del lines_batch
        del aug_text

    for i in tqdm(range(0, len(lines_in_test), batch_size)):
        lines_batch = lines_in_test[i:i+batch_size]
        aug_text = augmenter.augment(data=lines_batch)
        for j, p in enumerate(aug_text):
            origin_data_test[str(i+j)]['aug_text'] = p
        del lines_batch
        del aug_text
        
    fin_data = {'train':origin_data, 'test':origin_data_test}
    
    json.dump(fin_data, open(tgt_file, 'w'),
              ensure_ascii=False)
    
    print(dataset_name, 'done!')

def make_node2id_eng_text(dataset_name, augtype, remove_StopWord=False):
    stop_word=load_stopwords()
    stop_word.add('')
    os.makedirs(f'../data/{dataset_name}_data_{augtype}', exist_ok=True)

    f_train = json.load(open('./aug_{}_text/{}_{}_split.json'.format(augtype, dataset_name, augtype)))['train']
    f_test = json.load(open('./aug_{}_text/{}_{}_split.json'.format(augtype, dataset_name, augtype)))['test']

    from collections import defaultdict
    word_freq=defaultdict(int)
    for item in f_train.values():
        words=clean_str(item['text']).split(' ')
        aug_words=clean_str(item['aug_text']).split(' ')
        for one in words:
            word_freq[one.lower()]+=1
        for one in aug_words:
            word_freq[one.lower()]+=1
    for item in f_test.values():
        words=clean_str(item['text']).split(' ')
        aug_words=clean_str(item['aug_text']).split(' ')
        for one in words:
            word_freq[one.lower()]+=1
        for one in aug_words:
            word_freq[one.lower()]+=1
            
    freq_stop=0
    for word,count in word_freq.items():
        if count<5:
            stop_word.add(word)
            freq_stop+=1
    print('freq_stop num',freq_stop)

    ent2id_new = json.load(open('./pretrained_emb/NELL_KG/ent2ids_refined', 'r'))
    adj_ent_index = []
    query_nodes = []
    tag_set = set()
    entity_set = set()
    words_set = set()
    train_idx = []
    test_idx = []
    labels = []
    tag_list = []
    word_list = []
    ent_mapping = {} 
    
    for i, item in enumerate(tqdm(f_train.values())):
        # item=f_train[str(i)]
        query, aug_query = clean_str(item['text']), clean_str(item['aug_text'])
        if not query or not aug_query:
            print(query, aug_query)
            continue
        tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]
        aug_tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(aug_query))]
        if '' in tags or '' in aug_tags:
            print(item)

        tag_list.append(' '.join(tags))
        tag_list.append(' '.join(aug_tags)) #
        tag_set.update(tags, aug_tags) #
        labels.append(item['label'])
        #labels.append(item['label']) #
        if remove_StopWord:
            words = [one.lower() for one in query.split(' ') if one not in stop_word]
            aug_words = [one.lower() for one in aug_query.split(' ') if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(' ')] 
            aug_words = [one.lower() for one in aug_query.split(' ')]
        if '' in words:
            print(words)

        ent_list = []
        index = []
        aug_index = []
        for key in ent2id_new.keys():
            if key in query.lower():
                ent_list.append(key)
                if key not in ent_mapping:
                    ent_mapping[key] = len(ent_mapping)
                    entity_set.update(ent_list)
                if ent_mapping[key] not in index: index.append(ent_mapping[key])
            if key in aug_query.lower():
                ent_list.append(key)
                if key not in ent_mapping:
                    ent_mapping[key] = len(ent_mapping)
                    entity_set.update(ent_list)
                if ent_mapping[key] not in aug_index: aug_index.append(ent_mapping[key])            
        adj_ent_index.append(index)
        adj_ent_index.append(aug_index)
        word_list.append(' '.join(words))
        word_list.append(' '.join(aug_words)) 
        words_set.update(words, aug_words)
        # may have problem
        if query or aug_query:
            query_nodes.append(query)
            query_nodes.append(aug_query)
        else:
            print(item)
            print(query)
            print(aug_query)
        train_idx.append(len(train_idx))
    
    for i, item in enumerate(tqdm(f_test.values())):
        # item = f_test[str(i)]
        query, aug_query = clean_str(item['text']), clean_str(item['aug_text'])
        # print(query)
        if not query or not aug_query:
            print(query)
            print(aug_query) #
            continue
        tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]
        aug_tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(aug_query))] #
        tag_list.append(' '.join(tags))
        tag_list.append(' '.join(aug_tags)) #
        tag_set.update(tags, aug_tags)
        labels.append(item['label'])
        #labels.append(item['label']) # whether to use double label?
        if remove_StopWord:
            words = [one.lower() for one in query.split(' ') if one not in stop_word] 
            aug_words = [one.lower() for one in aug_query.split(' ') if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(' ')] 
            aug_words = [one.lower() for one in aug_query.split(' ')]
        if '' in words:
            print(words)
        ent_list = []
        index = []
        aug_index = []
        for key in ent2id_new.keys():
            if key in query.lower():
                if key not in ent_mapping:
                    ent_list.append(key)
                    ent_mapping[key] = len(ent_mapping)
                    entity_set.update(ent_list)
                if ent_mapping[key] not in index: index.append(ent_mapping[key])

            if key in aug_query.lower():
                if key not in ent_mapping:
                    ent_list.append(key)
                    ent_mapping[key] = len(ent_mapping)
                    entity_set.update(ent_list)
                if ent_mapping[key] not in aug_index: aug_index.append(ent_mapping[key])
        adj_ent_index.append(index)
        adj_ent_index.append(aug_index)

        word_list.append(' '.join(words))
        word_list.append(' '.join(aug_words))
        words_set.update(words, aug_words)
        if query or aug_query:
            query_nodes.append(query)
            query_nodes.append(aug_query)
        else:
            print(item)
            print(query)
            print(aug_query)

        test_idx.append(len(test_idx)+ len(train_idx))
    print('adj_ent length: ', len(adj_ent_index), 'query length: ', len(query_nodes))
    print(tag_set)
    json.dump([adj_ent_index, ent_mapping],
              open('../data/{}_data_{}/index_and_mapping.json'.format(dataset_name, augtype), 'w'), ensure_ascii=False)
    ent_emb = []
    TransE_emb_file = np.loadtxt('./pretrained_emb/NELL_KG/entity2vec.TransE')
    TransE_emb = []

    for i in range(len(TransE_emb_file)):
        TransE_emb.append(list(TransE_emb_file[i, :]))

    rows = []
    data = []
    columns = []

    max_num = len(ent_mapping)
    for i, index in enumerate(adj_ent_index):
        for ind in index:
            data.append(1)
            rows.append(i)
            columns.append(ind)

    adj_ent = coo_matrix((data, (rows, columns)), shape=(len(adj_ent_index), max_num))
    print('query_ent shape: ', adj_ent.shape)
    for key in ent_mapping.keys():
        ent_emb.append(TransE_emb[ent2id_new[key]])

    ent_emb = np.array(ent_emb)
    print('ent shape', ent_emb.shape)
    ent_emb_normed = ent_emb / np.sqrt(np.square(ent_emb).sum(-1, keepdims=True))
    adj_emb = np.matmul(ent_emb_normed, ent_emb_normed.transpose())
    print('entity_emb_cos', np.mean(np.mean(adj_emb, -1)))
    pkl.dump(np.array(ent_emb), open('../data/{}_data_{}/entity_emb.pkl'.format(dataset_name, augtype), 'wb'))
    pkl.dump(adj_ent, open('../data/{}_data_{}/adj_query2entity.pkl'.format(dataset_name, augtype), 'wb'))

    word_nodes = list(words_set)
    tag_nodes = list(tag_set)
    entity_nodes = list(entity_set)
    # nodes_all = list(query_nodes | tag_nodes | entity_nodes)
    nodes_all = query_nodes + tag_nodes + entity_nodes + word_nodes
    nodes_num = len(query_nodes) + len(tag_nodes) + len(entity_nodes) + len(word_nodes)
    print('query', len(query_nodes))
    print('tag', len(tag_nodes))
    print('ent', len(entity_nodes))
    print('word', len(word_nodes))

    if len(nodes_all) != nodes_num:
        print('duplicate name error')

    print('len_train',len(train_idx))
    print('len_test',len(test_idx))
    print('len_quries',len(query_nodes))

    tags_mapping = {key: value for value, key in enumerate(tag_nodes)}
    words_mapping = {key: value for value, key in enumerate(word_nodes)}
    adj_query2tag = tf_idf_transform(tag_list, tags_mapping)
    print('query_tag shape: ', adj_query2tag.shape)
    adj_tag = PMI(tag_list, tags_mapping, window_size=5, sparse=False)
    pkl.dump(adj_query2tag, open('../data/{}_data_{}/adj_query2tag.pkl'.format(dataset_name, augtype), 'wb'))
    pkl.dump(adj_tag, open('../data/{}_data_{}/adj_tag.pkl'.format(dataset_name, augtype), 'wb'))
    adj_query2word = tf_idf_transform(word_list, words_mapping, sparse=True)
    print('query_word shape: ', adj_query2word.shape)
    adj_word = PMI(word_list, words_mapping, window_size=5, sparse=True)
    pkl.dump(adj_query2word, open('../data/{}_data_{}/adj_query2word.pkl'.format(dataset_name, augtype), 'wb'))
    pkl.dump(adj_word, open('../data/{}_data_{}/adj_word.pkl'.format(dataset_name, augtype), 'wb'))
    json.dump(train_idx, open('../data/{}_data_{}/train_idx.json'.format(dataset_name, augtype), 'w'), ensure_ascii=False)
    json.dump(test_idx, open('../data/{}_data_{}/test_idx.json'.format(dataset_name, augtype), 'w'), ensure_ascii=False)

    label_map = {value: i for i, value in enumerate(set(labels))}
    json.dump([label_map[label] for label in labels], open('../data/{}_data_{}/labels.json'.format(dataset_name, augtype), 'w'),
              ensure_ascii=False)
    json.dump(query_nodes, open('../data/{}_data_{}/query_id2_list.json'.format(dataset_name, augtype), 'w'),
              ensure_ascii=False)
    json.dump(tag_nodes, open('../data/{}_data_{}/tag_id2_list.json'.format(dataset_name, augtype), 'w'), ensure_ascii=False)
    json.dump(entity_nodes, open('../data/{}_data_{}/entity_id2_list.json'.format(dataset_name, augtype), 'w'),
              ensure_ascii=False)
    json.dump(word_nodes, open('../data/{}_data_{}/word_id2_list.json'.format(dataset_name, augtype), 'w'), ensure_ascii=False)

    glove_emb = pkl.load(open('./pretrained_emb/old_glove_6B/embedding_glove.p', 'rb'))
    vocab = pkl.load(open('./pretrained_emb/old_glove_6B/vocab.pkl', 'rb'))
    embs = []
    err_count = 0
    for word in word_nodes:
        if word in vocab:
            embs.append(glove_emb[vocab[word]])
        else:
            err_count += 1
            # print('error:', word)
            embs.append(np.zeros(300, dtype=np.float64))
    print('err in word count', err_count)
    pkl.dump(np.array(embs, dtype=np.float64), open('../data/{}_data_{}/word_emb.pkl'.format(dataset_name, augtype), 'wb'))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=True, default="snippets", type=str)
parser.add_argument('--aug_type', default="context", type=str)
args = parser.parse_args()
dataset_name = args.dataset_name.strip()
aug_type = args.aug_type.strip()

if dataset_name in ['mr', 'snippets', 'TagMyNews', 'ohsumed']:
    remove_StopWord = True
else:
    remove_StopWord = False
ori_file = './{}_split.json'.format(dataset_name)
text_aug(ori_file, dataset_name, aug_type)
make_node2id_eng_text(dataset_name, aug_type, remove_StopWord)