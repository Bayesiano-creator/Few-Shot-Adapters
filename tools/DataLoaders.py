import os
import xml.etree.ElementTree as ET
from random import shuffle, sample
from pysentimiento.preprocessing import preprocess_tweet
from tools.TweetNormalizer import normalizeTweet
import torch
from torch.utils.data import Dataset


class BasePAN():
    
    def __init__(self, Dir, split, language, label_idx, class_dict, label_name):
        """
            Dir        (str) : the name of the directory where dataset is stored
            split      (str) : the name of split to use, train or test
            language   (str) : language to use
            label_idx  (int) : number of label to work on
            class_dict (dict): name of class to integer id of class
            label_name (str) : name of label
        """
        
        # save parameters -------------------------------------------------
        
        self.Dir         = Dir
        self.split       = split
        self.language    = language
        self.label_idx   = label_idx
        self.class_dict  = class_dict
        self.label_name  = label_name
        self.num_classes = len(class_dict)
        
        
        # get author ids and labels ---------------------------------------
        
        self.authors   = self.get_authors(Dir, split, language)
        self.author_lb = self.get_author_labels(Dir, split, language)
        
        
        # create dictionary author2idx -----------------------------------
        
        self.author_ids = {}
        for i in range(len(self.authors)):
            self.author_ids[ self.authors[i] ] = i
        
        
        # Save authors splited by classes ---------------------------------
        
        self.splited_authors = {}
        for i in self.class_dict.values():
            self.splited_authors[ i ] = []
        
        for author in self.authors:
            lb = self.author_lb[ author ]
            self.splited_authors[ lb ].append( author )
            
        
        # shuffle authors -------------------------------------------------
        
        for i in self.class_dict.values():
            shuffle(self.splited_authors[i])
        
        
        #------------------------------------------------------------------

    
    
    def get_authors(self, Dir, split, language):
        path    = os.path.join(Dir, split, language)
        files   = os.listdir(path)
        authors = [ file[0:-4] for file in files ] 
        
        return authors
    
    
    
    def get_author_labels(self, Dir, split, language):
        lb_file_name = os.path.join(Dir, split, language + '.txt')
        lb_file      = open(lb_file_name, "r")
        author_lb    = dict()

        for line in lb_file:
            attributes     = list( line.split(':::') )
            if attributes[-1][-1] == '\n':
                attributes[-1] = attributes[-1][:-1]
            
            author = attributes[0]
            lb     = attributes[ self.label_idx ]

            author_lb[author] = self.class_dict[lb]

        lb_file.close()
        
        return author_lb
    
    
    def get_tweets_in_batches(self, Dir, split, language, tweet_bsz):
        data   = []
        
        for author in self.authors:
            tw_file_name = os.path.join(Dir, split, language, author + '.xml')
            tree         = ET.parse(tw_file_name)
            root         = tree.getroot()
            documents    = root[0] if root[0].tag == 'documents' else root
            total_tweets = len(documents)

            for i in range(0, total_tweets, tweet_bsz):
                doc_batch = documents[i : i + tweet_bsz]
                tweets    = ''

                for document in doc_batch:
                    tweets += document.text + '\n'

                data.append( {'author': author, 'text': tweets, self.label_name: self.author_lb[author]} )
        
        shuffle(data)
        
        return data
    
    
    
    def get_tweets_in_batches_NLI(self, Dir, split, language, tweet_bsz, label_hyp, nli_label2id):
        data   = []
        data_tuples = []
        
        for author in self.authors:
            tw_file_name = os.path.join(Dir, split, language, author + '.xml')
            tree         = ET.parse(tw_file_name)
            root         = tree.getroot()
            documents    = root[0] if root[0].tag == 'documents' else root
            total_tweets = len(documents)

            for i in range(0, total_tweets, tweet_bsz):
                doc_batch = documents[i : i + tweet_bsz]
                tweets    = ''

                for document in doc_batch:
                    tweets += document.text + ' '
                
                instance = []
                for lb in label_hyp.keys():
                    if lb == self.author_lb[author]:
                        relation = 'entailment'
                    else:
                        relation = 'contradiction'
                    
                    instance.append( {'author': author, 
                                      'text': tweets, 
                                      self.label_name: self.author_lb[author],
                                      'hypothesis': label_hyp[lb], 
                                      'nli_label': nli_label2id[relation] })
                        
                instance = tuple(instance)
                data_tuples.append(instance)
            
        shuffle(data_tuples)
        for instance in data_tuples:
            data += list(instance)
                               
        return data
    
    
    
    def get_all_data(self, tweet_bsz, tokenizer, max_seq_len, preprocess_text, NLI=False, label_hyp=None, nli_label2id=None):
        print("\nReading data...")
        
        if NLI:
            self.data = self.get_tweets_in_batches_NLI(self.Dir, self.split, self.language, tweet_bsz, label_hyp, nli_label2id)
        else:
            self.data = self.get_tweets_in_batches(self.Dir, self.split, self.language, tweet_bsz)
        
        
        # -----------------------------------------------------------------------------
        if NLI:
            if preprocess_text:
                print("    Done\nPreprocessing text...")
                preprocessed_pre   = [normalizeTweet(instance['text'])   for instance in self.data]
                preprocessed_hyp   = [normalizeTweet(instance['hypothesis'])   for instance in self.data]

            else:
                preprocessed_pre   = [instance['text'] for instance in self.data]
                preprocessed_hyp   = [instance['hypothesis'] for instance in self.data]
            
            print("    Done\nTokenizing...")
        
            self.encodings = tokenizer(preprocessed_pre, preprocessed_hyp, max_length = max_seq_len, 
                                                                            truncation = True, 
                                                                            padding    = True,
                                                                            return_tensors = 'pt')
        
        # -----------------------------------------------------------------------------
        else:
            if preprocess_text:
                print("    Done\nPreprocessing text...")

                if self.language == 'es':
                    preprocessed   = [preprocess_tweet(instance['text']) for instance in self.data]
                elif self.language == 'en':
                    preprocessed   = [normalizeTweet(instance['text'])   for instance in self.data]

            else:
                preprocessed   = [instance['text'] for instance in self.data]

            print("    Done\nTokenizing...")

            self.encodings = tokenizer(preprocessed, max_length = max_seq_len, truncation = True, padding = True, return_tensors = 'pt')
        
        # ----------------------------------------------------------------------------- 
        print("    Done\nMerging data...")
        
        for i in range(len(self.data)):
            self.data[i].update( {key: self.encodings[key][i] for key in self.encodings.keys()} )
        
        print("    Done\n\nTotal Instances: " + str(len(self.data)) + '\n')
        
    
    
    def cross_val(self, k, num_authors):
        
        if k > 1:
            sz     = len(self.authors) // self.num_classes
            val_sz = sz // k
        if k == 1:
            sz     = len(self.authors) // self.num_classes
            val_sz = 0
        
        splits = []
        
        for val_idx in range(k):
            
            splited_train = {}
            splited_val   = {}
        
            for i in self.class_dict.values():
                sz     = len(self.splited_authors[i])
                val_sz = sz // k
                
                splited_train[i] = self.splited_authors[i][0:( val_sz*val_idx )] + self.splited_authors[i][( val_sz*(val_idx+1) ):sz]
                splited_val[i]   = self.splited_authors[i][( val_sz*val_idx ):( val_sz*(val_idx+1) )]

            authors_train = []
            authors_val   = []

            for i in self.class_dict.values():
                authors_train += sample(splited_train[i], min(num_authors, len(splited_train[i]) ))
                authors_val   += splited_val[i]

            splits.append( (authors_train, authors_val) )
        
        return splits

    
# ---------------------------------------------------------------------------------------------------------
# NORMAL --------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------


class DatasetPAN(Dataset):
    
    def __init__(self, Base_Dataset, label):
        self.Base_Dataset = Base_Dataset
        self.label        = label
        
        self.authors = self.Base_Dataset.authors
        
    def __len__(self):
        return len(self.Base_Dataset.data)
    
    def __getitem__(self, idx):
        keys = ['input_ids', 'attention_mask', 'author']
        item = {key: self.Base_Dataset.data[idx][key] for key in keys}
        item['labels'] = torch.tensor(self.Base_Dataset.data[idx][self.label])
        
        return item
    

    
class DatasetCrossVal(Dataset):
    
    def __init__(self, base, authors, label):
        self.authors = authors
        self.label   = label
        
        self.data = list(filter(lambda inst: inst['author'] in self.authors, base.data))
        
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        keys = ['input_ids', 'attention_mask', 'author']
        item = {key: self.data[idx][key] for key in keys}
        item['labels'] = torch.tensor(self.data[idx][self.label])
        
        return item

    
    
# ---------------------------------------------------------------------------------------------------------
# NLI -----------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------



class DatasetPANnli(Dataset):
    
    def __init__(self, Base_Dataset):
        self.Base_Dataset = Base_Dataset
        
        self.authors = self.Base_Dataset.authors
        
    def __len__(self):
        return len(self.Base_Dataset.data)
    
    def __getitem__(self, idx):
        keys = ['input_ids', 'attention_mask', 'author', 'hypothesis']
        item = {key: self.Base_Dataset.data[idx][key] for key in keys}
        item['labels'] = torch.tensor(self.Base_Dataset.data[idx]['nli_label'])
        
        return item
    

    
class DatasetCrossValnli(Dataset):
    
    def __init__(self, base, authors):
        self.authors = authors
        
        self.data = list(filter(lambda inst: inst['author'] in self.authors, base.data))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        keys = ['input_ids', 'attention_mask', 'author', 'hypothesis']
        item = {key: self.data[idx][key] for key in keys}
        item['labels'] = torch.tensor(self.data[idx]['nli_label'])
        
        return item
    
    