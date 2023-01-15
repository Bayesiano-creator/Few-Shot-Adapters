import os
import xml.etree.ElementTree as ET
from random import shuffle, sample
from pysentimiento.preprocessing import preprocess_tweet
from tools.TweetNormalizer import normalizeTweet
import torch
from torch.utils.data import Dataset


class BasePAN17():
    
    def __init__(self, Dir, split, language, tokenizer, gender_dict, variety_dict, tweet_batch_size, max_seq_len, preprocess_text):
        self.Dir          = Dir
        self.split        = split
        self.language     = language
        self.tokenizer    = tokenizer
        self.gender_dict  = gender_dict
        self.variety_dict = variety_dict
        self.tw_bsz       = tweet_batch_size
        
        print("\nReading data...")
        
        self.authors   = self.get_authors(Dir, split, language)
        self.author_lb = self.get_author_labels(Dir, split, language)
        
        self.author_ids = {}
        for i in range(len(self.authors)):
            self.author_ids[ self.authors[i] ] = i
        
        
        # Save authors splited by gender ----------------------------------
        
        # create empty dictionary of authors per label
        
        self.splited_authors = {}
        for i in gender_dict.values():
            self.splited_authors[ i ] = []
        
        # fill dictionary 
        
        for author in self.authors:
            gl = self.author_lb[author]['gender']
            self.splited_authors[ gl ].append(author)
            
        # shuffle authors
        
        for i in gender_dict.values():
            shuffle(self.splited_authors[i])
        
        #----------------------------------------------------------------
        
        self.data = self.get_tweets_in_batches(Dir, split, language)
        
        shuffle(self.data)
        
        if preprocess_text:
            print("    Done\nPreprocessing text...")
            
            if self.language == 'es':
                preprocessed   = [preprocess_tweet(instance['text']) for instance in self.data]
            elif self.language == 'en':
                preprocessed   = [normalizeTweet(instance['text'])   for instance in self.data]
            
        else:
            preprocessed   = [instance['text'] for instance in self.data]
        
        print("    Done\nTokenizing...")
        
        self.encodings = self.tokenizer(preprocessed, max_length = max_seq_len, 
                                                      truncation = True, 
                                                      padding    = True,
                                                      return_tensors = 'pt')
         
        print("    Done\nMerging data...")
        
        for i in range(len(self.data)):
            self.data[i].update( {key: self.encodings[key][i] for key in self.encodings.keys()} )
        
        print("    Done\n\nTotal Instances: " + str(len(self.data)) + '\n')

        
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
            author, gender, variety = line.split(':::')
            variety = variety[:-1]                       

            gl = self.gender_dict[gender]
            vl = self.variety_dict[variety]

            author_lb[author] = {'gender': gl, 'variety': vl}

        lb_file.close()
        
        return author_lb
     
        
    def get_tweets_in_batches(self, Dir, split, language):
        data   = []

        for author in self.authors:
            tw_file_name = os.path.join(Dir, split, language, author + '.xml')
            tree         = ET.parse(tw_file_name)
            root         = tree.getroot()
            documents    = root[0]
            total_tweets = len(documents)

            for i in range(0, total_tweets, self.tw_bsz):
                doc_batch = documents[i : i + self.tw_bsz]
                tweets    = ''

                for document in doc_batch:
                    tweets += document.text + '\n'

                data.append( {'author': author, 'text': tweets, **self.author_lb[author]} )
        
        return data
    
    
    def cross_val(self, k, val_idx, num_authors):
        
        if k > 1:
            sz     = int(len(self.authors) / len(self.gender_dict))
            val_sz = int(sz / k)
        if k == 1:
            sz     = int(len(self.authors) / len(self.gender_dict))
            val_sz = 0
        
        splited_train = {}
        splited_val   = {}
        
        for i in self.gender_dict.values():
            splited_train[i] = self.splited_authors[i][0:( val_sz*val_idx )] + self.splited_authors[i][( val_sz*(val_idx+1) ):sz]
            splited_val[i]   = self.splited_authors[i][( val_sz*val_idx ):( val_sz*(val_idx+1) )]
        
        authors_train = []
        authors_val   = []
        
        for i in self.gender_dict.values():
            authors_train += sample(splited_train[i], num_authors)
            authors_val   += splited_val[i]
        
        
        return authors_train, authors_val
    
    

class DatasetPAN17(Dataset):
    
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
# NLI------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------



class BasePAN17nli():
    
    def __init__(self, Dir, split, language, tokenizer, gender_dict, variety_dict, tweet_batch_size, max_seq_len, preprocess_text, label, label_hip, nli_label2id):
        self.Dir          = Dir
        self.split        = split
        self.language     = language
        self.tokenizer    = tokenizer
        self.tw_bsz       = tweet_batch_size
        self.gender_dict  = gender_dict
        self.variety_dict = variety_dict
        
        # NLI
        self.label        = label
        self.label_hip    = label_hip
        self.nli_label2id = nli_label2id
        
        
        print("\nReading data...")
        
        self.authors   = self.get_authors(Dir, split, language)
        self.author_lb = self.get_author_labels(Dir, split, language)
        
        self.author_ids = {}
        for i in range(len(self.authors)):
            self.author_ids[ self.authors[i] ] = i
        
        
        # Save authors splited by gender ----------------------------------
        
        # create empty dictionary of authors per label
        
        self.splited_authors = {}
        for i in gender_dict.values():
            self.splited_authors[ i ] = []
        
        # fill dictionary 
        
        for author in self.authors:
            gl = self.author_lb[author]['gender']
            self.splited_authors[ gl ].append(author)
            
        # shuffle authors
        
        for i in gender_dict.values():
            shuffle(self.splited_authors[i])
        
        #----------------------------------------------------------------
        
        self.data = self.get_tweets_in_batches(Dir, split, language)
        
        if preprocess_text:
            print("    Done\nPreprocessing text...")

            if self.language == 'es':
                preprocessed_pre   = [normalizeTweet(instance['text']) for instance in self.data]
                preprocessed_hyp   = [normalizeTweet(instance['hypothesis']) for instance in self.data]
            elif self.language == 'en':
                preprocessed_pre   = [normalizeTweet(instance['text'])   for instance in self.data]
                preprocessed_hyp   = [normalizeTweet(instance['hypothesis'])   for instance in self.data]
            
        else:
            preprocessed_pre   = [instance['text'] for instance in self.data]
            preprocessed_hyp   = [instance['hypothesis'] for instance in self.data]
        
        print("    Done\nTokenizing...")
        
        self.encodings = self.tokenizer(preprocessed_pre, preprocessed_hyp, max_length = max_seq_len, 
                                                                            truncation = True, 
                                                                            padding    = True,
                                                                            return_tensors = 'pt')
         
        print("    Done\nMerging data...")
        
        for i in range(len(self.data)):
            self.data[i].update( {key: self.encodings[key][i] for key in self.encodings.keys()} )
        
        print("    Done\n\nTotal Instances: " + str(len(self.data)) + '\n')

        
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
            author, gender, variety = line.split(':::')
            variety = variety[:-1]                       

            gl = self.gender_dict[gender]
            vl = self.variety_dict[variety]

            author_lb[author] = {'gender': gl, 'variety': vl}

        lb_file.close()
        
        return author_lb
     
        
    def get_tweets_in_batches(self, Dir, split, language):
        data   = []
        data_tuples = []
        
        for author in self.authors:
            tw_file_name = os.path.join(Dir, split, language, author + '.xml')
            tree         = ET.parse(tw_file_name)
            root         = tree.getroot()
            documents    = root[0]
            total_tweets = len(documents)

            for i in range(0, total_tweets, self.tw_bsz):
                doc_batch = documents[i : i + self.tw_bsz]
                tweets    = ''

                for document in doc_batch:
                    tweets += document.text + ' '
                
                instance = []
                for lb in self.label_hip.keys():
                    if lb == self.author_lb[author][self.label]:
                        relation = 'entailment'
                    else:
                        relation = 'contradiction'
                    
                    instance.append( {'author': author, 
                                      'text': tweets, 
                                      'hypothesis': self.label_hip[lb], 
                                      'nli_label': self.nli_label2id[relation],  
                                      **self.author_lb[author]} )
                        
                instance = tuple(instance)
                data_tuples.append(instance)
            
        shuffle(data_tuples)
        for instance in data_tuples:
            data += list(instance)
                               
        return data
    
    
    def cross_val(self, k, val_idx, num_authors):
        
        if k > 1:
            sz     = int(len(self.authors) / len(self.gender_dict))
            val_sz = int(sz / k)
        if k == 1:
            sz     = int(len(self.authors) / len(self.gender_dict))
            val_sz = 0
        
        splited_train = {}
        splited_val   = {}
        
        for i in self.gender_dict.values():
            splited_train[i] = self.splited_authors[i][0:( val_sz*val_idx )] + self.splited_authors[i][( val_sz*(val_idx+1) ):sz]
            splited_val[i]   = self.splited_authors[i][( val_sz*val_idx ):( val_sz*(val_idx+1) )]
        
        authors_train = []
        authors_val   = []
        
        for i in self.gender_dict.values():
            authors_train += sample(splited_train[i], num_authors)
            authors_val   += splited_val[i]
        
        
        return authors_train, authors_val
    
    

class DatasetPAN17nli(Dataset):
    
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
    
    