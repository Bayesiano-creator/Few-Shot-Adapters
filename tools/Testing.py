from tqdm import tqdm
import torch
import transformers.adapters.composition as AC  
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    
    acc   = (preds == p.label_ids).mean()
    f1s   = f1_score(y_true = p.label_ids, y_pred = preds, average = 'macro')
    
    return {"acc": acc, "f1": f1s}


def compute_accuracy_encdec(p: EvalPrediction):
    preds = np.argmax(p.predictions[0], axis=1)
    
    acc   = (preds == p.label_ids).mean()
    f1s   = f1_score(y_true = p.label_ids, y_pred = preds, average = 'macro')
    
    return {"acc": acc, "f1": f1s}


def compute_author_predictions(dataLoader, predictions, task, num_labels):
    y_true      = []
    y_pred_soft = []
    y_pred_hard = []
    
    pbar = tqdm(dataLoader.authors)
    
    for author in pbar:
        # finds all instances of author
        author_idx = [idx for idx in range(len(dataLoader.data)) if dataLoader.data[idx]['author'] == author]

        # get truth labels with fst instance and initialize scores
        fst      = dataLoader.data[author_idx[0]]
        truth    = fst[task] 
        
        # initialize votes
        scores_soft = np.zeros( num_labels ) 
        scores_hard = np.zeros( num_labels ) 

        for idx in author_idx:
            # get prediction and accumulate
            pred = predictions[idx]
            
            # soft vote
            y = np.exp(pred)/np.sum(np.exp(pred))
            scores_soft += y
            
            # hard vote
            scores_hard[ np.argmax(y) ] += 1
            
        
        y_true.append(truth)
        y_pred_soft.append(np.argmax(scores_soft))
        y_pred_hard.append(np.argmax(scores_hard))
    
    y_true      = np.array(y_true)
    y_pred_soft = np.array(y_pred_soft)
    y_pred_hard = np.array(y_pred_hard)
    
    return {"authors": dataLoader.authors, "true": y_true, "pred_soft": y_pred_soft, "pred_hard": y_pred_hard}


def compute_author_predictions_nli(dataLoader, predictions, task, num_labels, nli_label2id):
    
    y_true      = []
    y_pred_soft = []
    y_pred_hard = []
    
    successful_preds = 0
    cont = 0
    
    pbar = tqdm(dataLoader.authors)
    
    for author in pbar:
        # finds all instances of author
        author_idx = [idx for idx in range(len(dataLoader.data)) if dataLoader.data[idx]['author'] == author]

        # get truth labels with fst instance and initialize scores
        fst      = dataLoader.data[author_idx[0]]
        truth    = fst[task] 
        
        # initialize votes
        scores_soft = np.zeros( num_labels ) 
        scores_hard = np.zeros( num_labels ) 
        
        for i in range(0, len(author_idx), num_labels):
            
            pred = np.zeros(num_labels)
            for j in range(num_labels):
                pred[j] = predictions[author_idx[i+j], nli_label2id['entailment']]
            
            # soft vote
            y = np.exp(pred)/np.sum(np.exp(pred))
            scores_soft += y
            
            # hard vote
            scores_hard[ np.argmax(y) ] += 1
        
        y_true.append(truth)
        y_pred_soft.append(np.argmax(scores_soft))
        y_pred_hard.append(np.argmax(scores_hard))

        cont += 1
        if np.argmax(scores_soft) == truth:
            successful_preds += 1
        
        pbar.set_description("acc: " + str(successful_preds/cont))
        
    y_true      = np.array(y_true)
    y_pred_soft = np.array(y_pred_soft)
    y_pred_hard = np.array(y_pred_hard)
    
    return {"authors": dataLoader.authors, "true": y_true, "pred_soft": y_pred_soft, "pred_hard": y_pred_hard}


def compute_author_predictions_nli_LR(trainLoad, testLoad, trainPred, testPred, task, num_labels):
    
    # Train LR
    LR = LogisticRegression()
    
    N = len(trainPred)
    X = trainPred.reshape( (N//num_labels, 3*num_labels) )
    
    y = np.array([ instance[task] for instance in trainLoad.data])
    y = y.reshape(N//num_labels, num_labels)[:,0]
    
    LR.fit(X,y)
    
    # Compute New Predictions
    
    N = len(testPred)
    X_test = testPred.reshape( (N//num_labels, 3*num_labels) )
    predictions = LR.predict_proba(X_test)
    
    
    # Just as before
    
    y_true      = []
    y_pred_soft = []
    y_pred_hard = []
    
    successful_preds = 0
    cont = 0
    
    # ---------------------------------
    
    dataLoader = testLoad
    
    pbar = tqdm(dataLoader.authors)
    
    for author in pbar:
        # finds all instances of author
        author_idx = [idx for idx in range(len(dataLoader.data)) if dataLoader.data[idx]['author'] == author]

        # get truth labels with fst instance and initialize scores
        fst      = dataLoader.data[author_idx[0]]
        truth    = fst[task] 
        
        # initialize votes
        scores_soft = np.zeros( num_labels ) 
        scores_hard = np.zeros( num_labels ) 
        
        for i in range(0, len(author_idx), num_labels):
            
            # get prediction and accumulate
            pred = predictions[author_idx[i] // num_labels]
            
            # soft vote
            y = np.exp(pred)/np.sum(np.exp(pred))
            scores_soft += y
            
            # hard vote
            scores_hard[ np.argmax(y) ] += 1
        
        y_true.append(truth)
        y_pred_soft.append(np.argmax(scores_soft))
        y_pred_hard.append(np.argmax(scores_hard))

        cont += 1
        if np.argmax(scores_soft) == truth:
            successful_preds += 1
        
        pbar.set_description("acc: " + str(successful_preds/cont))
        
    y_true      = np.array(y_true)
    y_pred_soft = np.array(y_pred_soft)
    y_pred_hard = np.array(y_pred_hard)
    
    return {"authors": dataLoader.authors, "true": y_true, "pred_soft": y_pred_soft, "pred_hard": y_pred_hard}