'''Build Bert document classifier, the code is revised from
https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO#scrollTo=6J-FYdx6nFE_

'''
import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch.nn.functional as F

from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from nltk.tokenize import sent_tokenize
import random
import json
import copy

import io
import os
from collections import Counter

import evaluator

from transformers import BertModel, BertTokenizer
from torch import nn
import hyperparse 
usermode, usermode_str = hyperparse.parse("usermode")
if "anonfair" in usermode:
    usermode["rmna"] = usermode["anonfair"]
if "mfair" in usermode:
    usermode["rmna"] = usermode["mfair"]

if "seed" in usermode:
    random.seed(usermode["seed"])
    np.random.seed(usermode["seed"])

class BertForMultiTask(nn.Module):
    def __init__(self, bert_model_name, num_labels, num_labels_secondary):
        super(BertForMultiTask, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        if "head1" in usermode:
            return
        self.secondary_classifier = nn.Linear(self.bert.config.hidden_size, num_labels_secondary)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, protected_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        if "head1" in usermode:
            outputs.secondary_probs = input_ids.new_zeros(len(input_ids))
            return outputs

        if "mfair" in usermode:
            #secondary_probs = torch.softmax(self.secondary_classifier(outputs.hidden_states[-1][:, 0]), 1)
            secondary_probs = self.secondary_classifier(outputs.hidden_states[-1][:, 0])
        else:
            secondary_probs = torch.sigmoid(self.secondary_classifier(outputs.hidden_states[-1][:, 0]).squeeze())

        outputs.secondary_probs = secondary_probs
        if protected_labels is not None:
            loss_fct = nn.MSELoss()
            secondary_loss = loss_fct(secondary_probs, protected_labels.to(torch.float32))
            outputs.loss = outputs.loss + secondary_loss
        return outputs

class BertForMultiTask_del(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, num_labels_secondary):
        super(BertForMultiTask, self).__init__(BertConfig.from_pretrained(bert_model_name, num_labels=num_labels))
        # Initialize the secondary classifier
        if "head1" in usermode:
            return
        self.secondary_classifier = nn.Linear(self.config.hidden_size, num_labels_secondary)
        self.num_labels_secondary = num_labels_secondary

    def forward1(self, input_ids, attention_mask=None, labels=None, protected_labels=None):
        # First, call the forward method of the superclass to handle the primary classification task
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        if "head1" in usermode:
            outputs.secondary_probs = input_ids.new_zeros(len(input_ids))
            return outputs

        # Use the last hidden state for the secondary task
        # outputs[0] is the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        # Apply pooling over the last hidden state
        # Here we simply take the first token's representation as the pooled output
        secondary_pooled_output = last_hidden_state[:, 0]

        secondary_logits = self.secondary_classifier(secondary_pooled_output)

        if self.num_labels_secondary == 1:
            secondary_probs = torch.sigmoid(secondary_logits).squeeze()
        else:
            secondary_probs = torch.softmax(secondary_logits, dim=1)

        outputs.secondary_probs = secondary_probs

        if labels is not None and protected_labels is not None:
            if self.num_labels_secondary == 1:
                loss_fct = nn.MSELoss()
            else:
                loss_fct = nn.CrossEntropyLoss()
            secondary_loss = loss_fct(secondary_probs, protected_labels)
            outputs.loss = outputs.loss + secondary_loss

        return outputs


class BertForMultiTask_ext(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, num_labels_secondary):
        super(BertForMultiTask, self).__init__(BertConfig.from_pretrained(bert_model_name, num_labels=num_labels))

    def forward(self, input_ids, attention_mask=None, labels=None, protected_labels=None):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        return outputs

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_f1(preds, labels):
    macro_score = f1_score(
        y_true=labels, y_pred=preds,
        average='macro',
    )
    weighted_score = f1_score(
        y_true=labels, y_pred=preds,
        average='weighted',
    )
    print('Weighted F1-score: ', weighted_score)
    print('Macro F1-score: ', macro_score)
    return macro_score, weighted_score


def build_bert(lang, odir, params=None):
    '''Google Bert Classifier
        lang: The language name
        odir: output directory of prediction results
    '''
    if not params:
        params = dict()
        params['balance_ratio'] = 0.9
        params['freeze'] = False
        params['decay_rate'] = .001
        params['lr'] = 2e-5
        params['warm_steps'] = 100
        params['train_steps'] = 1000
        params['batch_size'] = 16
        params['balance'] = True

    split_dir = './data/'+lang+'/'

    if "data" in usermode:
        split_dir = './data/'+usermode["data"]+'/'

    if torch.cuda.is_available():
        device = "cuda:0"#str(get_freer_gpu())
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(device)

    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name())
    print('Number of GPUs: ', n_gpu)

    print('Loading Datasets and oversample training data...')
    train_df = pd.read_csv(split_dir+'train.tsv', sep='\t', na_values='x')

    if "enforce" in usermode:
        params['batch_size'] = 128
        from torch.nn import CrossEntropyLoss
    
    if "cda" in usermode:
        wlst = json.load(open("dev/gender.json"))
        wdict = dict(wlst + [w[::-1] for w in wlst])
        train_df_2 = copy.deepcopy(train_df)
        for i in range(len(train_df)):
            train_df_2.loc[i, 'text'] = " ".join([wdict.get(w, w) for w in train_df_2.loc[i, 'text'].split()])
        train_df = pd.concat([train_df, train_df_2])


    # oversample the minority class
    if params['balance']:
        label_count = Counter(train_df.label)
        for label_tmp in label_count:
            sample_num = label_count.most_common(1)[0][1] - label_count[label_tmp]
            if sample_num == 0:
                continue
            train_df = pd.concat([train_df,
                train_df[train_df.label==label_tmp].sample(
                    int(sample_num*params['balance_ratio']), replace=True
                )])
        train_df = train_df.reset_index() # to prevent index key error

        valid_df = pd.read_csv(split_dir+'valid.tsv', sep='\t', na_values='x')
        test_df = pd.read_csv(split_dir+'test.tsv', sep='\t', na_values='x')
        data_df = [train_df, valid_df, test_df]
        if "smalldbg" in usermode:
            for i in range(len(data_df)):
                data_df[i] = data_df[i][:int(0.1 * len( data_df[i]))]
        if ("rmna" in usermode or "anonfair" in usermode) and "data" not in usermode:
            print("Size before remove NA: ", [len(df) for df in data_df])
            data_df = [df[df[usermode["rmna"]].notnull()].copy() for df in data_df]
            [df.reset_index(drop=True, inplace=True) for df in data_df]
            print("Size after remove NA: ", [len(df) for df in data_df])
            
            datastat = dict(zip(["Train", "Dev", "Test"], [len(df) for df in data_df]))
            json.dump(datastat, open(os.path.join(odir, "datastat.json"), "w"))
            if "datastat" in usermode:
                exit(0)
            if "rebalance" in usermode:
                df = data_df[0]
                class_counts = df[usermode["rmna"]].value_counts()
                major_class = class_counts.idxmax()
                minor_class = class_counts.idxmin()

                # Calculate the number of samples to generate
                sample_difference = class_counts[major_class] - class_counts[minor_class]

                # Separate the majority and minority classes into different DataFrames
                df_major = df[df[usermode["rmna"]] == major_class]
                df_minor = df[df[usermode["rmna"]] == minor_class]

                # Sample the difference
                df_minor_oversampled = df_minor.sample(sample_difference, replace=True)

                # Concatenate the original DataFrame with the oversampled DataFrame
                df_balanced = pd.concat([df, df_minor_oversampled], ignore_index=True)

                # Shuffle the DataFrame to mix the rows up
                data_df[0] = df_balanced.sample(frac=1).reset_index(drop=True)
                print("Size after rebalance: ", [len(df) for df in data_df])


    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(lambda x: '[CLS] '+ x +' [SEP]')

    if lang == 'English':
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )
    elif lang == 'Chinese':
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-chinese',
            do_lower_case=True
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-uncased',
            do_lower_case=True
        )

    print('Padding Datasets...')
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(lambda x: tokenizer.tokenize(x))

    # convert to indices and pad the sequences
    max_len = 25
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(
            lambda x: pad_sequences(
                [tokenizer.convert_tokens_to_ids(x)],
                maxlen=max_len, dtype="long"
                )[0])

    # create attention masks
    for doc_df in data_df:
        attention_masks = []
        for seq in doc_df.text:
            seq_mask = [float(idx>0) for idx in seq]
            attention_masks.append(seq_mask)
        doc_df['masks'] = attention_masks

    # format train, valid, test
    train_inputs = torch.tensor(data_df[0].text)
    train_labels = torch.tensor(data_df[0].label)
    train_masks = torch.tensor(data_df[0].masks)
    valid_inputs = torch.tensor(data_df[1].text)
    valid_labels = torch.tensor(data_df[1].label)
    valid_masks = torch.tensor(data_df[1].masks)
    test_inputs = torch.tensor(data_df[2].text)
    test_labels = torch.tensor(data_df[2].label)
    test_masks = torch.tensor(data_df[2].masks)
    if ("anonfair" in usermode and "dbgm" not in usermode) or "enforce" in usermode or "mfair" in usermode:
        print(data_df[0])
        protected_train_labels = torch.tensor(data_df[0][usermode["rmna"]].values)
        protected_valid_labels = torch.tensor(data_df[1][usermode["rmna"]].values)
        protected_test_labels = torch.tensor(data_df[2][usermode["rmna"]].values)

    batch_size = params['batch_size']

    if ("anonfair" in usermode and "dbgm" not in usermode) or "enforce" in usermode or "mfair" in usermode:
        train_data = TensorDataset(train_inputs, train_masks, train_labels, protected_train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        valid_data = TensorDataset(
        valid_inputs, valid_masks, valid_labels, protected_valid_labels)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=batch_size)
        test_data = TensorDataset(
            test_inputs, test_masks, test_labels, protected_test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=batch_size)
    else:
        train_data = TensorDataset(
            train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size)
        valid_data = TensorDataset(
            valid_inputs, valid_masks, valid_labels)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=batch_size)
        test_data = TensorDataset(
            test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=batch_size)

    # load the pretrained model
    print('Loading Pretrained Model...')
    if lang == 'English':
        if "anonfair" in usermode or "mfair" in usermode:
            model = BertForMultiTask('bert-base-uncased', num_labels=2, num_labels_secondary=len(usermode["rmna"]) if "mfair" in usermode else 1)
            if "dbgm" in usermode:
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            if "dropout" in usermode:
                for layer in model.bert.bert.encoder.layer:
                    layer.attention.self.dropout.p = usermode["dropout"]
                    layer.attention.output.dropout.p = usermode["dropout"]
                    layer.output.dropout.p = usermode["dropout"]
        else:
            model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2)
    elif lang == 'Chinese':
        model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese', num_labels=2)
    else: # for Spanish, Italian, Portuguese and Polish
        if "anonfair" in usermode:
            model = BertForMultiTask('bert-base-multilingual-uncased', num_labels=2, num_labels_secondary=1)
            if "dbgm" in usermode:
                model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
        else:
            model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-uncased', num_labels=2)
    model.to(device)

    # organize parameters
    param_optimizer = list(model.named_parameters())
    if params['freeze']:
        no_decay = ['bias', 'bert'] # , 'bert' freeze all bert parameters
    else:
        no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': params['decay_rate']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=params['warm_steps'],
        num_training_steps=params['train_steps']
    )

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 10

    # Training
    print('Training the model...')
    for _ in trange(epochs, desc='Epoch'):
        model.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # train batch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            if "anonfair" in usermode or "mfair" in usermode:
                b_input_ids, b_input_mask, b_labels, b_protected_labels = batch if len(batch) == 4 else batch + (None,)
                if "dbgm" in usermode:
                    outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs.loss
                else:
                    outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, protected_labels=b_protected_labels)
                    loss = outputs[0]
            elif "enforce" in usermode:
                b_input_ids, b_input_mask, b_labels, b_protected_labels = batch if len(batch) == 4 else batch + (None,)
                outputs = model(
                    b_input_ids, token_type_ids=None,
                    attention_mask=b_input_mask, labels=b_labels
                )
                loss_fct = CrossEntropyLoss(reduction='none')
                eachloss = loss_fct(outputs[1].view(-1, model.num_labels), b_labels.view(-1))
                if "dp" in usermode:
                    b_protected_labels = b_protected_labels.masked_select(b_labels.bool().unsqueeze(1)).view(-1, b_protected_labels.shape[1])
                    eachloss = eachloss.masked_select(b_labels.bool())
                if len(b_protected_labels.shape) == 2:
                    avgs = torch.stack([eachloss[b_protected_labels[:, i].bool()].mean() for i in range(b_protected_labels.shape[1])])
                    bloss = (avgs - avgs.mean()).abs().mean()
                else:
                    bloss = (b_protected_labels * eachloss).mean() - ((1 - b_protected_labels) * eachloss).mean()
                loss = outputs[0] + 0.1 * bloss
            else:
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                # Forward pass
                outputs = model(
                    b_input_ids, token_type_ids=None,
                    attention_mask=b_input_mask, labels=b_labels
                )
                loss = outputs.loss
            # backward pass
    #            outputs[0].backward()
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            scheduler.step()

            # Update tracking variables
            tr_loss += outputs[0].item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/max(1, nb_tr_steps)))

        '''Validation'''
        best_valid_f1 = 0.0
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        # tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # batch eval
        y_valid_preds = []
        y_valid_protected_probs = []
        y_valid_logits = []
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            if "anonfair" in usermode or "mfair" in usermode: 
                b_input_ids, b_input_mask, b_labels, b_protected_labels = batch
                with torch.no_grad():
                    outputs = model(b_input_ids, attention_mask=b_input_mask)
                    class_logits, protected_probs = outputs.logits, outputs.secondary_probs
                logits = class_logits.detach().cpu().numpy()
                y_valid_logits.extend(logits)
                pred_flat = np.argmax(logits, axis=1).flatten()
                if "mfair" in usermode:
                    protected_pred_flat = protected_probs.detach().cpu().numpy()
                else:
                    protected_pred_flat = protected_probs.detach().cpu().numpy().flatten()
                y_valid_protected_probs.extend(protected_pred_flat)
            else:
                b_input_ids, b_input_mask, b_labels, b_protected_labels = batch if len(batch) == 4 else batch + (None,)
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    outputs = model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)
                # Move logits and labels to CPU
                logits = outputs[0].detach().cpu().numpy()
            # record the prediction
            pred_flat = np.argmax(logits, axis=1).flatten()
            y_valid_preds.extend(pred_flat)

            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        # evaluate the validation f1 score
        f1_m_valid, f1_w_valid = flat_f1(y_valid_preds, data_df[1].label)
        if f1_m_valid > best_valid_f1:
            print(f'Test {usermode_str}....')
            best_valid_f1 = f1_m_valid
            y_preds = []
            y_probs = []
            y_test_protected_probs = []
            y_test_logits = []

            # test if valid gets better results
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                if "anonfair" in usermode or "mfair" in usermode:
                    b_input_ids, b_input_mask, b_labels, b_protected_labels = batch
                    with torch.no_grad():
                        outputs = model(b_input_ids,  attention_mask=b_input_mask)
                        class_logits, protected_probs = outputs.logits, outputs.secondary_probs
                    logits = class_logits
                    if "mfair" in usermode:
                        test_protected_probs_flat = protected_probs.detach().cpu().numpy()
                    else:
                        test_protected_probs_flat = protected_probs.detach().cpu().numpy().flatten()
                    y_test_protected_probs.extend(test_protected_probs_flat)
                    y_test_logits.extend(logits.detach().cpu().numpy())
                else:
                    b_input_ids, b_input_mask, b_labels, b_protected_labels = batch if len(batch) == 4 else batch + (None,)
                    with torch.no_grad():
                        outputs = model(
                            b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
                    logits = outputs[0]
                    if "outnpy" in usermode:
                        y_test_logits.extend(logits.detach().cpu().numpy())
                probs = F.softmax(logits, dim=1)
                probs = probs.detach().cpu().numpy()
                pred_flat = np.argmax(probs, axis=1).flatten()
                y_preds.extend(pred_flat)
                y_probs.extend([item[1] for item in probs])
            # save the predicted results
            test_df = pd.read_csv(os.path.join(split_dir, 'test.tsv'), sep='\t', na_values='x')
            if "rmna" in usermode or "anonfair" in usermode:
                valid_df = pd.read_csv(os.path.join(split_dir, 'valid.tsv'), sep='\t', na_values='x')
                valid_df = valid_df[valid_df[usermode["rmna"]].notnull()].copy()
                test_df = test_df[test_df[usermode["rmna"]].notnull()].copy()
            if "data" in usermode:
                test_df = pd.read_csv(os.path.join(split_dir, 'test.tsv'), sep='\t', na_values='x')
                valid_df = pd.read_csv(os.path.join(split_dir, 'valid.tsv'), sep='\t', na_values='x')
            if "smalldbg" in usermode:
                valid_df = valid_df[:int(0.1 * len( valid_df))]
                test_df = test_df[:int(0.1 * len( test_df))]
            def get_logits(y_logits):
                if "logitd" in usermode:
                    return y_logits[:, -1] - y_logits[:, 0]
                elif "logit0d1" in usermode:
                    return y_logits[:, 0] - y_logits[:, -1]
                elif "logit0" in usermode:
                    return y_logits[:, 0]
                elif "logit1" in usermode:
                    return y_logits[:, -1]
                return y_logits[:, -1]
            if "anonfair" in usermode or "mfair" in usermode:
                # valid
                valid_df['pred'] = y_valid_preds
                valid_df['protected_probs'] = y_valid_protected_probs
                valid_df['logits'] = y_valid_logits

                # Save results                    

                if "mfair" in usermode:
                    outputs_val = np.hstack([get_logits(np.stack(y_valid_logits))[:, None], np.stack(y_valid_protected_probs)])
                else:
                    outputs_val = np.stack([get_logits(np.stack(y_valid_logits)), y_valid_protected_probs]).transpose()
                np.save(os.path.join(odir, "outputs_val.npy"), outputs_val)
                if "data" in usermode:
                    outputs_test = np.hstack([get_logits(np.stack(y_test_logits))[:, None], np.stack(y_test_protected_probs)])
                    np.save(os.path.join(odir, "protected_label_val.npy"), valid_df[usermode["rmna"]].to_numpy().argmax(1))
                    np.save(os.path.join(odir, "protected_label_test.npy"), test_df[usermode["rmna"]].to_numpy().argmax(1))
                else:
                    outputs_test = np.stack([get_logits(np.stack(y_test_logits)), y_test_protected_probs]).transpose()
                    np.save(os.path.join(odir, "protected_label_val.npy"), valid_df[usermode["rmna"]].to_numpy())
                    np.save(os.path.join(odir, "protected_label_test.npy"), test_df[usermode["rmna"]].to_numpy())
                np.save(os.path.join(odir, "outputs_test.npy"), outputs_test)
                np.save(os.path.join(odir, "target_label_test.npy"), test_df["label"].to_numpy())
                np.save(os.path.join(odir, "target_label_val.npy"), valid_df["label"].to_numpy())
                
                
                output_file = os.path.join(odir, f"{lang}-valid.tsv")
                valid_df.to_csv(output_file, sep='\t', index=False)



                # test
                test_df['protected_preds'] = y_test_protected_probs
                test_df['logits'] = y_test_logits
            elif "outnpy" in usermode:
                outputs_test = np.stack([get_logits(np.stack(y_test_logits)), np.zeros(len(y_test_logits))]).transpose()
                if "data" in usermode:
                    np.save(os.path.join(odir, "protected_label_test.npy"), test_df[usermode["rmna"]].to_numpy().argmax(1))
                else:
                    np.save(os.path.join(odir, "protected_label_test.npy"), test_df[usermode["rmna"]].to_numpy())
                np.save(os.path.join(odir, "outputs_test.npy"), outputs_test)
                np.save(os.path.join(odir, "target_label_test.npy"), test_df["label"].to_numpy())

            # Assuming y_preds and y_probs are lists with the same length as the test data
            test_df['pred'] = y_preds
            test_df['pred_prob'] = y_probs

            # Save the modified DataFrame
            if len(test_df) != len(y_preds) or len(test_df) != len(y_probs):
                print(f"Mismatch in lengths. DataFrame: {len(test_df)}, y_preds: {len(y_preds)}, y_probs: {len(y_probs)}")
            output_file = os.path.join(odir, f"{lang}-test.tsv")
            test_df.to_csv(output_file, sep='\t', index=False)


            # save the predicted results
            evaluator.eval(
                odir+lang+'-test.tsv',
                odir+lang+'.score'
            )


if __name__ == '__main__':
    langs = [
        'English'#, 'Italian', 'Polish',
        #'Portuguese', 'Spanish'
    ]
    language_dict = {
        "en": "English",
        "it": "Italian",
        "pl": "Polish",
        "pt": "Portuguese",
        "es": "Spanish",
    }
    odir = './results/bert/'
    if usermode:
        odir = f'./output/{usermode_str}/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    #for lang in langs:
    lang = "English"
    if "lang" in usermode:
        lang = language_dict[usermode["lang"]]
    print('Working on: ', lang)
    build_bert(lang, odir)

