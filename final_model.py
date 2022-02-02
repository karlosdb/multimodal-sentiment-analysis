#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt

import torch.nn as nn

import pickle

import pandas as pd

import torch.optim as optim

# this is not quite the same as the dataset downloaded via CMU-MultimodalSDK
# it's already word aligned, sentence length is 50, and feature size is small
# source: http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosi/seq_length_50/
dataset_path = r"C:\Users\offic\Documents\SFF\CODE\multimodal-sentiment-analysis\mosei_senti_data.pkl" # path of MOSI file
#dataset_path = r"C:\Users\offic\Documents\SFF\CODE\Working With MOSI\mosi_data.pkl" # path of MOSI file
dataset = pickle.load(open(dataset_path, 'rb'))


# In[10]:


from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


# In[25]:


class MOSIDataset(Dataset): # Create Dataset object for the dataset, to extract and be able to work with the data
    def __init__(self, subset, media_type):
        data = dataset[subset] # subset = 'train', 'valid', or 'test' 
        if media_type == 'vision' or media_type == 'audio' or media_type == 'text': # choose modality 
            self.features = torch.from_numpy(data[media_type]).float() # media_type = 'vision', 'audio', 'text'
        elif media_type == 'all': # choose all modalities concatenated
            np_features = np.concatenate((data['vision'], data['audio'], data['text']), axis=2)
            self.features = torch.from_numpy(np_features).float()
        elif media_type == 'vt': # choose visual and textual modalities concatenated 
            np_features = np.concatenate((data['vision'], data['text']), axis=2)
            self.features = torch.from_numpy(np_features).float()
            
        self.labels = torch.from_numpy(data['labels']).float() # extract labels
        
    def __len__(self): # get length of dataset
        return self.labels.shape[0]
    
    def __getitem__(self, idx): # get specific item from dataset
        sample = {'features': self.features[idx, :, :], 'label': self.labels[idx, 0, 0]}
        return sample


# In[26]:


# run one training session of the model with chosen parameters
def run_model(modality='all', batch_size=32, dropout=0.1, hidden_dim=256, n_layers=2):

    # extract/create objects for the training, validation, and test datasets with
    mosi_train = MOSIDataset('train', modality) 
    mosi_valid = MOSIDataset('valid', modality)
    mosi_test = MOSIDataset('test', modality)


    # get sequence length and feature size of dataset
    seq_len, feat_size = mosi_train[0]['features'].shape



    # Create DataLoader objects to be able to iterate over the dataset
    train_loader = DataLoader(mosi_train, batch_size=batch_size,  shuffle=True)
    valid_loader = DataLoader(mosi_valid, batch_size=batch_size,  shuffle=True)
    test_loader = DataLoader(mosi_test, batch_size=batch_size,  shuffle=True)


    # In[29]:



    # Create the RNN module / architecture of the ML model
    class RNN(nn.Module):
        def __init__(self, feat_size, hidden_dim, output_dim, n_layers, dropout):

            super().__init__() # inheirit the super class of neural network module

            self.fc0 = nn.Linear(feat_size, 100) # first fully connected layer, takes in features and brings it down to 100 dimensions

            # GRU neural network layer with dropout
            self.rnn = nn.GRU(100,#feat_size, 
                       hidden_dim, 
                       num_layers=n_layers, 
                       bidirectional=False,
                       dropout=dropout,
                       batch_first=True) # (batch, seq, feat)

            # final fully connected layer, outputs to final prediction
            self.fc = nn.Linear(hidden_dim, output_dim) # for unidirectional LSTM

            # dropout for the final fc layer
            self.dropout = nn.Dropout(dropout)

        # function for what to do/update with each batch iteration
        def forward(self, batch):
            batch0 = self.fc0(batch)
            output, hidden = self.rnn(batch0) # GRU
            hidden = self.dropout(hidden[-1,:,:]) # for unidirectional LSTM
            out = self.fc(hidden)

            return out

    # create model object
    model = RNN(feat_size, hidden_dim, 1, n_layers, dropout)


    # In[30]:


    # In[31]:



    optimizer = optim.Adam(model.parameters()) # Adam hyper parameter optimizer
    
    # choose loss function, MAE/Mean Absolute Error or MSe/Mean Squared Error
    loss_name = 'MAE'
    if loss_name == 'MSE':
        criterion = nn.MSELoss()
    elif loss_name == 'MAE':
        criterion = nn.L1Loss()


    # In[32]:

    # put the model and criterion object onto the gpu if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    model = model.to(device)
    criterion = criterion.to(device)


    # In[33]:


    # In[34]:

    # function for finding binary accuracy of batch using the predictions and labels for that batch
    def accuracy(prediction, label):

        num_agreements = 0 # variable for counting the number of 'agreements', or predictions that match the labels

        # count each agreement, defined by if the prediction and label are on the same side of 0 (both positive or both negative)
        for i in range(len(prediction)):
            if ((prediction[i] > 0 and label[i] > 0) or (prediction[i] < 0 and label[i] < 0)):
                num_agreements += 1
        
        # average agreement rate
        accuracy = num_agreements/len(prediction) 

        # return average agreement rate/accuracy
        return accuracy


    # function for training the RNN model
    def train(model, dataset_loader):
        epoch_loss = 0 # count the loss value for the epoch
        accuracy_score = 0 # count the binary accuracy for the epoch
        model.train() # define this function as the training fuction

        # loop through each batch to predict, optimize, and count loss and accuracy
        for _, batch in enumerate(dataset_loader):
            fts, lbs = batch['features'].to(device), batch['label'].to(device) # extract features and labels
            optimizer.zero_grad() # reset the optimizer
            predictions = model(fts) # run the model on the features to get the predictions
            loss = criterion(predictions, lbs.unsqueeze(1)) # calculate the loss from the labels and predictions
            epoch_loss += loss.item() # add the loss to the total epoch loss

            accuracy_score += accuracy(predictions, lbs.unsqueeze(1)) # calculate and add the binary accuracy to the epoch accuract

            loss.backward() # backwards propogation
            optimizer.step() # run optimizer based on results

        epoch_loss /= len(dataset_loader) # calculate average epoch loss
        accuracy_score /= len(dataset_loader) # calculate average epoch binary accuracy
        return epoch_loss, accuracy_score # return epoch loss and binary accuracy


    # In[35]:

    # function for evaluating the model performance on the validation and testing sets
    def evaluate(model, dataset_loader):
        epoch_loss = 0 # count the loss value for the epoch
        accuracy_score = 0 # count the binary accuracy for the epoch
        model.eval() ## define this function as the evaluation function

        
        with torch.no_grad():# without adjusting any paramers/no optimization
            for _, batch in enumerate(dataset_loader): # loop through each batch
                fts, lbs = batch['features'].to(device), batch['label'].to(device) # extract features and labels
                predictions = model(fts) # run the model on the features to get predictions
                loss = criterion(predictions, lbs.unsqueeze(1)) # calculate loss from predictions and labels
                epoch_loss += loss.item() # add loss to epoch loss
                accuracy_score += accuracy(predictions, lbs.unsqueeze(1)) # add binary accuracy score to epoch score

        epoch_loss /= len(dataset_loader) # calculate average epoch loss
        accuracy_score /= len(dataset_loader) # calcualte average epoch accuracy
        return epoch_loss, accuracy_score # return epoch loss and accuracy score



    # In[36]:


    n_epochs = 20 # number of epochs to train the model per session

    # lists of losses for each epoch for training, validation, and test sets
    tr_losses = [] 
    vl_losses = []
    ts_losses = []

    for i_epoch in range(n_epochs): # loop for number of desired epochs

        # train and evaluate the model and return the loss and accuracy for each set in that epoch
        train_loss, train_accuracy = train(model, train_loader)
        valid_loss, valid_accuracy = evaluate(model, valid_loader)
        test_loss, test_accuracy = evaluate(model, test_loader)

        # add losses to lists of losses
        tr_losses.append(train_loss)
        vl_losses.append(valid_loss)
        ts_losses.append(test_loss)

        # print the epoch loss and accuracy for each set
        print(f'epoch: {i_epoch} train loss: {train_loss:.3f} train accuracy: {train_accuracy:.3f} valid loss: {valid_loss:.3f} valid accuracy {valid_accuracy:.3f} test loss: {test_loss:.3f} test accuracy: {test_accuracy:.3f}')


    # In[37]:

    # create graph of epoch number vs loss
    plt.clf()
    plt.plot(tr_losses,'r-',label='train')
    plt.plot(vl_losses,'g--',label='valid')
    plt.plot(ts_losses,'b-.',label='test')
    plt.legend()
    plt.title('modality: '+modality)
    plt.xlabel('epoch')
    plt.ylabel(loss_name)
    plt.savefig(f'MMSA__modality_{modality}__batch_size_{batch_size}__dropout{dropout*10}__hidden_dim{hidden_dim}__n_layers{n_layers}.png')
    
    return ts_losses[-1] # returning test loss using model trained on max number of epochs


# main function to test parameters 
if __name__ == '__main__':
    # lists of different options for hyperparameters, ever combination will be tested
    # currently set the the best results we have achieved with this model
    modality_options = ['vt']
    batch_size_options = [64]
    dropout_options = [0.5]
    hidden_dim_options = [64]
    n_layers_options = [3]

    # tables to record each
    test_losses = []
    modality_table_values = []
    batch_size_table_values = []
    dropout_table_values = []
    hidden_dim_table_values = []
    n_layers_table_values = []

    # grid search through all of the hyper parameter options
    for modality in modality_options:
        for batch_size in batch_size_options:      
            for dropout in dropout_options:
                for hidden_dim in hidden_dim_options:
                    for n_layers in n_layers_options:  

                        # run model on this combination of the hyper parameters
                        test_loss = run_model(modality=modality, batch_size=batch_size, dropout=dropout, hidden_dim=hidden_dim, n_layers=n_layers)

                        # append the hyper params and the test loss to their respective lists
                        test_losses.append(test_loss)
                        modality_table_values.append(modality)
                        batch_size_table_values.append(batch_size)
                        dropout_table_values.append(dropout)
                        hidden_dim_table_values.append(hidden_dim)
                        n_layers_table_values.append(n_layers)

                        # display the settings and result of the model
                        print('running model on modality', modality, 'batch_size', batch_size, 'test_loss', test_loss, 'dropout', dropout, 'hidden_dim', hidden_dim, 'n_layers', n_layers)
                        print('-'*100)


    # create a csv file for every result using every setting that was run in the grid search            
    d = {'modality': modality_table_values, 'batch_size': batch_size_table_values, 'dropout': dropout_table_values, 'hidden_dim': hidden_dim_table_values, 'n_layers': n_layers_table_values, 'MAE_test_loss': test_losses}
    df = pd.DataFrame(data=d)
    df.to_csv('results.csv', index=False)
