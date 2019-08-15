# Created by: Luke Richards
# Purpose: This file outlines a general framework for a more modular GLS. This code as of now will not run as it is pseudo-code written by Frank Ferraro

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import pandas as pd

class GroundedLanguageClassifier(nn.Module):
    def __init__(self, text_encoder, percept_encoder, corr_scorer):

        super(GroundedLanguageClassifier, self).__init__()

        self.text_encoder = text_encoder
        self.percept_encoder = percept_encoder
        self.scoring = corr_scorer
    def forward(self, text, percepts):
        #changing rep with representation
        text_rep = self.text_encoder(text)
        percept_rep = self.percept_encoder(percepts)
        correspond_score = self.scoring(text_rep, percept_rep)
        return correspond_score

class EmptyExtractor(nn.Module): #Semantic parsing: Luke's research
    def __init__(self):
        super(EmptyExtractor, self).__init__()
        pass
    def forward(self, *args):
        return torch.empty(0)

# Extract features from Luke RSS 2019 dataset
# from a csv file, read in file, forward a single vector for given image name
class CNNFeatureExtractor(nn.Module):
    def __init__(self, filePath,num_features):
        super(CNNFeatureExtractor, self).__init__()

        self.data = pd.read_csv(filePath)
        self.data.set_index('image_name', inplace=True)

    def forward(self, image_name,*args):
        return self.data.loc[image_name, :].values

class MultiLabelBinaryMLPScorer(nn.Module):
    def __init__(self, input_size, num_classifiers,
        num_layers=0, layer_size=[],
        activation=nn.functional.tanh):

        layer_size_list = self._get_layer_dims(num_layers, layer_size)
        self.layers, prev_size = [], input_size
        self.activation_fns = []
        for l in layer_size_list:
            self.activation_fns.append(activation)
            self.layers.append(nn.Linear(prev_size, l))
            prev_size = l
        self.layers.append(nn.Linear(prev_size, num_classifiers))
        self.activation_fns.append(nn.Sequential())

    def _get_layer_dims(self, num_layers, layer_size):
        if type(layer_size) == type([]):
            if len(layer_size) == num_layers:
                layer_size_list = layer_size
            else:
                raise Exception("bad")
        elif type(layer_size) == type(0):
            layer_size_list = [layer_size for _ in range(num_layers)]
        return layer_size_list

    def forward(self, text_feats, vis_feats):
        logits = concat(text_feats, vis_feats)
        for act_fn, layer in zip(self.activation_fns, self.layers):
            logits = act_fn(layer(logits))
        return nn.functional.logsigmoid(logits)

class GLSLearner:
    def __init__(self, model, loss = nn.NLLLoss(), optim_factory = torch.optim.Adam):
        self.model = model
        self.loss = loss
        self.optimizer = optim_factory(self.model.parameters())

    def train(self, train_dataset, val_dataset, train_params):
        iter_index = 0
        ## assume train_params has a function that allows us to train
        ## for a max number of iterations, or has some criteria based
        ## on the loss and/or its gradient
        while train_params.continue_training(loss_val, iter_index):
            train_batch = get_next_batch(train_dataset, iter_index)
            predicted_values = self.model(train_batch.text, train_batch.percepts)
            loss_score = self.loss(y_true=train_batch.labels,
                             y_pred=predicted_values)
            loss_score.backward()
            self.optimizer.step()
            if train_params.do_val(iter_index):
                val_batch = get_next_batch(val_dataset, iter_index)
                predictions = self.predictions(val_batch)
                self.evaluate(val_batch.labels, predictions)
            iter_index += 1

    def predict(self, dataset):
        # did you mean test phase?
        ## get predictions
        return predictions

    def evaluate(self, y_true, y_pred):
        ## perform whatever evaluations
        print("eval")
