import os
import torch
import torch.nn as nn
import training as t
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class embedder(nn.Module):

    def __init__(self,anime_count:int,studio_count:int,genre_count:int,type_count:int,encode_dim:int,
                      score_dict:dict,type_dict:dict,genre_dict:dict,members_dict:dict,studio_dict:dict):
        super(embedder,self).__init__()
        self.anime_embedding = nn.Embedding(anime_count,encode_dim,padding_idx=0)
        self.studio_embedding = nn.Embedding(studio_count,encode_dim,padding_idx=0)
        self.hidden = nn.Linear(encode_dim,30)
        self.dropout = nn.Dropout(.2)
        self.rating_output = nn.Linear(30,1)
        self.members_output = nn.Linear(30,1)
        self.type_output = nn.Linear(30,type_count)
        self.genre_output = nn.Linear(30,genre_count)
        self.anime_output = nn.Linear(30,anime_count)

        self.get_score = np.vectorize(lambda i: score_dict.get(i,0))
        self.get_members = np.vectorize(lambda i: members_dict.get(i,0))
        self.get_type = np.vectorize(lambda i: type_dict.get(i,0))
        self.get_genre = np.vectorize(lambda i: genre_dict.get(i,0))
        self.get_studio = np.vectorize(lambda i: studio_dict.get(i,0))
        self.relu = nn.ReLU()


    def forward(self,show:torch.Tensor,studio:torch.Tensor):
        a = self.anime_embedding(show)
        b = self.studio_embedding(studio)
        x = a*b
        x = self.hidden(x)
        x = self.dropout(x)
        rating = torch.self.rating_output(x)
        members = self.members_output(x)
        type = self.type_output(x)
        genre = self.genre_output(x)
        anime = self.anime_output(x)
        return rating, members, type, genre, anime


    def train_step(self,optimizer,animes:torch.Tensor,surroundings:torch.Tensor):
        mseloss = nn.MSELoss()
        nll = nn.NLLLoss()

        losses = []
        batch_num = 0
        for show in animes:

            optimizer.zero_grad()
            outputs = self(show,self.get_studio(show))
            loss = 0
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if batch_num % 50 ==0 :
                print('batch {} loss: {}'.format(batch_num,loss.item()))
            batch_num+=1
        print("average loss ",sum(losses)/len(losses))

    def encode_all(self):
        input = torch.arange(1,self.anime_count).to(device)
        self.encodings = self.embedding(input)

    def decode(self,inputs : torch.Tensor,topk=1):
        if self.encodings is None:
            print("should encode first")
            exit(1)
        cossim = nn.CosineSimilarity(dim=-1)
        goal = inputs.unsqueeze(-2).repeat(1,1,self.encodings.shape[0],1).to(device)
        return torch.topk(cossim(self.encodings,goal),dim=-1,k=topk).indices

if __name__ == "__main__":
