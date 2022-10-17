import os
import torch
import torch.nn as nn
import training as t
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence, pad_sequence

print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def left_shift(mat,x):
    return torch.concat((mat[:,x:],torch.zeros(mat.size()).to(device)[:,:x]),axis=1)

def series_loss(pred_y,y):
    loss_fun = nn.CrossEntropyLoss(ignore_index=0)
    loss = torch.tensor([]).to(device)
    classes = pred_y.shape[-1]

    #np.power(.7,i+1)*
    y = left_shift(y,1)
    #loss = loss_fun(pred_y.view(-1,classes),y.to(torch.long).view(-1))
    
    for i in range(5):
        loss_out = loss_fun(pred_y.view(-1,classes),y.to(torch.long).view(-1))
        loss = torch.cat((loss,loss_out.unsqueeze(0)),0)
        y = left_shift(y,1)
    loss = torch.mean(torch.min(loss,dim=0).values)
    
    return loss

class recommender(nn.Module):
    def __init__(self,anime_count:int,encode_dim:int):
        super(recommender, self).__init__()
        lstm_width = 200
        hidden1_width = 600
        hidden2_width = 400
        hidden3_width = 100
        encode_dim = 15

        self.embedding = nn.Embedding(anime_count,encode_dim,padding_idx=0)
        self.lstm = nn.LSTM(input_size=hidden1_width,hidden_size=lstm_width,batch_first=True)
        self.hidden1 = nn.Linear(3+6+encode_dim,hidden1_width)
        self.hidden3 = nn.Linear(hidden1_width,hidden1_width)
        self.hidden2 = nn.Linear(lstm_width,hidden2_width)
        #self.hidden4 = nn.Linear(hidden2_width,hidden3_width)
        self.norm = nn.LayerNorm(hidden1_width)
        self.anime_out = nn.Linear(hidden2_width,anime_count)
        self.encodings = None
        self.titles = ["anime_id","my_score","my_status","my_watched_episodes"]
        self.anime_count = anime_count
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(.3)

    def forward(self,show:torch.Tensor, rating:torch.Tensor, status:torch.Tensor, episodes:torch.Tensor,length:torch.Tensor):
        #change shape of input vectors
        status = nn.functional.one_hot(status.to(torch.int64),6)
        show = self.embedding(show.to(torch.int64))
        unrated = torch.unsqueeze(torch.eq(rating,0).int(),-1)
        #year = torch.unsqueeze(year,-1)
        episodes = torch.unsqueeze(episodes,-1)
        rating = torch.unsqueeze(rating,-1)

        x = torch.concat((show,rating,status,episodes,unrated),-1)
        x = self.dropout(x)
        x = self.relu(self.hidden1(x))
        x = self.norm(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout(x)
        p = pack_padded_sequence(x,length,batch_first=True,enforce_sorted=False)
        x,(_,_)= self.lstm(p)
        x, length = pad_packed_sequence(x,batch_first=True)
        x = self.relu(self.hidden2(x))
        #x = self.relu(self.hidden4(x))
        x = self.dropout(x)
        x = self.anime_out(x)
        #x = self.softmax(x)
        return x

    def train_step(self,optimizer,inputs,lengths):

        losses = []
        batch_num = 0
        for show, rating, status, episodes,length in zip(*[inputs[i] for i in self.titles],lengths):
            optimizer.zero_grad()
            outputs = self(show,rating,status,episodes,length)
            show = pack_padded_sequence(show,length,batch_first=True,enforce_sorted=False)
            show, _ = pad_packed_sequence(show,batch_first=True)
            label = show.to(torch.int64)
            loss = series_loss(outputs,label)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if batch_num % 50 ==0 :
                print('batch {} loss: {}'.format(batch_num,loss.item()))
            batch_num+=1
        print("average loss ",sum(losses)/len(losses))
        return sum(losses)/len(losses)

    def validate(self,inputs,lengths):
        losses = []
        for show, rating, status, episodes,length in zip(*[inputs[i] for i in self.titles],lengths):
            with torch.no_grad():
                outputs = self(show,rating,status,episodes,length)
                show = pack_padded_sequence(show,length,batch_first=True,enforce_sorted=False)
                show, _ = pad_packed_sequence(show,batch_first=True)
                label = show.to(torch.int64)
                loss = series_loss(outputs,label)

                losses.append(loss.item())
        return sum(losses)/len(losses)

    # inputs should be a dict of a single Tensor from one list    
    def predict(self, inputs : dict):
        length = torch.Tensor([len(inputs[self.titles[0]][0])])
        with torch.no_grad():
            return self(*[inputs[i] for i in self.titles],length)

        
def load_model(file,model):
    return model.load_state_dict(torch.load(file))

def dict_map(input,*args: dict) -> np.array:
    input = np.array(input)
    shape = input.shape
    input = input.flatten()
    for dict_map in args:
        input = list(map(lambda i: dict_map.get(i,0),input))
    
    return np.reshape(input,shape)

if __name__ == "__main__":
    df, studios, other_studios, genres = t.load_animes(os.path.join("data","anime_cleaned.csv"))

    anime_ids = {v:k+1 for k,v in df["anime_id"].sort_values().to_dict().items()}
    backwards_anime_ids = {k+1:v for k,v in df["anime_id"].to_dict().items()}
    anime_lookup = dict(zip(df["anime_id"],df["title"]))

    anime_count = len(anime_lookup.values())
    print(anime_ids)
    def process_data(df):
        df = t.process_data(df)
        df.loc[:,"anime_id"] = df["anime_id"].map(lambda i: anime_ids.get(i,0))
        return df
    
    model_location = os.path.join("models","recommender3")

    titles = ["anime_id","my_watched_episodes","my_score","my_status"]

    train = False
    if train:
        dfs = []
        insert_fn = lambda i: dfs.append(i)
        iterations = 400
        total_lines = 10000
        for i in range(iterations):
            print("iteration: ",i)
            #t.load_anime_lists(os.path.join("data","UserAnimeList.csv"),i,fn=process_data,insert_fn=insert_fn,lines=total_lines)
            t.load_anime_lists(os.path.join("new_data","animelist.csv"),i,
                fn=lambda df: df.loc[df["my_last_updated"]>0],insert_fn=insert_fn,sort=False,lines=total_lines)

        inputs = {}
        test_inputs = {}
        batch_size = 10
        total = len(dfs)
        batch_num = int(total/batch_size)
        dfs = dfs[:int(batch_num)*batch_size]
        test_batch_num = 4

        batch_num-=test_batch_num

        print(dfs[0])
        print(len(dfs))
        for title in titles:
            values = [torch.Tensor(i[title].to_numpy()).to(device) for i in dfs[:-test_batch_num*batch_size]]
            padded = pad_sequence(values,batch_first=True,padding_value=0)
            inputs[title] = padded.view(batch_num,-1,padded.shape[-1])
            print(inputs[title])
        seq_len = torch.Tensor(list(map(len,values)))
        seq_len = seq_len.view(batch_num,-1)
        print(seq_len)
        

        for title in titles:
            values = [torch.Tensor(i[title].to_numpy()).to(device) for i in dfs[-test_batch_num*batch_size:]]
            padded = pad_sequence(values,batch_first=True,padding_value=0)
            test_inputs[title] = padded.view(test_batch_num,-1,padded.shape[-1])
        

        test_seq_len = torch.Tensor(list(map(len,values)))
        test_seq_len = test_seq_len.view(test_batch_num,-1)
        

        #print(inputs["my_score"])
        #convert to batches

        rec = recommender(anime_count,20).to(device)
        epochs = 50
        test_batches = 2
        optim = torch.optim.Adam(rec.parameters(),lr=0.001)
        last = 0
        losses = []
        neg_num = 0
        for i in range(epochs):
            print("epoch: ",i)
            if i %100 ==0:
                print(i)
        
            current = rec.train_step(optim,inputs,seq_len)
            
            test = rec.validate(test_inputs,test_seq_len)
            print("test loss: ",test)
            print("diff: ",test-last)
            #early stopping
            if last != 0 and test>last:
                neg_num+=1
                if neg_num>1:
                    break
            else:
                neg_num=0
            last = test
            
            losses.append(test)
        torch.save(rec.state_dict(),model_location)
        plt.plot(losses)
        plt.show()
        
    else:
        rec = recommender(anime_count,20)
        rec.load_state_dict(torch.load(model_location))
        device = "cpu"
        rec.to(device)
        dfs = []
        insert_fn = lambda i: dfs.append(i)
        t.load_anime_lists(os.path.join("data","UserAnimeList.csv"),16,fn=process_data,insert_fn=insert_fn,lines=20000)
        inputs = {} 
        for title in titles:
            inputs[title] = torch.Tensor(np.expand_dims(dfs[2][title].to_numpy(),0)).to(device)

        #rec.encode_all()
        output = rec.predict(inputs)
        print(output)
        #output_id = rec.decode(output,topk=3)
        output_id = torch.topk(output,dim=-1,k=3).indices
        input_mapped = dict_map(inputs["anime_id"],backwards_anime_ids,anime_lookup)
        for i, score,ep in zip(input_mapped[0],inputs["my_score"][0],inputs["my_watched_episodes"][0]):
            print(i,int(score),int(ep))
        output_mapped = dict_map(output_id,backwards_anime_ids,anime_lookup)
        for i in output_mapped[0]:
            print(i)
