import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import training as t

# loss function should consider that it exact order isn't the most important
def left_shift(mat,x):
    # left shifts the second dimension if a matrix leaving zeros
    return tf.concat((mat[:,x:],tf.zeros(tf.shape(mat))[:,:x]),axis=1)
    
def rnn_cossim(true,predicted):
    cossim = keras.losses.CosinSimilarity()
    loss = 0
    for i in range(3):
        loss += tf.math.pow(.5,i+1)*cossim(true,predicted)
        true =left_shift(true,1)
    return loss/3


class recommender(tf.keras.Model):
    '''
    def __init__(self,anime_count,encode_dim):
        super(recommender,self).__init__()
        self.embedding = layers.Embedding(anime_count,encode_dim); 

        info = keras.Input(shape=[None, 20],name="info_in")
        show = keras.Input(shape=[None],name="show_in")
        rating = keras.Input(shape=[None,1],name="ratin_in")

        x = self.embedding(show); 
        x = layers.Concatenate()([info,x])
        #from gpt
        x2 = layers.MultiHeadAttention(5,10)(x, x)
        x3 = layers.MultiHeadAttention(5,10)(rating, x)
        x = x+x2+x3
        x = layers.LayerNormalization()(x)
        x2 = layers.Dense(120,activation="relu")(x)
        x2 = layers.Dense(120)(x2)
        x = x + x2
        x = layers.LayerNormalization()(x)

        out_rating = layers.Dense(1,name="rating_out")(x)
        out_embedding = layers.Dense(encode_dim,name="embedding_out")(x)

        self.model = keras.Model(
            inputs=[info,show,rating],
            outputs=[out_embedding,out_rating]
        )
    def call(self,inputs):
        return self.model(inputs)

    def compile(self,optimizer):
        super(recommender, self).compile()
        self.optimizer = optimizer

    def training_step(self, data):
        embedding, rating = self.model(data)
        
        pass
  '''
    def __init__(self,anime_count,encode_dim):
        super(recommender,self).__init__()
        self.embedding = layers.Embedding(anime_count,encode_dim); 
        
        inputs = keras.Input(shape=[None,6],dtype=tf.int32)
        new_show =self.embedding(inputs[:,:,0])
        new_status = tf.one_hot(inputs[:,:,3],5)
        x = layers.Concatenate(-1)([tf.cast(new_show,tf.float32),tf.cast(inputs[:,:,1:2],tf.float32),tf.cast(inputs[:,:,4:5],tf.float32),tf.cast(new_status,tf.float32)])

        '''
        #info = keras.Input(shape=[None, 20],name="info_in")
        show = keras.Input(shape=[None],name="show_in")
        rating = keras.Input(shape=[None,1],name="rating_in",dtype=tf.uint8)
        year = keras.Input(shape=[None,1],name="year_in")
        status = keras.Input(shape=[None],name="status",dtype=tf.uint8)
        episodes_watched = keras.Input(shape=[None,1],name="episodes")
        new_show = self.embedding(show)
        new_status = tf.one_hot(status,5)
        # if show's rating is 0 then 1
        unrated = keras.Input(shape=[None,1],name="unrated",dtype=tf.uint8)
        x = layers.Concatenate(-1)([tf.cast(new_show,tf.float32),tf.cast(rating,tf.float32),tf.cast(unrated,tf.float32), year,tf.cast(new_status,tf.float32),episodes_watched])
        '''
        x = layers.LSTM(8,return_sequences=True)(x)
        

        out_embedding = layers.Dense(encode_dim,name="embedding_out")(x)

        self.model = keras.Model(
            inputs=inputs,
            outputs=out_embedding
        )

    def call(self,inputs):
        return self.model(inputs)

    def compile(self,optimizer):
        super(recommender, self).compile()
        self.optimizer = optimizer
      

if __name__ == "__main__":
  
    df, studios, other_studios, genres = t.load_animes(os.path.join("myanimelist","data","anime_cleaned.csv"))
    anime_lookup = dict(zip(df["anime_id"],df["title"]))
    print(anime_lookup[21])
    anime_count = len(df.index)
  
    dfs = t.load_anime_lists(os.path.join("myanimelist","data","UserAnimeList.csv"),0,lines=20000)
    arr = [list(i.drop(["username","my_last_updated"],axis=1).to_numpy()) for i in dfs]

    rec = recommender(anime_count,20)
    rec.model.summary()
    data = tf.ragged.constant(arr)
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    print('starting training')
    rec.compile(optimizer="Adam",loss=rnn_cossim,metrics=["mae", "acc"])
    history = rec.fit(data,data,batch_size=5)

    #inputs = {"show_in":arr[:,1],"rating_in":arr[:,.3],"year_in":arr[:,7],"status":arr[:,4],"episodes":arr[:,2]}
    #print(inputs)

    #rec.fit(epochs=2)

    """
    layer = layers.MultiHeadAttention(num_heads=2, key_dim=5)
    target = keras.Input(shape=[2, 16])
    target2 = keras.Input(shape=[2])
     
    x = layers.Embedding(2000,100)(target2)
    x = layers.Concatenate()([target,x])

    output_tensor, weights = layer(x, x, return_attention_scores=True)
    print(output_tensor.shape)
    """
