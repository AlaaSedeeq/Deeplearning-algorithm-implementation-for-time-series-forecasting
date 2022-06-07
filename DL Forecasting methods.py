#Basic libraries
import pandas as pd
import re
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt 
import warnings
import datetime
from warnings import filterwarnings, catch_warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)               

#DL libraries
import keras
from keras import Model, regularizers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Input, concatenate, Conv1D, MaxPooling1D, MaxPooling1D, ConvLSTM2D,
                          Flatten, BatchNormalization, LSTM, Bidirectional, TimeDistributed,)

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

from evaluations import nrmse, mape



class DL_Forecaste:
    """
    This class creates and train a DL model for univariate TS.
    ...
    Attributes
    ----------
    training : Pandas dataframe
    Data to fit the model with.
    
    algorrithm : str
    Type of DL algorithm (MLP, CNN, LSTM)
    
    lstm_method : str
    If using LSTM model, you need to specify the type of LSTM. 
    (Vanila, Stacked, Bidirectional LSTM, CNN-LSTM, ConvLSTM)
    """
    def __init__(self, training, algorithm, lstm_method=None):
        # save sequence, index
        self.sequence = training.values
        self.index = training.index
        self.algorithm = algorithm
        self.lstm_method = lstm_method
        if self.lstm_method:
            self.lstm_method = re.sub(r'[^\w]', ' ', lstm_method).lower()
            if self.lstm_method not in ['vanila', 'stacked','bidirectional','cnn','convlstm']:
                raise ValueError('Invalid {}. Expecting on of (Vanila, Stacked, Bidirectional LSTM, CNN-LSTM, ConvLSTM).'\
                                 .format(self.lstm_method))

    # split a univariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequence(self, data, train_test=False):
        """
        Docstring:
        This function transforms the raw dataset(TS) into inputs(Features) and output.
        ...
        Attributes
        ----------
        sequence : Pandas DataFrame
        The data you want to transform.
        
        w_size : integar number
        The size of applied window.
        
        ret : bool
        If True, the functions returns the transformed features.
        Returns
        ----------
        Two numpy arrays(X, y).
        """
        self.sequence = data
        # create our dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.sequence)
        # apply window function to the dataset to convert the sequence to features and target
        dataset = dataset.window(self.w_size, shift=1, drop_remainder=True)
        # flatten the dataset to numpy array to start dealing with it
        dataset = dataset.flat_map(lambda window: window.batch(self.w_size))
        # shuffling and splitting the dataset into X,y
#         dataset = dataset.shuffle(50)
        dataset = dataset.map(lambda window: (window[:self.n_steps_in], window[-self.n_steps_out:]))
        # split into x, y
        self.X = np.array([x.numpy() for x, _ in dataset])
        self.y = np.array([y.numpy() for _, y in dataset])
        
#         # reshape from [samples, timesteps] into [samples, timesteps, features]
#         self.n_features = 1
#         self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        
        if self.lstm_method=='cnn lstm':
            # A CNN model can be used in a hybrid model with an LSTM backend where the CNN is used to interpret 
            # subsequences of input that together are provided as a sequence to an LSTM model to interpret. 
            # Split the input sequences into subsequences to be handeled by the CNN (w_size = n_input + 1)
            # Each sample can then be split into two sub-samples, each with n time steps.(w_size=n_seq*n_steps)'one sample'
            # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
            self.X = self.X.reshape((self.X.shape[0], self.n_seq, self.n_steps, self.n_features))
            
        elif self.lstm_method=='convlstm':
            # The ConvLSTM was developed for reading two-dimensional spatial-temporal data, but can be adapted for
            # use with univariate time series forecasting. 
            # The layer expects input as a sequence of two-dimensional data, therefore the shape of input data must be:
            # [samples, timesteps, rows, columns, features].
            self.X = self.X.reshape((self.X.shape[0], self.n_seq, 1, self.n_steps, self.n_features))
            
        # scaling the data 
        if self.is_scale:
            self.X, self.y = self.scale_data(self.X, self.y)
        
        # Split data using train proportion of (70% - 30%)
        if train_test:
            self.split_point = int(0.7 * len(self.X)) 
            self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
            self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]            
            return self.X_train, self.X_test, self.y_train, self.y_test
        
        else:
            return self.X, self.y
    
    def scale_data(self, X, y):
        # scale x
        self.scaler_features = MinMaxScaler().fit(X.reshape(-1, 1))
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1)).reshape(np.shape(X))
        # scale y
        self.scaler_label = MinMaxScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1)).reshape(np.shape(y))
        # return scaled fvalues
        return self.scaled_features, self.scaled_label

    def build(self, n_steps_in, n_steps_out, units=100, batch=256, kernel_size=4, lr=0.03, epochs=100, 
              is_scale=False, verbose=0, filter_num=64, pool_size=2, n_seq = 2, n_steps=2, convlstm_kernel_size=(1,2)):

        # define model parameters
        self.lr = lr
        self.units = units
        self.batch = batch
        self.epochs = epochs
        self.is_scale = is_scale
        self.n_seq = n_seq
        self.n_steps = n_steps #int(self.w_size-1/self.n_seq)
        self.n_steps_in =  n_steps_in
        self.n_steps_out = n_steps_out
        self.w_size = self.n_steps_in + self.n_steps_out
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.convlstm_kernel_size = convlstm_kernel_size
        self.n_features =  1 #len(self.data.columns)

        # split the sequence into samples
        self.X_train, self.X_test, self.y_train, self.y_test=self.split_sequence(self.sequence, train_test=True)       
        # Build the model
        if self.algorithm.lower()=='mlp':
            self.model = Sequential()
            self.model.add(BatchNormalization())
            self.model.add(Dense(units=self.units,input_dim=self.n_steps_in,kernel_initializer='normal', activation='sigmoid'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.units, kernel_initializer='normal', activation='sigmoid'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.units//2, kernel_initializer='normal', activation='sigmoid')) 
            self.model.add(Dense(self.n_steps_out))
        
        elif self.algorithm.lower()=='cnn':
              
            self.model = Sequential()
            self.model.add(Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, activation='relu',
                                  kernel_initializer='normal', input_shape=(self.n_steps_in, self.n_features)))
            self.model.add(MaxPooling1D(pool_size=self.pool_size))
            self.model.add(Flatten())
            self.model.add(BatchNormalization())
            self.model.add(Dense(units=self.units, kernel_initializer='normal', activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.units, kernel_initializer='normal', activation='relu')) 
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.units//4, kernel_initializer='normal', activation='relu')) 
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            #         self.model.add(Dense(units=self.units/2, kernel_initializer='normal', activation='relu')) 
            #         self.model.add(Dropout(0.2))
            self.model.add(Dense(self.n_steps_out))
              
        
        elif (self.algorithm.lower()=='lstm' and self.lstm_method=='vanila'):
            
            self.model = Sequential()
            self.model.add(LSTM(self.units, input_shape=(self.n_steps_in, self.n_features), activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dense(self.n_steps_out))

        elif (self.algorithm.lower()=='lstm' and self.lstm_method=='stacked'):
                  
            # Build the model
            self.model = Sequential()
            self.model.add(LSTM(self.units, input_shape=(self.n_steps_in, self.n_features), 
                                return_sequences=True, activation='relu'))
            self.model.add(LSTM(self.units//2, activation='relu', return_sequences=True))
            self.model.add(LSTM(self.units//2, activation='relu'))
            self.model.add(Dense(self.units, activation='relu'))
            self.model.add(Dense(10, activation='relu'))
            self.model.add(Dense(self.n_steps_out))

        elif (self.algorithm.lower()=='lstm' and self.lstm_method=='bidirectional'):
            
            # Build the model
            self.model = Sequential()
            self.model.add(Bidirectional(LSTM(self.units,activation='relu'),input_shape=(self.n_steps_in,self.n_features)))
            self.model.add(Dense(self.n_steps_out))
            
        elif (self.algorithm.lower()=='lstm' and self.lstm_method=='cnn lstm'):
              
            # define the input cnn model
            self.model = Sequential()
            # TimeDistributed allows to apply a layer to every temporal slice of an input.
            # So, we use TimeDistributed to apply the same Conv2D layer to each timestep. (sharable parameters)
            self.model.add(TimeDistributed(Conv1D(filters=self.filter_num,kernel_size=self.kernel_size,activation='tanh'), 
                                           input_shape=(None, self.n_steps, self.n_features)))
#                                            (None,((self.n_steps_in)//(self.n_seq*self.n_steps)),self.n_features)))
            self.model.add(TimeDistributed(MaxPooling1D(pool_size=self.pool_size, padding='same')))
            self.model.add(TimeDistributed(Flatten()))
            # define the output model
            self.model.add(LSTM(self.units, activation='tanh'))
            self.model.add(Dense(self.units, activation='relu'))
            self.model.add(Dense(self.n_steps_out))
                    
        elif (self.algorithm.lower()=='lstm' and self.lstm_method=='convlstm'):

            # define model
            self.model = Sequential()
            self.model.add(ConvLSTM2D(filters=self.filter_num, kernel_size=self.convlstm_kernel_size, activation='relu', 
                                      input_shape=(n_seq, 1, self.n_steps, self.n_features)))
            self.model.add(Flatten())
            self.model.add(Dense(self.n_steps_out)) 
              
        # callback
        callback = [keras.callbacks.EarlyStopping(patience=5, verbose=0),    #EarlyStopping
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.00001, verbose=0)] #lr Decaying
        
        # define the optimizer
        opt = Adam(learning_rate=self.lr)
        # Compile model
        self.model.compile(optimizer=opt, loss='mse')
        # fit data to the model
        self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                      validation_data=(self.X_test, self.y_test),
                                      batch_size=self.batch, 
                                      callbacks=callback, 
                                      epochs=self.epochs, 
                                      verbose=verbose)
    
    def forecast(self, data):
        X, y = self.split_sequence(data.values)
        # predict
        # (num ot testing samples, shape of single training instance)
        X_shape = sum(((-1,), self.X_train.shape[1:]), ()) 
        pred = self.model.predict([X.reshape(X_shape)]).flatten()
        #rescale
        if self.is_scale:    
            pred = self.scaler_label.inverse_transform([pred.flatten()])
        # save predicted values 
        pred = pd.Series(pred.flatten(), index=data.index[-self.n_steps_out:])
        return pred
