from keras.models import load_model
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

for i in range(9,-1,-1):
    # input data
    data_train = pd.read_csv("C:/Users/Yu/Desktop/Set_"+str(i)+"_Train.csv", index_col=0)
    data_test = pd.read_csv("C:/Users/Yu/Desktop/Set_"+str(i)+"_Test.csv",index_col = 0)
    #data_train = data_train.sort_values(['target_date','ticker'],ascending = True)
    #data_test = data_test.sort_values(['target_date','ticker'],ascending = True)
    data_train = data_train.reset_index(drop = True)
    data_test = data_test.reset_index(drop = True)
    #train data
    x_train = data_train.iloc[:, :-3]
    y_train = data_train.iloc[:, -3]
    y_train = y_train.values.reshape((y_train.shape[0],1))
    #test data
    x_test = data_test.iloc[:,:-3]
    y_test = data_test.iloc[:,-3]
    y_test = y_test.values.reshape((y_test.shape[0],1))

    # set parameters
    nSample_train = x_train.shape[0]
    nSample_test = x_test.shape[0]
    timestep = x_train.shape[1]
    features = 1

    x_train = x_train.values.reshape((nSample_train, timestep,features))
    x_test = x_test.values.reshape((nSample_test, timestep,features))

    #load model from single file
    model = load_model('lstm_model_'+str(i+1)+'.h5')

    # checkpoints & earlystopping
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                         save_weights_only=True, mode='min', period=1)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
    callbacks_list = [checkpoint, earlystop]

    # fit model
    history = model.fit(x_train, y_train, epochs=1000, callbacks= callbacks_list, shuffle=False, verbose=1, validation_split=0.2)

    # evaluate model
    loss, accuracy = model.evaluate(x_test, y_test)
    print ('loss = ', loss, '\naccuracy =', accuracy)

    # save model to single file
    model.save('lstm_model_'+str(i)+'.h5')

    # make predictions
    y_hat = model.predict_classes(x_test,verbose=0)
    y_prob = model.predict_proba(x_test,verbose=0)
    
    result = pd.DataFrame({'y_hat':y_hat.flatten(),'y_true':y_test.flatten(),'y_prob':y_prob.flatten(),\
                       'Date':data_test.iloc[:,-1],'Ticker': data_test.iloc[:,-2]},index = range(y_hat.size))
    
    result = result.sort_values(['Date','y_prob'],ascending = True)
    result = result.reset_index(drop = True)
    result.to_csv('result_'+str(i)+'.csv')
