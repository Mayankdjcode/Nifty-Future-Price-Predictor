def runModel():
    import yfinance as yf
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from datetime import datetime
    from datetime import timedelta
    import time

    stock_symbol = '^NSEI'

    data = yf.download(tickers=stock_symbol,period='7d',interval='1m')
    opn = data[['Open']] # Opening price of the stock
    ds = opn.values      # Opening price values alloted to 'ds'(data set) variable
    print(len(ds))
    time.sleep(10)
    normalizer = MinMaxScaler(feature_range=(0,1))                   # normalizer (normalizes value between 0 and 1)
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1)) # normalizing the opening price values , reshaping dimensions of the array

    train_size = int(len(ds_scaled)*0.70) # 70 % data is for training
    test_size = len(ds_scaled) - train_size # 30 % is for testing

    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1] 

    def create_ds(dataset,step):                    # Function for Creating database, the values are incremented at each step by step.
        Xtrain, Ytrain = [], []                     # and fed into it, we get a continuous Price-Time graph
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)   # In both X-Y dimensions of the graph

    time_stamp = 100
    X_train, y_train = create_ds(ds_train,time_stamp) #for training 
    X_test, y_test = create_ds(ds_test,time_stamp)    #for testing

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1) 
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1))) #LSTM,RNN multilayered . 50 units used.
    model.add(LSTM(units=50,return_sequences=True))                                  
    model.add(LSTM(units=50))                                  
    model.add(Dense(units=1,activation='linear')) # Dense layer is the regular deeply connected neural network layer, output = activation(dot(input, kernel) + bias),1 output
    model.compile(loss='mean_squared_error',optimizer='adam') #Adam optimizer is the extended version of stochastic gradient descent which could be implemented in various deep learning applications
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1,batch_size=64) # 1 iteration, the number of training examples utilized in one iteration=64

    loss = model.history.history['loss']

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = normalizer.inverse_transform(train_predict) #inverse_transform(X)[source] Undo the scaling of X according to feature_range. 
                                                                #Parameters: Xarray-like of shape (n_samples, n_features) Input data that will be transformed
    test_predict = normalizer.inverse_transform(test_predict)

    test = np.vstack((train_predict,test_predict)) #VSTACK returns the array formed by appending each of the array arguments in a row-wise fashion

    #plt.plot(normalizer.inverse_transform(ds_scaled))
    #plt.plot(test)

    fut_inp = ds_test[287:] #get past 500 minutes of data
    fut_inp = fut_inp.reshape(1,-1)

    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 3 minutes price using the current data
    lst_output=[]
    n_steps=500
    i=0
    print(fut_inp.shape)
    reShape = fut_inp.shape[1]
    while(i<10):
        if(len(tmp_inp)>500):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, 500, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, reShape,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    #print(lst_output)

    plot_new=np.arange(1,101)
    plot_pred=np.arange(101,111)

    ds_new = ds_scaled.tolist()

    ds_new.extend(lst_output)

    final_graph = normalizer.inverse_transform(ds_new).tolist()

    #Plotting final results with predicted value after 10 minutes
    #for i in range(len(final_graph)):
    #    final_graph[i] = [float(*final_graph[i])+45]
    
    final_graph[len(final_graph)-1] = [float(*final_graph[len(final_graph)-1])-(final_graph[len(final_graph)-1]-ds[-1])]
    #print(final_graph)
    plt.plot(final_graph,)
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.title("{0} prediction of next 3mins price".format(stock_symbol))
    now = datetime.now()
    now_plus_3 = now + timedelta(minutes = 3)
    plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'At time {0}, predicted price: {1}'.format(now_plus_3.strftime("%H:%M:%S"),round(float(*final_graph[len(final_graph)-1]),2)))
    #plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 3minutes: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    runModel()