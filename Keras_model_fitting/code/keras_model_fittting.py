import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

Base_path='D:\Data_sbk_fitting'
Data_path='data_found\\data'
Model_path='Keras_model_fitting\\model'
Result_path='Keras_model_fitting\\result'
All_data=[]

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(units=64, activation='relu', input_shape=(58,)))
    model.add(layers.Dense(units=64, activation='relu'))
    #default function
    model.add(layers.Dense(units=1))
    # mse as loss, mae sa metrics
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def data_process():
    #data preprocess to adapt the model
    #preparing the data
    data_path=os.path.join(Base_path,Data_path,'wettability_data_sbk_processed.csv')
    # read file
    df=pd.read_csv(data_path)
    df.reset_index(drop=True)
    df=df.loc[:, ~df.columns.str.match('Unnamed')]
    scalar = StandardScaler()
    y = df["receding angle"]

    #split the data (80% to train 20% to test)
    data_scaled = scalar.fit_transform(df.drop(columns=['receding angle']))
    x_train,x_test,y_train,y_test=train_test_split(data_scaled,y,test_size=0.20, random_state=470)

    # k-fold validation
    k = 4
    num_val_samples = len(x_train) // k
    num_epochs = 500
    all_val_mae_histories = []

    #load model
    model = build_model()

    #K-fold test
    for i in range(k):
        print('the number of fold is：' + str(i))
        # validation set
        val_data = x_train[i * num_val_samples:(i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples:(i + 1) * num_val_samples]

        # train set
        x_train_data = np.concatenate([x_train[:i * num_val_samples],
                                       x_train[(i + 1) * num_val_samples:]], axis=0)
        y_train_targers = np.concatenate([y_train[:i * num_val_samples],
                                          y_train[(i + 1) * num_val_samples:]], axis=0)

        history = model.fit(x=x_train_data,
                            y=y_train_targers,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs,
                            batch_size=1,
                            verbose=1)

        # keep average absolute value
        val_mae_history = history.history['val_mae']
        all_val_mae_histories.append(val_mae_history)

    #calculate the average value of k-flod score each turn
    average_mae_history = [np.mean([x[i] for x in all_val_mae_histories]) for i in range(num_epochs)]

    #plot validation score
    plt.plot(range(1 + 10, len(average_mae_history) + 1), average_mae_history[10:])
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.savefig(os.path.join(Base_path,Result_path,'Validation MAE.jpg'))
    plt.show()

    All_data.append(x_train)
    All_data.append(x_test)
    All_data.append(y_train)
    All_data.append(y_test)

def model_run(train_data,train_targets,test_data,test_targets):
    model = build_model()
    model.fit(x=train_data,
              y=train_targets,
              epochs=100,
              batch_size=16)

    # evaluate the model
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print('test_mse_score:' + str(test_mse_score))
    print('test_mae_score:' + str(test_mae_score))

    #linear regression
    y_pre = model.predict(test_data)
    linear = LinearRegression()
    y_test = np.array(test_targets).reshape(14,1)
    y_pre  = np.array(y_pre).reshape(14,1)
    linear.fit(y_test,y_pre)
    plt.scatter(y_test,y_pre,c='blue')
    pred_y = linear.predict(y_test)
    plt.plot(y_test,pred_y,c = 'red')
    plt.xlabel('Expremental receding angle', fontsize=20)
    plt.ylabel('Predicted receding angle', fontsize=20)
    plt.savefig(os.path.join(Base_path,Result_path,'Liner_regression.jpg'))
    plt.show()

    data={'y_test':y_test,'y_pre':y_pre}
    print(data)
    score=r2_score(y_test,pred_y)
    print('test_R2_score:'+str(score))

    # save model
    model.save(os.path.join(Base_path,Model_path))

if __name__=='__main__':
    data_process()
    model_run(All_data[0],All_data[2],All_data[1],All_data[3])
