#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt


# In[251]:


def StartDf(Ticker): #Inicializar dataframe de preços e remover dados inválidos
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2017, 1,1)
#     df = web.DataReader(Ticker, 'yahoo', start, end)
#     df = df.dropna()
#     df = df.to_csv(str(Ticker)+'.csv')   
    df = pd.read_csv(str(Ticker)+'.csv')
    return df


# In[165]:


def CreatePriceAndPreview(df,Length):#Dataframe em que cada linha contém Length preços subsequentes de df
    List3 = []
    for n in range (Length, len(df)):
        List2 = []
        for i in range(n-Length,n):
            List2.append(df['Adj Close'][i])
        List3.append(List2)
    return pd.DataFrame(List3)


# In[166]:


#determinar o sinal relativo ao j-ésimo dia
def CreateSignalColumn(df,Length,j):
    ListSignal = []
    for i in range(0,len(df)-1):
        if df[Length-(Length-j)][i] < df[Length-1][i+1]:
            ListSignal.append(1) #sinal = 1, preço da ação irá ser maior em relação ao j-ésimo dia
        else:
            ListSignal.append(0)#sinal = 0, preço da ação irá ser menor em relação ao j-ésimo dia
    return ListSignal


# In[167]:


#Inserir coluna, apagar última linha
def UpDateDataFrame(df,Length):
    PriceAndPreview = CreatePriceAndPreview(df,Length)
    ListSignal = []
    for i in range(0,Length):
        ListSignal.append(CreateSignalColumn(PriceAndPreview,Length,i))  
    PriceAndPreview = PriceAndPreview.drop([len(PriceAndPreview)-1])
    for i in range(0,Length):
        PriceAndPreview['Signal'+str(i)] = ListSignal[i]
    return PriceAndPreview


# In[168]:


#Treinar e calcular a acurácia usando SVM
def SVMScore(X_train,y_train,X_test,y_test):
    model = SVC(kernel='linear')
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    return model.score(X_test,y_test)


# In[169]:


#Treinar e calcular a acurácia usando MLP
def MLPScore(X_train,y_train,X_test,y_test):
    mlp = MLPClassifier(solver='lbfgs',max_iter=500,alpha=1e-5,random_state=1)
    mlp.fit(X_train,y_train)
    pred = mlp.predict(X_test)
    return accuracy_score(y_test,pred)


# In[170]:


#Treinar e calcular a acurácia usando KNN
def KnnScore(X_train,y_train,X_test,y_test,k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)


# In[171]:


#Determinar o score dos modelos para um conjunto de treinamento e teste dado
def DictModelsAndPred(X_train,y_train,X_test,y_test): 
    predSVM = SVMScore(X_train,y_train,X_test,y_test)    
    predMLPScore = MLPScore(X_train,y_train,X_test,y_test)
    predKnn = KnnScore(X_train,y_train,X_test,y_test,6)
    DictPred = {'SVM':predSVM,'MLP':predMLPScore,'KNN':predKnn}
    return DictPred


# In[172]:


#Para cada dia, determinar o modelo com melhor acurácia e os valores previstos por cada um
def DictScores(X,Length,PriceAndPreview):
    DictScores = {'SVM':[],'MLP':[],'KNN':[]}
    for i in range(0,Length):
        y = PriceAndPreview['Signal'+str(i)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)             
        DictPred = DictModelsAndPred(X_train, y_train,X_test,y_test)
        for Model in DictScores:
            DictScores[Model].append(DictPred[Model])
    return DictScores


# In[173]:


#Retorna a acurácia de cada modelo para cada dia
def FuncAccuracyMatrix(ticker,Length):
    df = StartDf(ticker)
    PriceAndPreview = UpDateDataFrame(df,Length)#Create a dataframe with the last five days price, Add signal column and delete last line
#     display(PriceAndPreview)
    ListDrop = []
    for i in range(0,Length):
        ListDrop.append('Signal'+str(i))
    X = PriceAndPreview.drop(ListDrop,axis='columns')#Save dataframe without the Signal column on X variable   
    return pd.DataFrame(DictScores(X,Length,PriceAndPreview))    


# In[189]:


#Funções para calcular os valores previstos por cada modelo
def PredSVM(X_train,y_train,X_test):
    model = SVC(kernel='linear')
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    return pred
def PredMLP(X_train,y_train,X_test):
    mlp = MLPClassifier(solver='lbfgs',max_iter=500,alpha=1e-5,random_state=1)
    mlp.fit(X_train,y_train)
    pred = mlp.predict(X_test)
    return pred
def PredKnn(X_train,y_train,X_test,k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred


# In[228]:


#Calcular o valor previsto por um modelo dado
def PredModel(X_train, X_test, y_train, Model):           
    if(Model=='SVM'):
        return PredSVM(X_train,y_train,X_test)
    elif(Model=='MLP'):
        return PredMLP(X_train,y_train,X_test)
    elif(Model=='KNN'):
        return PredKnn(X_train,y_train,X_test,5)


# In[191]:


#Construir um dataframe com os valores de test e as predições de cada dia dos modelos que melhor performaram
def CreateBestPredMatrix(AccuracyMatrixI, ticker, Length):
    Max = AccuracyMatrixI.idxmax(axis=1)
    df = StartDf(ticker)
    PriceAndPreview = UpDateDataFrame(df,Length)
    ListDrop = []
    for i in range(0,Length):
        ListDrop.append('Signal'+str(i))
    X = PriceAndPreview.drop(ListDrop,axis='columns') 
    DictPred = {}
    for i in range(0,Length):
        y = PriceAndPreview['Signal'+str(i)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)   
        DictPred['Pred'+str(i)] = PredModel(X_train, X_test, y_train, Max[i])
    X_test = X_test.reset_index(drop=True)     
    p = pd.DataFrame(DictPred)
    p = X_test.join(p)
    pred = GenerateCombineSignal(p,Length)
    return p


# In[192]:


def Signal(BestPred,Length,List,i):
    for j in range(0,Length-1):    
        if(BestPred.loc[i][j] > BestPred.loc[i][Length-1] and BestPred.loc[i]['Pred'+str(j)]==1):
            return 1  
        if(BestPred.loc[i][j] < BestPred.loc[i][Length-1] and BestPred.loc[i]['Pred'+str(j)]==0):
            return 0 
        else:
            return BestPred.loc[i]['Pred'+str(Length-1)]


# In[193]:


def GenerateCombineSignal(BestPredMatrix ,Length):
    List = [0]
    S = 0
    for i in range(0,len(BestPredMatrix)):
        S = Signal(BestPredMatrix,Length,List,i)
        List.append(S)
    List.pop(0)
    return List


# In[194]:


def GenerateRandomBinaryList(Length):
    randomlist = []
    for i in range(0,Length):
        n = random.randint(0,1)
        randomlist.append(n)
    return randomlist
def GenerateBuyAndHoldList(Length):
    List = []
    for i in range(0,Length-1):
        List.append(1)
    List.append(0)
    return List


# In[195]:


def StartSimulationMatrix(BestPredMatrix ,Length):
    List=[Length-1]
    for i in range(0,Length):
        List.append('Pred'+str(i))
    return BestPredMatrix[List]


# In[196]:


def GenerateSimulationMatrix(BestPredMatrix,Length):    
    SimulationMatrix = StartSimulationMatrix(BestPredMatrix ,Length)
    SimulationMatrix['Rand'] = GenerateRandomBinaryList(len(SimulationMatrix))
    SimulationMatrix['NewSignal'] = GenerateCombineSignal(BestPredMatrix,Length)
    SimulationMatrix['BuyAndHold'] = GenerateBuyAndHoldList(len(SimulationMatrix))
    return SimulationMatrix


# In[197]:


def DailyOrderUpdate(Dict,Order,i,Price):
    if(Dict['PositionI'][i] == 0 and  Order[i] == 1):
        Dict['PositionF'].append(1)
        Dict['AtiviesF'].append(Dict['DollarI'][i]/Price)
        Dict['DollarF'].append(0)
    elif(Dict['PositionI'][i] == 1 and  Order[i] == 0):
        Dict['PositionF'].append(0)
        Dict['DollarF'].append(Dict['AtiviesI'][i]*Price)
        Dict['AtiviesF'].append(0)
    else:
        Dict['PositionF'].append(Dict['PositionI'][i])
        Dict['AtiviesF'].append(Dict['AtiviesI'][i])
        Dict['DollarF'].append(Dict['DollarI'][i])
    Dict['Return'].append(Dict['DollarF'][i]+Dict['AtiviesF'][i]*Price)
    


# In[217]:


def Experiment(BestPredMatrix ,Length,SimulationMatrix):     
    Dict = {'PositionI':[],'AtiviesI':[],'DollarI':[],'PositionF':[],'AtiviesF':[],'DollarF':[],'Return':[]}
    Dict['PositionI'].append(0)
    Dict['AtiviesI'].append(0)
    Dict['DollarI'].append(100)
    
    Order = SimulationMatrix['Pred3']
    for i in range(0,len(SimulationMatrix)):
        DailyOrderUpdate(Dict,Order,i,SimulationMatrix[Length-1][i])
        Dict['PositionI'].append(Dict['PositionF'][i])
        Dict['AtiviesI'].append(Dict['AtiviesF'][i])
        Dict['DollarI'].append(Dict['DollarF'][i])   
    Dict['PositionI'].pop()
    Dict['AtiviesI'].pop()
    Dict['DollarI'].pop()
    ExperimentDf = pd.DataFrame(Dict)
    ExperimentDf['Price'] = SimulationMatrix[Length-1]
    return ExperimentDf


# In[233]:


def BuildExperiment(Ticker,Length):
    AccuracyMatrixI = FuncAccuracyMatrix(Ticker,Length)
    BestPredMatrix = CreateBestPredMatrix(AccuracyMatrixI,Ticker,Length)    
    SimulationMatrix = GenerateSimulationMatrix(BestPredMatrix ,Length)  
    return Experiment(BestPredMatrix ,Length,SimulationMatrix)


# In[240]:


BuildExperiment('AMZN',5)


# In[202]:


def GeneratePlots(Ticker,Length):
    AccuracyMatrixI = FuncAccuracyMatrix(Ticker,Length)
    AccuracyMatrixI.idxmax(axis=1)
    display(AccuracyMatrixI)    
    AccuracyMatrixI.plot(kind = 'bar')
    plt.title(str(Ticker)+'  '+str(Length)+' dias')
    plt.show()


# In[232]:


Tickers = ['GOOGL','AMZN','AAPL','FB']
for Ticker in Tickers:
    GeneratePlots(Ticker,7)

