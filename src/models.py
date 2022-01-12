
# Import user defined functions
import featuresExtraction as er
#import visualization as vis

import nlp_analysis as er

# Data manipulation
import pandas as pd

# Import Pipeline
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Import Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# to save model
import joblib

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

import umap
import matplotlib.pyplot as plt

def model_input(df):
    """
    Description:
    function is split input dataset into features and labels.
    
    Inputs:
    df - dataframe to convert to word2vector
    
    Outputs:
    X,y,combine - Features, Labels and Original datasets.
    """

    #Feature Extraction 
    word_tok_vec_avg,model,df1 = er.googleModel(df)

    y = pd.Series(df["sentiment_summary"])
    X = pd.DataFrame(word_tok_vec_avg)
    #df = pd.concat([df,pd.DataFrame(word_tok_vec_avg)])

    return X,y,df


def pipeline_model(X_train, X_test, y_train, y_test,inp_steps):
    """
    Description:
    function is to create pipeine for each model
    
    Inputs:
    X_train, X_test, y_train, y_test,inp_steps - Training and Testing datasets and steps like scaling, PCA etc.
    
    Outputs:
    preds,score,pl - predictions, score and model returned.
    """

    #dictionary for defining the steps

    steps = {
        'scaler':('scaler',StandardScaler()),
        'pca':('pca',PCA(n_components=4)),
        'logreg':('logreg',LogisticRegression()),
        'tree': ('tree',DecisionTreeClassifier()),
        'randfor' : ('randfor',RandomForestClassifier(n_estimators=200)),
        'gradboost':  ('gradboost',GradientBoostingClassifier(learning_rate=0.2,n_estimators=500,n_iter_no_change=0)),
        'svm': ('svm',SVC(gamma='scale',probability=True))
    }

    list_steps = []

    for k in steps.keys():
        if k in inp_steps:
            list_steps.append(steps[k])

    # Pipeline with scaler and Logistic regression
    pl = Pipeline(list_steps)

    # Train model with pipeline classifier
    pl.fit(X_train, y_train)

    # Make predictions on test data with trained model
    #preds = pl.predict_proba(X_test)
    preds = pl.predict(X_test)

    # score :  Predictions for X_test are compared with y_test and either accuracy (for classifiers) or R² score (for regression estimators) is returned
    score = pl.score(X_test, y_test)
    #score = pl.score(y_test, ps)
    
    return preds,score,pl

def model_altair(preds,X_test,combine):
    """
    Description:
    function is to get UMAP output for passing as input to visualize the model outputs
    
    Inputs:
    preds,X_test,combine - prediction value, original and testing datasets
    
    Outputs:
    em - umap output for visualization
    """
    em = pd.DataFrame()
    embedding = umap.UMAP(n_neighbors=10,min_dist=0.3,metric='correlation').fit_transform(X_test)
    preds = preds.tolist()
    em['x'] = embedding[:,0]
    em['y'] = embedding[:,1]
    #em['row'] = em.index
    em['color'] = preds
    em['post_type'] = combine.loc[X_test.index,'post_type'].values
    em['followers'] = combine.loc[X_test.index,'followers'].values
    em['sentiment_summary'] = combine.loc[X_test.index,'sentiment_summary'].values
    em['body'] = combine.loc[X_test.index,'body'].values
    em['clusters_0'] = combine.loc[X_test.index,'clusters_0'].values
  
    return em

def find_best_model(X, y,df):
    """
    Description:
    function is for using pipeline for finding the best model out of a list of models and give the results
    
    Inputs:
    X - features
    y - labels
    df - original dataset
    
    Outputs:
    model_result - results for all the models
    final_model - best model from the pipeline
    best_X_train, best_X_test, best_y_train, best_y_test - Training and testing datasets for modelling
    max_em - embedding for the best model
    """   
    models = {0: ['scaler','tree'],
          1:['scaler','randfor'],
          2: ['scaler','pca','tree'],
          3:['scaler','pca','randfor'],
          4:['scaler','gradboost'],
          5:['randfor'],
          6:['scaler','pca','gradboost'],
          7:['scaler','svm']}

    model_desc = {0: 'Decision Tree classifier', 1: 'Random Forest', 
                    2: 'Decision Tree classifier w/PCA', 3: 'Random Forest w/PCA',
                    4: 'Gradient Boosting', 5: 'Random Forest w/parameters',
                    6: 'Gradient Boosting w/parameters',7: 'SVM'}

    model_clf = {0: 'DecisionTreeClassifier()', 1: 'RandomForestClassifier()', 
                2: 'DecisionTreeClassifier()', 3: 'RandomForestClassifier()',
                4: 'GradientBoostingClassifier()', 5: 'RandomForestClassifier()',
                6: 'GradientBoostingClassifier()', 7: 'SVC()'}

    model_id = []
    model_name = []
    model_preds = []
    model_score = []
    model_precision = []
    model_recall = []
   # model_train_test = []
    model_f1score = []
    max_score = 0
    model_max = ''
    model_max_desc =""

    for i,each in enumerate(models):
    
        inp_steps = models[each]
        best_score = 0
        best_precision = 0
        best_pred = 0
        best_recall = 0
        best_f1score = 0
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        pred,score,pl = pipeline_model(X_train, X_test, y_train, y_test,inp_steps)
        
        em = model_altair(pred,X_test,df)
        
        precision,recall,f1score = confusionmatrix(em)
        
        if(round(score*100,2)>0.65 and round(score*100,2)>= best_score and precision>0.60 and precision>=best_precision):
            best_score  = round(score*100,2)
            best_pred = pred
            best_X_train, best_X_test, best_y_train, best_y_test = X_train, X_test, y_train, y_test
            best_model = pl
            best_precision,best_recall,best_f1score = precision,recall,f1score
            best_em = em

        model_id.append(i)
        model_name.append(model_desc[i])
        model_preds.append(best_pred)
        model_score.append(best_score)
        model_precision.append(best_precision)
        model_recall.append(best_recall)
        model_f1score.append(best_f1score)
        
        if(max_score<best_score):

            max_score = best_score
            model_max = i
            preds = best_pred
            model_max_desc = model_desc[i]
            max_em = best_em
            max_model = best_model

    final_model =  {'id':model_max,'name':model_max_desc,'score':max_score,'classifier':model_clf[model_max],'Preds':preds}
        
    model_result = {'modelid':model_id,'name':model_name,'Preds':model_preds,'score':model_score,'precision':model_precision,'recall':model_recall,'f1Score':model_f1score}
    model_result = pd.DataFrame(model_result)

    joblib.dump(max_model, '../models/final_model.pkl', compress=1)
    print("Best model for the dataset is " , final_model['name'])
    print("Best model's accuracy score is " , final_model['score'], "%")

    return model_result,final_model,best_X_train, best_X_test, best_y_train, best_y_test,max_em

def load_model():
    """
    Description:
    function is load saved model to classifiying new news stories
    
    Inputs:
    None
    
    Outputs:
    model - saved model
    """
    # load the model from disk
    model = joblib.load(open('../models/final_model.pkl', 'rb'))
    return model

def confusionmatrix(em):
    """
    Description:
    function is to create a confusion matrix
    
    Inputs:
    em - embedding of the X_test
    
    Outputs:
    precision,recall,f1score - metrics in  confusion matrix
    """
    #Confusion Matrix
    # em['out'] = em['color']
    # fp = (em['out']!=em['sentiment_summary']) & (em['sentiment_summary']==0)
    # fp = sum(fp)  
    # fn = (em['out']!=em['sentiment_summary']) & (em['sentiment_summary']==1)
    # fn = sum(fn)

    # tp = (em['out']==em['sentiment_summary']) & (em['sentiment_summary']==1)
    # tp  = sum(tp )  
    # tn = (em['out']==em['sentiment_summary']) & (em['sentiment_summary']==0)
    # tn = sum(tn)
    #print("{fp :",fp,", fn :",fn,", tp :",tp,", tn :",tn,"}")

    # if tp !=0:
    #     precision = round(tp/(tp+fp),2)
    #     recall = round(tp/(tp+fn),2)
    #     #print('precision :' ,precision)
    #     #print('recall :' ,recall)
    #     f1score = 2 * (precision * recall) / (precision + recall)
    # else:
    #     precision,recall,f1score = 0,0,0
    y_pred = em["color"]
    y_test = em["sentiment_summary"]
    precision = precision_score(y_test,y_pred,average = "micro")
    recall = recall_score(y_test,y_pred,average = "micro")
    f1score = f1_score(y_test,y_pred,average = "micro")
    return precision,recall,f1score