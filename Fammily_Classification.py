#@title Default title text
def read_CSV(path):
    articles = []
    labels = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            article = row[1]
            for word in STOPWORDS:
                token = ' ' + word + ' '
                article = article.replace(token, ' ')
                article = article.replace(' ', ' ')
            articles.append(article)
        return articles,labels


    
def gen_Class_Prob_Distribution_Train(articles,labels): #cols should addd
    
    from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn import decomposition, ensemble
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer
    #from catboost import CatBoostClassifier
    #clf = CatBoostClassifier(logging_level='Silent')
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix

    import pandas, xgboost, numpy, string
    
    kf = KFold(n_splits=10,shuffle=False)
    cols = ["Hash_val","Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy", 'predict_fam','real_fam']    
    # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
    accuracy_model = []
    df = pd.DataFrame(columns=cols)

    # Iterate over each train-test split
    for train_index, test_index in kf.split(articles):
        # Split train-test
        X_train, X_test = np.array(articles)[train_index.astype(int)], np.array(articles)[test_index.astype(int)]
        y_train, y_test = np.array(labels["Family"])[train_index.astype(int)], np.array(labels["Family"])[test_index.astype(int)]
        z_train, z_test = np.array(labels["Hash_val"])[train_index.astype(int)], np.array(labels["Hash_val"])[test_index.astype(int)]

        nb = Pipeline([('vect', TfidfVectorizer()),('tfidf',TfidfTransformer()),('clf', ensemble.RandomForestClassifier()),])
        nb.fit(X_train, y_train) 


        y_pred = nb.predict(X_test)
        print('accuracy %s' % accuracy_score(y_pred, y_test))

        # Append to accuracy_model the accuracy of the model
        #accuracy_model.append(accuracy_score(y_pred, y_test, normalize=True)*100)

        #---------------------------------------------------------------------------------------------

        X=X_test
        y=y_test
        z=z_test
        p=len(X)
        prob=[]
        lst = []
        for i in range(p):
                a=[]
                a.append(X[i])
                pred = nb.predict_proba(a)
                b=[]
                for x in np.nditer(pred):
                    b.append(x)
                lst.append([z[i],b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],nb.predict(a)[0], y[i]])                
                print(lst)
        df1 = pd.DataFrame(lst, columns=cols)
        df = df.append(df1, ignore_index = True)
    print(df.shape[0])
    return df

def gen_Outlier_Detector(df1,fam_name):
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    
    df2=df1.loc[(df1['real_fam'] == fam_name)&(df1['predict_fam'] == fam_name)]
    lof = LocalOutlierFactor(n_neighbors = 10, novelty = True)
    lof.fit(df2.drop(['Hash_val','predict_fam', 'real_fam'],axis = 1))
    #print(df2.head(50))
    
    #df2=df1.loc[df1['real_fam'] == fam_name]
    #lof = OneClassSVM().fit(df2.drop(['predict_fam', 'real_fam'],axis = 1))
    
    #print(df2.head(20))
    Pkl_Filename = "Pickle_" + fam_name + "_Model.pkl"  
    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(lot, file)

    return lof
    
    
def feature_engineering(articles,labels):
    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(articles, labels)

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(articles)
    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(articles)
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(articles)
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(articles)
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
    
def featEngin_CountVect(articles,labels):
    # label encode the target variable 
    
    from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn import decomposition, ensemble
    
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(labels)
    
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    xtrain_count=count_vect.fit_transform(articles)
    return xtrain_count, train_y
        
    
    
    
    
def train_model(classifier, feature_vector_train, label, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    #predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return classifier

def gen_Class_Prob_Distribution_Eval(articles,labels,article_Pred,label_Pred,hash_val): #cols should addd
    from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn import decomposition, ensemble
    #from catboost import CatBoostClassifier
    #clf = CatBoostClassifier(logging_level='Silent')
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.metrics import accuracy_score, confusion_matrix

    import pandas, xgboost, numpy, string

    cols = ["Hash_val","Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt",
            "Tescrypt","Urausy", 'predict_fam','real_fam','outlier_val']     
    # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
    accuracy_model = []
    df = pd.DataFrame(columns=cols)

 
    X_test = np.array(article_Pred)
    y_test = np.array(label_Pred)
    hash =  np.array(hash_val)

    X_train = np.array(articles)
    y_train = np.array(labels["Family"])

    

    nb = Pipeline([('vect', TfidfVectorizer()),('tfidf', TfidfTransformer()),('clf', ensemble.RandomForestClassifier()),])
    nb.fit(X_train, y_train)


    y_pred = nb.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))

    # Append to accuracy_model the accuracy of the model
    #accuracy_model.append(accuracy_score(y_pred, y_test, normalize=True)*100)

    #---------------------------------------------------------------------------------------------

    X=X_test
    y=y_test
    z=hash
    p=len(X)
    prob=[]
    lst = []
    for i in range(p):
        a=[]
        a.append(X[i])
        pred = nb.predict_proba(a)
        b=[]
        for x in np.nditer(pred):
            b.append(x)

        if(y[i] in (' Loktrom',' Dorkbot',' Nymaim',' JigsawLocker',' Genasom')):
            outlier_val=-1
        else:
            outlier_val=1

        lst.append([z[i],b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],nb.predict(a)[0], y[i],outlier_val])
            
    df1 = pd.DataFrame(lst, columns=cols)
    df = df.append(df1, ignore_index = True)
    
    #df.loc[df['Class'] == 1, "Class"] = -1
    #df.loc[df['Class'] == 0, "Class"] = 1
    
    print(df.shape[0])
    return df

def novelty_detect_eval(df,outlierDetec_Dinome,outlierDetec_Lockyenc,outlierDetec_Sorikrypt,
                        outlierDetec_Tescrypt,outlierDetec_Urausy,outlierDetec_Locky,outlierDetec_Reveton,outlierDetec_Crowti):
    from sklearn.metrics import precision_recall_fscore_support, classification_report,confusion_matrix, precision_recall_curve

    df['pred_outlier_val']=0
    for index, row in df.iterrows():
        if(row['predict_fam']==' Sorikrypt'):
            df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]
            #print(df2)
            val=outlierDetec_Sorikrypt.predict([df2])[0]
            df.at[index, 'pred_outlier_val']=val
                  
        elif(row['predict_fam']==' Tescrypt'):
            df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]
            val=outlierDetec_Tescrypt.predict([df2])[0]
            df.at[index, 'pred_outlier_val']=val

        elif(row['predict_fam']==' Urausy'):
            df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]
            val=outlierDetec_Urausy.predict([df2])[0]
            df.at[index, 'pred_outlier_val']=val

        elif(row['predict_fam']==' Locky'):
            df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]
            val=outlierDetec_Locky.predict([df2])[0]
            df.at[index, 'pred_outlier_val']=val

        elif(row['predict_fam']==' Reveton'):
            df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]
            val=outlierDetec_Reveton.predict([df2])[0]
            df.at[index, 'pred_outlier_val']=val

        elif(row['predict_fam']==' Crowti'):
            df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]
            val=outlierDetec_Crowti.predict([df2])[0]
            df.at[index, 'pred_outlier_val']=val
            
        elif(row['predict_fam']==' Dinome'):
            df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]
            val=outlierDetec_Dinome.predict([df2])[0]
            df.at[index, 'pred_outlier_val']=val

      

        elif(row['predict_fam']==' Lockyenc'):
            df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]
            val=outlierDetec_Lockyenc.predict([df2])[0]
            df.at[index, 'pred_outlier_val']=val
            
    df.loc[df.outlier_val ==-1, 'outlier_val'] = 'f'
    df.loc[df.pred_outlier_val ==-1, 'pred_outlier_val'] = 'f'
    df.loc[df.outlier_val ==1, 'outlier_val'] = 't'
    df.loc[df.pred_outlier_val ==1, 'pred_outlier_val'] = 't'


    y_val = df[['outlier_val']]
    pred_val = df[['pred_outlier_val']]
    print(classification_report(y_val, pred_val.round()))
    #print(accuracy_score(y_val, pred_val))
            
    return df

def new_data_classication(new_data,classificationModel,outlierDetec_Dinome, outlierDetec_Lockyenc,
                          outlierDetec_Sorikrypt,outlierDetec_Tescrypt,outlierDetec_Urausy,outlierDetec_Locky,
                          outlierDetec_Reveton,outlierDetec_Crowti):
  
  cols = ["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy", 'predict_fam']
  pred = classificationModel.predict_proba(new_data)
  b=[]
  row=[]
  for x in np.nditer(pred):
      b.append(x)
  row.append([b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],classificationModel.predict(new_data)[0])
  df1 = pd.DataFrame(lst, columns=cols)


  if(row['predict_fam']==' Sorikrypt'):
      val=outlierDetec_Sorikrypt.predict([df2])[0]
      
            
  elif(row['predict_fam']==' Tescrypt'):
      val=outlierDetec_Tescrypt.predict([df2])[0]

  elif(row['predict_fam']==' Urausy'):
      val=outlierDetec_Urausy.predict([df2])[0]

  elif(row['predict_fam']==' Locky'):
      val=outlierDetec_Locky.predict([df2])[0]

  elif(row['predict_fam']==' Reveton'):
      val=outlierDetec_Reveton.predict([df2])[0]

  elif(row['predict_fam']==' Crowti'):
      val=outlierDetec_Crowti.predict([df2])[0]
      
  elif(row['predict_fam']==' Dinome'):
      val=outlierDetec_Dinome.predict([df2])[0]

  elif(row['predict_fam']==' Lockyenc'):
      val=outlierDetec_Lockyenc.predict([df2])[0]

  print(val)


            



def main():
    from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
    import pandas as pd                     
    import matplotlib.pyplot as plt          # plotting
    import numpy as np                       # dense matrices
    from scipy.sparse import csr_matrix      # sparse matrices
    %matplotlib inline
    
    
    from google.colab import drive
    drive.mount('/content/gdrive')
    copied_path = "file_path"
    data = pd.read_csv(copied_path)
    pd.set_option('display.max_columns', None)
    pd.options.display.max_colwidth = 100000
    data = data[data.Family != ' Occamy']
    data = data[data.Family != ' Genasom']
    data = data[data.Family != ' Tobfy']
    data = data[data.Family != 'Locky']
    data = data[data.Family != ' Loktrom']
    data.drop_duplicates(keep=False,inplace=True) 
    labels=data[['Family','Hash_val']]
    articles=data['API seq']
    
    df = gen_Class_Prob_Distribution_Train(articles,labels)
   

    #generate outlier detector per class

    outlierDetec_Dinome = gen_Outlier_Detector(df," Dinome")
    #outlierDetec_Loktrom = gen_Outlier_Detector(df," Loktrom")
    outlierDetec_Lockyenc = gen_Outlier_Detector(df," Lockyenc")
    outlierDetec_Sorikrypt = gen_Outlier_Detector(df," Sorikrypt")
    #outlierDetec_Tobfy = gen_Outlier_Detector(df," Tobfy")
    outlierDetec_Tescrypt = gen_Outlier_Detector(df," Tescrypt")
    #outlierDetec_Genasom = gen_Outlier_Detector(df," Genasom")
    outlierDetec_Urausy = gen_Outlier_Detector(df," Urausy")
    outlierDetec_Locky = gen_Outlier_Detector(df," Locky")
    outlierDetec_Reveton = gen_Outlier_Detector(df," Reveton")
    outlierDetec_Crowti = gen_Outlier_Detector(df," Crowti")
    
    #evaluate model
    #articles,labels = read_CSV("all.csv")
    df2 = gen_Class_Prob_Distribution_Eval(articles,labels,article_Pred,label_Pred,hash_val)
    df4=novelty_detect_eval(df2,outlierDetec_Dinome,outlierDetec_Lockyenc,
                            outlierDetec_Sorikrypt,outlierDetec_Tescrypt,outlierDetec_Urausy,
                            outlierDetec_Locky,outlierDetec_Reveton,outlierDetec_Crowti)
    df4=df4[df4.outlier_val == 'f']
    #print(df4)
    

if __name__ == "__main__":
    main()


  #  "Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Tobfy","Urausy"
    
    
