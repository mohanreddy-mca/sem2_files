import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk import word_tokenize, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn import metrics, tree
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#uncomment your own directory accordingly
data_msia = r"C:\Users\GelloMark\Google Drive\School\NUS\GitHub\KE5205-TextMining-CA\data\MsiaAccidentCases - Cleaned.xlsx"
#data_msia="C:\githome\ivarunkumar\KE5205-TextMining-CA\data\MsiaAccidentCases.xlsx"

stopword_custom = ["victim", "year", "morning", "afternoon","00","00pm","10","107","11","12","13","13th","14","15","16","18","20","200","2011","22","24th","25","26","26th","28","2km","30","32","34","35","360","39","3rd","40","43","45","47","50th","52","55","6th","75","80","100","16th","27","301","45pm","4th"]
stopword = stopwords.words('english') + stopword_custom

snowball = SnowballStemmer('english')
wnl = WordNetLemmatizer()

def preprocessDocument(text):
    toks = word_tokenize(text)
    toks = [t.lower() for t in toks if t not in string.punctuation]
    toks = [t for t in toks if t not in stopword]
    toks = [wnl.lemmatize(t) for t in toks]
    out= " ".join(toks)
    return out

def loadFileAndProcess(path) :
    dataFrame = pd.read_excel(path)
    
    #drop all rows with no data
    dataFrame=dataFrame.dropna(how='all')
    #print("rows, columns: " + str(dataFrame.shape))
    
    #check for empty title case
    #dataFrame[dataFrame["Title Case"].isnull()]

    #Count the length of each document
    length=dataFrame['Summary Case'].apply(len)
    dataFrame=dataFrame.assign(Length=length)
    
    #Plot the distribution of the document length for each category
    #dataFrame.hist(column='Length',by='Cause',bins=10)
    #plt.show()
    
    #Apply the function on each document
    dataFrame['Text'] = dataFrame['Summary Case'].apply(preprocessDocument)
    #dataFrame.head()
    return dataFrame

#load and process file
df_msia = loadFileAndProcess(data_msia)

#check groups
'''
Mark: data is not balanced, i.e. exposure having 3 counts only, need to fix this before doing a model
I'll get more cases of exposure from osha file, need help to balance other group when you're free
'''
df_msia.groupby('Cause').describe()

#export to csv
df_msia.to_csv('msia_lemmatized.csv', index=False, encoding='utf-8')

'''
Mark: after balancing the cause counts, we can proceed to this and build the dtm
'''
#split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(df_msia.Text, df_msia.Cause, test_size=0.33, random_state=12)





'''



Please help to clean code below.



'''

#Build a pipeline: Combine multiple steps into one

#def runPipeline() :
#    text_clf = Pipeline([('vect', CountVectorizer()),  
#                         ('tfidf', TfidfTransformer()),
#                         ('clf', MultinomialNB()),
#                         ])
#    return text_clf


#def buildModel() :
#    
#def __main__() :
#    generateDTM(data_msia)
#    

#def generateDTM(path) : 
#    df = loadFileAndClean(path)
#    for d in df :
#        print (d[2])

#Create dtm by using word occurence

#Create dtm by using word occurence
count_vect = CountVectorizer(stop_words=stopword)
x_train_count = count_vect.fit_transform(x_train)
x_train_count.shape
count_vect.get_feature_names()

dtm1 = pd.DataFrame(x_train_count.toarray().transpose(), index = count_vect.get_feature_names())
dtm1=dtm1.transpose()
dtm1.head()
dtm1.to_csv('dtm1.csv',sep=',')

#SVM
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                      ('clf', SGDClassifier(
                                            alpha=1e-3
                                             ))
                    ])

text_clf.fit(x_train, y_train)  

predicted = text_clf.predict(x_test)
 
print(metrics.confusion_matrix(y_test, predicted))
print(np.mean(predicted == y_test) )

# If we give this parameter a value of -1, 
#grid search will detect how many cores are installed and uses them all:
parameters = {
                  'tfidf__use_idf': (True, False),
                   'clf__alpha': (1e-2, 1e-3),
                }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(x_train, y_train)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
###################################################################
# Analyze the distribution
#words = [word for doc in cleaned for word in doc]
#type(cleaned)

#fd_words = FreqDist(words)
#fd_words.plot()
#fd_most_common=fd_cat.most_common(40)


 