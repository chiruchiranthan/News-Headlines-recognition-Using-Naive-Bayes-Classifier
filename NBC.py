import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


    
#Loading of dataset
news = pd.read_csv('F:/CHIRU/PESIT/1SEM/SML/Final_Assignment/ques1/SML_Data.csv.')
print("Categories of news are")
print(news['Categories'].unique())


#Cleaning the dataset
a=news['Headlines']
def text_cleaning(a):
    remove_punctuation=[char for char in str(a) if char not in string.punctuation]
    remove_punctuation=''.join(remove_punctuation)
    return [word for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]


#Accuracy calculation
def acc():
    X_train, X_test, y_train, y_test = train_test_split(news.Headlines, news.Categories,test_size=0.1,random_state=109)
    #Training 
    vectorizer = CountVectorizer(analyzer=text_cleaning).fit(X_train.values.astype('U'))
    X = vectorizer.transform(X_train.values.astype('U'))
    tfidf_transformer=TfidfTransformer().fit(X)
    headlines_tfidf=tfidf_transformer.transform(X)
    mnb = MultinomialNB().fit(headlines_tfidf,y_train.values.astype('U'))
    #Testing
    X = vectorizer.transform(X_test.values.astype('U'))
    headlines_tfidf=tfidf_transformer.transform(X)
    predicted_NBC=mnb.predict(headlines_tfidf)
    print("Accuracy:",metrics.accuracy_score(y_test, predicted_NBC))
    print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, predicted_NBC))

acc()


def singleline():
    vectorizer = CountVectorizer(analyzer=text_cleaning).fit(news['Headlines'].values.astype('U'))
    X = vectorizer.transform(news['Headlines'].values.astype('U'))

    tfidf_transformer=TfidfTransformer().fit(X)
    headlines_tfidf=tfidf_transformer.transform(X)

    #Using MultinomialNaivebased classifier
    mnb = MultinomialNB().fit(headlines_tfidf,news['Categories'].values.astype('U')) 


    #Testing of new headlines
    docs_new=input("enter the headlines\n\n")
    docs_new=[docs_new] 
    X_new_count=vectorizer.transform(docs_new)
    X_new_tfidf=tfidf_transformer.transform(X_new_count)
    predicted_NBC=mnb.predict(X_new_tfidf)
    print("\n")
    print(predicted_NBC)
    
singleline()









#if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
#    csv_path = sys.argv[1]