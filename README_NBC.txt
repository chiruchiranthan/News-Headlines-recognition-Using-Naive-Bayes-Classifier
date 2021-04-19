Implement a Naive Bayes Classifier(NBC) , of English news paper Head Lines into
Politics, Sports, Education, Healthcare, Finance ( 5- Class Labels)
---------------------------------------------------------------------------------------------------
Objective:-The  main aim of the problem is to classify news into categories based on their headline.

>>The method used to do the classification is naive bayes classifier. 

>>Before applying the MultiniomailNB the dataset is cleaned by removing
the stopwords and punctuations.

>>Then the dataset is divided to train and test data, then training dataset
is trained using MultinomialNB.

>>After training, the test data is tested and accuracy of the test is calculated.

>>When a fresh new headlines is encountered then trained model is used to test 
the headlines and classify to a particular category.