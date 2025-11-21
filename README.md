This project builds a complete sentiment-analysis pipeline for IMDB movie reviews. I used two different NLP techniques (BOW and TF-IDF) to tokenize and convert each text into numerical features. 
Using these representations, I trained and evaluated a Decision Tree and a Logistic Regression classifier to predict whether a review is positive or negative. 
I compared the models using accuracy and detailed classification metrics, and for the Decision Tree, examined which words contributed most to its predictions.


The models both performed pretty well, with TFIDF performing way better even when increasing the ngrams of BOW. Here is an example output:
Accuracy BOW: 0.7205454545454546
Accuracy TfIDF: 0.9063030303030303
              precision    recall  f1-score   support

    negative       0.72      0.73      0.72      8250
    positive       0.72      0.71      0.72      8250
    accuracy                           0.72     16500
   macro avg       0.72      0.72      0.72     16500
weighted avg       0.72      0.72      0.72     16500

              precision    recall  f1-score   support

    negative       0.91      0.90      0.91      8250
    positive       0.90      0.92      0.91      8250
    accuracy                           0.91     16500
   macro avg       0.91      0.91      0.91     16500
weighted avg       0.91      0.91      0.91     16500
