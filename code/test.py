from sklearn.cross_validation import train_test_split
import pandas as pd

from multinomial_bayes_model import build_model
from predict import predict

X = pd.read_csv('/Users/drewrice/Desktop/Github/Galvanize/final-assessment-2/data/train.txt', header=None)
X.columns = ['body']
y = pd.read_csv('/Users/drewrice/Desktop/Github/Galvanize/final-assessment-2/data/labels.txt', header=None)
y.columns = ['label']

# train test split for evaluation purposes
X_train, X_test, y_train, y_test = train_test_split(X['body'],y['label'])

# run the model
_vectorizer, _model = build_model(X_train,y_train)
preds = predict(_vectorizer,_model,X_test,y_test)
print preds
