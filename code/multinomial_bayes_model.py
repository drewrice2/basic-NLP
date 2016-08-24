from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def build_model(X,y):
    '''
    FUNC:   build model and fit_transform to the test dataset

    input:  X = raw article text in dataframe form
            y = the label dataframe
    output: vectorizer for vectorizing new input
            model for scoring and predicting
    '''
    vect, _model = TfidfVectorizer(stop_words='english'), MultinomialNB()

    _vectorizer = vect.fit_transform(X)
    model = _model.fit(_vectorizer,y)
    return vect, model
