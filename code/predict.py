def predict(vectorizer,X,y):
    '''
    FUNC: test model, print accuracy

    input:  X = raw article text in dataframe form
            y = the true label dataframe
    output: predictions
    '''
    X = vectorizer.transform(X)
    print "Accuracy:", model.score(X, y)
    return "Predictions:", model.predict(X)
