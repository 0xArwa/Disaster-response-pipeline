# Import libraries
import re
import sys
import pickle
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])
from sqlalchemy import create_engine
from sqlalchemy import text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Load data from database and return features, 
    targets and category names.
    
    Args:
    database_filepath: str - path to the database file
    
    Returns:
    X: pandas dataframe - features
    Y: pandas dataframe - targets
    category_names: list - list of category names
    
    """
    # load data from database
    conn = create_engine(f'sqlite:///{database_filepath}')
    query = text("""SELECT * FROM Responses""")
    df = pd.read_sql(query, conn.connect())

    #splitting labels
    X = df['message'] 
    Y = df.iloc[:, 4:] 
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text, normalize, remove stop words, stem and lemmatize.
    
    Args:
    text: str - text to be tokenized
    
    Returns:
    lemmed: list - list of cleaned and lemmatized tokens
    
    """
    # normalize text
    text = text.lower() 
    #clean text
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    #tokenize text using this function
    words = word_tokenize(text) 
    #store stop words 
    stop_words = stopwords.words('english') 
    #remove all stop words
    new_text = [x for x in words if x not in stop_words]
    #lemming
    lemmed = [WordNetLemmatizer().lemmatize(x) for x in new_text]
    return lemmed


def build_model():
    """
    Build a pipeline for features extraction, transformation, and classification using the best parameters
    obtained through grid search.
    
    Returns:
    pipeline1: sklearn pipeline - pipeline for features extraction, transformation, and classification
    
    """
    #pipeline for features extraction, transformation, classification 
    #using best paramteres through grid search used in pipeline prep.
    pipeline1 = Pipeline([
        ('vectorize', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('KNN',  MultiOutputClassifier(KNeighborsClassifier(n_neighbors = 10)))
    ])

    return pipeline1

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the model by calculating accuracy
    score and classification report.
    
    Args:
    model: sklearn pipeline - trained model
    X_test: pandas dataframe - test features
    Y_test: pandas dataframe - test targets
    category_names: list - list of category names
    
    Returns:
    None
    
    """
    #storing predictions to test performance
    y_pred = model.predict(X_test)
    y_true = Y_test.to_numpy()

    #printing f1 score, recall, pricision for each column 
    for i in range(y_true.shape[1]):
        print(f"Column {i}:")
        print(classification_report(y_true[:, i], y_pred[:, i]))
        print('\n')


def gridSearch(model, X_train, y_train):
    """
    Searching for best parameters for the KNN model
    
    Args:
    model: sklearn pipeline - trained model
    X_train: pandas dataframe - train features
    y_train: pandas dataframe - train labels
    Returns:
    best params
    
    """
    parameters = {
    'KNN__estimator__n_neighbors': [5,7,10]}

    cv = GridSearchCV(model, parameters)
    grid_search = cv.fit(X_train, y_train)
    return grid_search.best_params_

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
    model: sklearn pipeline - trained model
    model_filepath: str - path to save the model
    
    Returns:
    None
    
    """
    #exporting the model 
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """
    Train, evaluate and save the multioutput model.
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Tunning the model...')
        print(gridSearch(model, X_train, Y_train))
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

