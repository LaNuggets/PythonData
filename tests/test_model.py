from src.model import TitanicModel
from src.preprocessor import TitanicPreprocessor
import pandas as pd
import pytest



@pytest.fixture
def basic_info():
    df = pd.read_csv("data/train.csv")
   
    prep = TitanicPreprocessor()
    X, y = prep.fit_transform(df)
    
    return X, y

def test_train(basic_info):

    model = TitanicModel()
    X_train, X_test, y_train, y_test = model.train(basic_info[0], basic_info[1])
    assert len(X_train) == len(y_train)


def test_predict(basic_info):

    model = TitanicModel()
    X_train, X_test, y_train, y_test = model.train(basic_info[0], basic_info[1])
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(X_test)


def test_evaluate(basic_info):
    model = TitanicModel()
    X_train, X_test, y_train, y_test = model.train(basic_info[0], basic_info[1])
    y_pred = model.predict(X_test)
    acc = model.evaluate(y_test, y_pred)
    assert acc >= 0.7, "Accuracy is to bad, got {acc}"