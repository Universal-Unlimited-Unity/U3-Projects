from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib
PATH = 'U3-Projects/lawdata.csv'

def Train_LawClf(PATH):
  df = pd.read_csv(PATH)
  x = list(df['sentence'])
  y = list(df['law'])
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  LawClf = Pipeline([('tfidf', TfidfVectorizer()),
                ('model', LogisticRegression()
               )])
  LawClf.fit(x_train, y_train)
  
  return LawClf, x_test, y_test

def LawClf_Report(PATH, save_model=False):
  LawClf, x_test, y_test = Train_LawClf(PATH)
  y_predict = LawClf.predict(x_test)
  report = classification_report(y_test, y_predict)
  print(report)
  if save_model:
    joblib.dump(LawClf, 'LawClf.joblib')
  return report

def Train_RuleClf(PATH2):
  df = pd.read_csv(PATH2)
  x = df['sentence']
  y = df['label']
  RuleClf = Pipeline([('Tfidf', TfidfVetorizer()),
                      ('model', LogisticRegression())])
  x_train, x_train, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  return RuleClf, x_test, y_test

def RuleClf_Report(PATH2, save_model=False):
  RuleClf, x_test, y_test = Train_RuleClf(PATH2)
  y_pred = RuleClf.predict(x_test)
  report = classifiaction_report(y_test, y_predict)
  print(report)
  if save_model:
    joblib.dump(RuleClf, 'RuleClf.joblib')
  return report

