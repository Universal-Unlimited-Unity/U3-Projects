import joblib

TEXT = ['divorce happens if the wife wants']
def LawClf_predict(TEXT):
  LawClf = joblib.load('LawClf.joblib')
  return LawClf.predict_proba(TEXT), LawClf.classes_

if __name__ == '__main__':
  print(LawClf_predict(TEXT))
