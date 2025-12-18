import joblib

TEXT = []
def LawClf_predict(TEXT):
  LawClf = joblib.load('LawClf.joblib')
  return LawClf.predict(TEXT)

if __main__ == '__main__':
  LawClf_predict(TEXT)
