import joblib

TEXT = []
def LawClf_predict(TEXT):
  LawClf = joblib.load('LawClf.joblib')
  return LawClf.predict(TEXT)

if __name__ == '__main__':
  print(LawClf_predict(TEXT))
