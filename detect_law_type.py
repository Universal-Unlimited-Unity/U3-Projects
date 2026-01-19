import joblib
import pandas as pd

class Law:
  def __init__(self, law: list[str] | None):
    self.law = law
    self.ready = False
    
  def __isready(self):
    return self.ready
      
  def __ensure_law(self):
    if not self.law:
      raise Exception("Law List Is Empty")
  def __prepare(self):
    self.__ensure_law()
    RuleClf = joblib.load('RuleClf.joblib')
    rule_soft_type = RuleClf.predict_proba(self.law)
    rule_soft_type = pd.DataFrame(rule_soft_type, columns=RuleClf.classes_)
    rule_soft_type.insert(0, 'sents', self.law)
    self.rule_soft_type = rule_soft_type
    rule_hard_type = RuleClf.predict(self.law)
    rule_hard_type = pd.DataFrame({'sents': self.law, 'label': rule_hard_type})
    self.rule_hard_type = rule_hard_type
    self.defs = self.rule_hard_type[rule_hard_type['label'] == 'definition']['sents']
    self.oblgs = self.rule_hard_type[rule_hard_type['label'] == 'obligation']['sents']
    self.prohs = self.rule_hard_type[rule_hard_type['label'] == 'prohibition']['sents']
    self.sancs = self.rule_hard_type[rule_hard_type['label'] == 'sanction']['sents']
    self.exceps = self.rule_hard_type[rule_hard_type['label'] == 'exception']['sents']
    
    # prepared
    
    self.ready = True

  def rules_types(self) -> pd.DataFrame:
    self.__ensure_law()
    LawClf = joblib.load('LawClf.joblib')
    probs = LawClf.predict_proba(self.law)
    df = pd.DataFrame(probs, columns=LawClf.classes_)
    df.insert(0, 'sents', self.law)
    return df

  def types_ratio(self, normalize: bool = False) -> pd.DataFrame:
    x=1
    if normalize:
      x=100
    df = self.rules_types()
    df.drop('sents', axis=1, inplace=True)
    df = df.mean().reset_index()
    df.columns = ['Law Type', 'Percentage']
    df['Percentage'] = df['Percentage']*x
    return df.sort_values(by='Percentage', ascending=False).reset_index(drop=True)


  def type(self, margin: float | None = None) -> str | pd.DataFrame:
    if margin is None:
      margin = 0.01
    df = self.types_ratio()
    maxv = df['Percentage'].max()
    df = df[df['Percentage'] > maxv - margin]
    if len(df) == 1:
      return df['Law Type'].iloc[0]
    else:
      return df[['Law Type', 'Percentage']].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)

  def legal_function(self):
    if self.__isready():
      return self.rule_hard_type
    self.__prepare()
    return self.rule_hard_type

  def legal_function_probs(self):
    if self.__isready():
      return self.rule_soft_type
    self.__prepare()
    return self.rule_soft_type

  def definitions(self):
    if self.__isready():
      return self.defs
    self.__prepare()
    return self.defs

  def obligations(self):
    if self.__isready():
      return self.oblgs
    self.__prepare()
    return self.oblgs

  def prohibitions(self):
    if self.__isready():
      return self.prohs
    self.__prepare()
    return self.prohs
    
  def sanctions(self):
    if self.__isready():
      return self.sancs
    self.__prepare()
    return self.sancs
    
  def exceptions(self):
    if self.__isready():
      return self.exceps
    self.__prepare()
    return self.exceps
    
    
law = [
    "For the purposes of this Act, a public authority is any body exercising governmental functions.",
    "Personal data means any information relating to an identified or identifiable natural person.",
    "Processing refers to any operation performed on personal data, whether automated or not.",
    "A data controller is the entity that determines the purposes and means of processing personal data.",
    "A data processor is a person who processes personal data on behalf of the controller.",

    "The controller must ensure that personal data is processed lawfully and transparently.",
    "The controller must implement appropriate technical and organizational security measures.",
    "Public authorities are required to respond to access requests within the statutory period.",
    "Employees must receive adequate training regarding data protection obligations.",
    "The processor shall act only on documented instructions from the controller.",

    "It is prohibited to process personal data without a valid legal basis.",
    "No person shall disclose confidential information obtained through official duties.",
    "Personal data shall not be retained longer than necessary for the stated purpose.",
    "Unauthorized access to information systems is forbidden.",
    "The transfer of personal data to third parties without authorization is prohibited.",

    "Any person who unlawfully processes personal data shall be subject to administrative fines.",
    "Failure to comply with security obligations shall result in suspension of processing activities.",
    "A controller who violates this Act may be ordered to pay compensation to affected persons.",
    "Repeated violations shall give rise to increased penalties.",
    "Noncompliance with enforcement orders shall be punishable by additional sanctions.",

    "This Act shall not apply where processing is carried out for national security purposes.",
    "Disclosure obligations do not apply where legal professional privilege exists.",
    "Access rights may be restricted where disclosure would seriously impair an ongoing investigation.",
    "The obligation to erase data shall not apply where retention is required by law.",
    "Processing may continue without consent in cases of vital interest of the data subject.",

    "A fine may be imposed for each day of continued noncompliance.",
    "Except in cases of emergency, inspections shall be announced in advance.",
    "A controller is not liable where the damage was caused by an unforeseeable external event.",
    "Any sanction imposed must be proportionate to the severity of the infringement.",
    "Except as otherwise provided by law, enforcement decisions shall be immediately applicable."
]

law = Law(law)

print(law.rules_types())
print(law.types_ratio())
print(law.type())
print(law.legal_function_probs())
print(law.legal_function())

print("defs")
print(law.definitions())
print("oblgs")
print(law.obligations())
print("exps")
print(law.exceptions())
print("sancts")
print(law.sanctions())
print("prohs")
print(law.prohibitions())
