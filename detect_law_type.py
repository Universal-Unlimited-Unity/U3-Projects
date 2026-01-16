import joblib
import pandas as pd
def Law_Rules_Types(LAW: list[str]) -> pd.DataFrame:
  if not LAW:
    raise Exception("LaW List Is Empty")
  LawClf = joblib.load('LawClf.joblib')
  probs = LawClf.predict_proba(LAW)
  df = pd.DataFrame(probs, columns=LawClf.classes_)
  df.insert(0, 'sents', LAW)
  return df

def Law_Types_Ratio(LAW: list[str], normalize: bool = False) -> pd.DataFrame:
  x=1
  if normalize:
    x=100
  df = Law_Rules_Types(LAW)
  df.drop('sents', axis=1, inplace=True)
  df = df.mean().reset_index()
  df.columns = ['Law Type', 'Percentage']
  df['Percentage'] = df['Percentage']*x
  return df.sort_values(by='Percentage', ascending=False)

def Law_Exact_Type(LAW: list[str], margin: float | None = None) -> str | pd.DataFrame:
  if margin is None:
    margin = 0.01
  df = Law_Types_Ratio(LAW)
  maxv = df['Percentage'].max()
  df = df[df['Percentage'] > maxv - margin]
  if len(df) == 1:
    return df['Law Type'].iloc[0]
  else:
    return df[['Law Type', 'Percentage']].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


LAW = [
    "Any person who intentionally causes the death of another person commits an offense.",
    "An act is considered intentional when the result is desired or knowingly accepted.",
    "A person is criminally responsible if they had the capacity to understand the nature of their actions.",
    "No offense is committed where the act was carried out under lawful self-defense.",
    "An attempt occurs when a person begins the execution of an offense but does not complete it.",
    "Participation in a criminal act includes aiding, abetting, or facilitating its commission.",
    "A person who coerces another to commit an offense is punishable as a principal offender.",
    "Criminal liability requires both a prohibited act and a culpable mental state.",
    "An omission may give rise to liability when there is a legal duty to act.",
    "A penalty may be reduced when the offender voluntarily abandons the criminal conduct.",
    "Recidivism is established when a person commits a new offense after a final conviction.",
    "The use of force is unlawful unless justified by necessity or proportionality.",
    "An offense is aggravated when it is committed with premeditation or cruelty.",
    "A minor may not be held criminally liable under the same conditions as an adult.",
    "Evidence obtained by unlawful means may be excluded from criminal proceedings.",
    "A confession is valid only if it is given freely and without coercion.",
    "An accomplice is a person who knowingly assists in the commission of an offense.",
    "Criminal sanctions aim to punish unlawful conduct and prevent its repetition.",
    "A person is presumed innocent until guilt is established by a final judgment.",
    "Mitigating circumstances may justify a lesser penalty than the statutory maximum.",
    "An offense is deemed completed once all its legal elements are fulfilled.",
    "Intent may be inferred from the circumstances surrounding the act."
]

print(Law_Rules_Types(LAW))
print(Law_Types_Ratio(LAW, normalize=True))
print(Law_Exact_Type(LAW, margin=0.01))

