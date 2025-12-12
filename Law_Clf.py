from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

clf = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

x = ["Any person who commits theft shall be punished according to the penal code. Murder is considered a serious crime and is subject to imprisonment or death penalty depending on the jurisdiction. Fraudulent activities, including embezzlement or forgery, will be investigated and prosecuted by the authorities. Individuals who intentionally cause bodily harm to others may face criminal charges and penalties.",
     "A tenant must pay rent at the agreed time according to the rental contract. If a contract is breached by either party, the affected party may seek compensation through civil court. Property disputes between neighbors shall be resolved under civil procedures. Family law matters, including divorce and child custody, are handled by civil courts to ensure fairness and adherence to the law.",
     "The administration may revoke a permit if the holder violates the regulations. Government agencies have the authority to impose fines on businesses that fail to comply with licensing rules. Public officials are required to follow administrative procedures when making decisions affecting citizens. Applications for official approvals must meet the administrative requirements set by the authorities.",
     "Companies must register with the trade authority before starting operations. Breaches of commercial contracts, such as failure to deliver goods or services, can lead to legal action. Bankruptcy procedures must be followed when a company is unable to meet its financial obligations. Shareholders have rights and responsibilities defined under corporate law.",
     "All individuals and businesses must report their income to the tax authorities annually. Failure to pay taxes on time may result in fines or legal action. Businesses must comply with accounting standards and submit audited reports. Customs duties and import taxes must be calculated and paid according to the applicable financial regulations.",
     "Every citizen has the right to freedom of speech and expression. Elections must be conducted according to the constitution to ensure fairness. The separation of powers ensures that legislative, executive, and judicial branches operate independently. Fundamental rights are protected and cannot be infringed by government authorities.",
     "Countries must comply with international treaties to which they are signatories. Human rights violations can be addressed through international courts. Maritime boundaries and trade agreements are regulated according to international law. Disputes between nations may be resolved through arbitration or diplomacy."
]
y = ["penal", "civil", "administrative", "commercial", "tax", "constitutional", "international"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf.fit(x_train, y_train)

