#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


df = pd.read_csv("Delinquency_prediction_dataset.csv")

df.head()


# In[5]:


df.info()
df.describe()


# In[6]:


df.isnull().sum()


# In[13]:


df.columns


# In[10]:


#visual eda for histogram
import matplotlib.pyplot as plt

numeric_cols = [
    "Age",
    "Income",
    "Credit_Score",
    "Credit_Utilization",
    "Missed_Payments",
    "Loan_Balance",
    "Debt_to_Income_Ratio",
    "Account_Tenure",
    "Month_1", "Month_2", "Month_3",
    "Month_4", "Month_5", "Month_6"
]

for col in numeric_cols:
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# In[14]:


import matplotlib.pyplot as plt
import numpy as np

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()


# In[18]:


corr_target = corr["Delinquent_Account"].sort_values(ascending=False)
corr_target


# In[19]:


plt.figure(figsize=(8,5))
corr_target.plot(kind='bar')
plt.title("Correlation of Features with Delinquent_Account")
plt.ylabel("Correlation Value")
plt.tight_layout()
plt.show()


# In[31]:


from sklearn.preprocessing import LabelEncoder

cat_cols = ["Employment_Status", "Credit_Card_Type", "Location"]

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])
# here drop the customer_id column 


# In[32]:


#train test split
from sklearn.model_selection import train_test_split

X = df.drop("Delinquent_Account", axis=1)
y = df["Delinquent_Account"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[34]:


df.dtypes


# In[35]:


month_cols = ["Month_1", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6"]

for col in month_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# In[36]:


df[month_cols] = df[month_cols].fillna(df[month_cols].median())


# In[37]:


df.dtypes


# In[40]:


for col in ["Month_1","Month_2","Month_3","Month_4","Month_5","Month_6"]:
    print(col, df[col].unique()[:10])


# In[41]:


month_cols = ["Month_1","Month_2","Month_3","Month_4","Month_5","Month_6"]

for col in month_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace("Missed","", regex=False), 
                            errors="coerce")


# In[42]:


df[month_cols] = df[month_cols].fillna(df[month_cols].median())


# In[44]:


df.dtypes


# In[46]:


df.isnull().sum()


# In[49]:


df = df.drop(["Month_1", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6"], axis=1)


# In[51]:


df.isnull().sum()


# In[53]:


#  Drop Customer_ID FIRST
df = df.drop("Customer_ID", axis=1)


# In[54]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

cat_cols = ["Employment_Status", "Credit_Card_Type", "Location"]
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

#  Impute missing numeric values
from sklearn.impute import SimpleImputer
num_cols = df.select_dtypes(include=[np.number]).columns

imputer = SimpleImputer(strategy="mean")
df[num_cols] = imputer.fit_transform(df[num_cols])

# Check no NaN remains
print(df.isnull().sum())

# Train-test split
from sklearn.model_selection import train_test_split

X = df.drop("Delinquent_Account", axis=1)
y = df["Delinquent_Account"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))


# In[1]:


import pandas as pd
import numpy as np

# Separate classes
df_majority = df[df["Delinquent_Account"] == 0]
df_minority = df[df["Delinquent_Account"] == 1]

# Oversample minority manually
df_minority_oversampled = df_minority.sample(
    n=len(df_majority),
    replace=True,
    random_state=42
)

# Combine them
df_balanced = pd.concat([df_majority, df_minority_oversampled])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_balanced["Delinquent_Account"].value_counts())


# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = df_balanced.drop("Delinquent_Account", axis=1)
y = df_balanced["Delinquent_Account"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight=None   # IMPORTANT: DO NOT USE balanced here
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))


# In[64]:


model.fit(X_train, y_train)
pred = model.predict(X_test)


# In[65]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# CONFUSION MATRIX

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#ROC Curve

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# In[ ]:




