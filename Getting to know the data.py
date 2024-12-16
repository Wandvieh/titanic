# %%
import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots

df = pd.read_csv("train.csv")

# Größe des DataFrames
print(df.shape) # 12 columns, 891 rows
df_nomissing = df.dropna()
# Anzahl der Rows mit fehlenden Angaben
print(df.shape[0] - df_nomissing.shape[0]) # 708 rows

# Namen, Datentypen, ungültige Werte der Spalten
print(df.columns) # ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
print(df.dtypes)
for col in df.columns:
    print("Ungültige Werte in {0}:".format(col), df[col].shape[0] - df[col].count())
# -> Cabin hat zu viele ungültige Werte (687); ist vermutlich nicht nutzbar
# Fraglich, ob Age mit 177 ungültigen Werten nutzbar ist

# Infos zu allen numerischen Spalten: PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare
print(df.describe())

# Einzigartige Elemente in den restlichen interessanten Spalten: Sex, Ticket
for col in df[['Sex', 'Ticket']].columns:
    print("Einzigartige Elemente in {0}".format(col))
    print(np.unique(df[col]))

"""
-> Nicht interessant für die Prediction: Cabin, Name, ID, Fare, Embarked, Ticket (das vllt doch aber sieht nicht nach guten Infos aus)
-> Visualisierungen machen für Survived und: Pclass, Sex, Age (evtl), SibSp, Parch
"""

# Pclass zu categorical umformen
df.Pclass = pd.Series(df.Pclass, dtype='category')
df.Sex = pd.Series(df.Sex, dtype='category')
df.SibSp = pd.Series(df.SibSp, dtype='category')
df.Parch = pd.Series(df.Parch, dtype='category')
print(df.dtypes)

# %% Visualisierungen
"""
PassengerId      int64
Survived         int64
Pclass        category
Sex           category
Age            float64
SibSp         category
Parch         category
"""
fig, ax = subplots(figsize=(8, 8))

#df.boxplot(col, by='Survived', ax=ax)
#fig.show()
df.hist('Age', by='Survived', ax=ax)
fig.show()



# %%
