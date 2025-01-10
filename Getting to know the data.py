# %% Getting to know the data
import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, summarize)

df = pd.read_csv("train.csv")
original_df = df.copy()

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
"""df.Survived = pd.Series(df.Survived, dtype='category')
df.Pclass = pd.Series(df.Pclass, dtype='category')
df.Sex = pd.Series(df.Sex, dtype='category')
df.SibSp = pd.Series(df.SibSp, dtype='category')
df.Parch = pd.Series(df.Parch, dtype='category')
print(df.dtypes)"""

# %% Erste Visualisierungen
"""
PassengerId      int64
Survived         int64
Pclass        category
Sex           category
Age            float64
SibSp         category
Parch         category
"""
pd.plotting.scatter_matrix(df, diagonal='kde')

#fig, ax = subplots(figsize=(8, 8))

#df.boxplot(col, by='Survived', ax=ax)
#fig.show()
#df.hist('Age', by='Survived', ax=ax)
#fig.show()

# %% Erster Test: Visualisierungen mit verschiedenen Farben
fig, ax = subplots(figsize=(8, 8))
# Die Rows filtern, die Survived 0 bzw 1 sind
df_positive = df.loc[lambda df: df['Survived'] == 1, :]
df_negative = df.loc[lambda df: df['Survived'] == 0, :]
ax.plot(df_positive['Age'], df_positive['Sex'], 'o', fillstyle='none', markersize=15.0)
ax.plot(df_negative['Age'], df_negative['Sex'], 'o', fillstyle='none')
fig.show()

# %% Der Versuch, ein kumullatives Line Chart zu bauen (für die relativen Überlebenschancen nach Alter)
fig, ax = subplots(figsize=(8, 8))
df.boxplot('Age', by='Survived', ax=ax)

#klappt nicht so recht

# %% Simple Linear Regression auf allen möglichen Variablen mit Plotting

def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)

y = df['Survived']
for column in ['PassengerId', 'Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']:
    X = pd.DataFrame({'intercept': np.ones(df.shape[0]),
                      'Pclass': df[column]})
    model = sm.OLS(y,X, missing='drop') # ordinary least squares model
    results = model.fit()
    print("Modell für {0}:\n".format(column), summarize(results))

    ax = df.plot.scatter(column, "Survived")
    abline(ax, results.params[0], results.params[1], 'r--', linewidth=3)


# %% Multiple linear regression
X_2 = MS(['Pclass', 'Age', 'Sex', 'SibSp']).fit_transform(df)
model_2 = sm.OLS(y,X_2,missing='drop')
results_2 = model_2.fit()
summarize(results_2)

# %%
X_3 = MS(['Pclass', 'Parch', 'Fare']).fit_transform(df)
model_3 = sm.OLS(y,X_3,missing='drop')
results_3 = model_3.fit()
summarize(results_3)
# %%
df['Survived'] = df['Survived'].astype("category")
df['Pclass'] = df['Pclass'].astype("category")
df['Sex'] = df['Sex'].astype("category")
df['Embarked'] = df['Embarked'].astype("category")
print(df.info())
# %% Boxplots for categorization
for column in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    fig, ax = subplots(figsize=(8, 8))
    df.boxplot(column, by='Survived', ax=ax)

# %%
