# Titantic Dataset

[Kaggle competition](https://www.kaggle.com/competitions/titanic)

My goal is to use the titanic dataset to implement the different kinds of models that I learned in order to understand them better and get more fluent in implementing them. Writing concise and efficient code is not of importance for this.

In future exercises, another focus might be calculation speed or proper approach to choosing variables and polynomials / interaction terms.

## Final results

- Least Squares Regression
  - implemented polynomial and interaction terms
  - R^2 is at 45.9%
  - The classification error is at 14.5%
  - The final model includes:
    - Pclass (first and second degree)
    - Sex (first and second degree)
    - Fare (first degree)
    - Age:Sex (first degree)
    - Pclass:Sex (first and second degree)
    - Pclass:SibSp (first degree)
    - Age:Parch (first degree)
    - Age:Fare (first degree)
    - Parch (first degree)

## My To Do List
Models
[x] Least Squares Regression
[ ] Logistic Regression
[ ] Poisson Regression
[ ] LDA
[ ] QDA

Extensions
- interaction terms (e.g. between the Cabin Letters and their numbers - but there are so many missing entries that it probably won't be important)
- polynomial methods

Validations
- bootstrap
- 5fold / 10fold CV

Other considerations
- does the data have a gaussian distribution? -> if not, logistic regression will work better than LDA
- do the variables have a common covariance matrix? -> if not, logistic regression will work better than LDA
