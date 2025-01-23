# Titantic Dataset

Original on kaggle{https://www.kaggle.com/competitions/titanic}

things to try out still
- Converting Cabin also into its Number and add an interaction term between the two

methods to use
- Logistic Regression
- Poisson Regression
- LDA
- QDA

extensions of those methods
- interaction terms
- polynomial methods

validations to use on all methods
- bootstrap
- 5fold / 10fold CV

methods to evaluate
- does the data have a gaussian distribution? -> if not, logistic regression will work better than LDA
- do the variables have a common covariance matrix? -> if not, logistic regression will work better than LDA