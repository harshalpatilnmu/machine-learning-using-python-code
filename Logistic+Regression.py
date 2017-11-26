
# coding: utf-8

# # Logistic Regression

# In[ ]:

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report


# In[ ]:

get_ipython().magic('matplotlib inline')
rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')


# ## Logistic regression on the titanic dataset
# The first thing we are going to do is to read in the dataset using the Pandas' read_csv() function. We will put this data into a Pandas DataFrame, called "titanic", and name each of the columns.

# In[ ]:

url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
titanic.head()


# Just a quick fyi (we will examine these variables more closely in a minute):
# 
# ##### VARIABLE DESCRIPTIONS
# 
# Survived - Survival (0 = No; 1 = Yes)<br>
# Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)<br>
# Name - Name<br>
# Sex - Sex<br>
# Age - Age<br>
# SibSp - Number of Siblings/Spouses Aboard<br>
# Parch - Number of Parents/Children Aboard<br>
# Ticket - Ticket Number<br>
# Fare - Passenger Fare (British pound)<br>
# Cabin - Cabin<br>
# Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# ### Checking that your target variable is binary
# Since we are building a model to predict survival of passangers from the Titanic, our target is going to be "Survived" variable from the titanic dataframe. To make sure that it's a binary variable, let's use Seaborn's countplot() function.

# In[ ]:

sb.countplot(x='Survived',data=titanic, palette='hls')


# Ok, so we see that the Survived variable is binary (0 - did not survive / 1 - survived)
# 
# ### Checking for missing values
# It's easy to check for missing values by calling the isnull() method, and the sum() method off of that, to return a tally of all the True values that are returned by the isnull() method.

# In[ ]:

titanic.isnull().sum()


# Well, how many records are there in the data frame anyway?

# In[ ]:

titanic.info()


# Ok, so there are only 891 rows in the titanic data frame. Cabin is almost all missing values, so we can drop that variable completely, but what about age? Age seems like a relevant predictor for survival right? We'd want to keep the variables, but it has 177 missing values. Yikes!! We are going to need to find a way to approximate for those missing values!

# ### Taking care of missing values
# ##### Dropping missing values
# So let's just go ahead and drop all the variables that aren't relevant for predicting survival. We should at least keep the following:
# - Survived - This variable is obviously relevant.
# - Pclass - Does a passenger's class on the boat affect their survivability?
# - Sex - Could a passenger's gender impact their survival rate?
# - Age - Does a person's age impact their survival rate?
# - SibSp - Does the number of relatives on the boat (that are siblings or a spouse) affect a person survivability? Probability
# - Parch - Does the number of relatives on the boat (that are children or parents) affect a person survivability? Probability
# - Fare - Does the fare a person paid effect his survivability? Maybe - let's keep it.
# - Embarked - Does a person's point of embarkation matter? It depends on how the boat was filled... Let's keep it.
# 
# What about a person's name, ticket number, and passenger ID number? They're irrelavant for predicting survivability. And as you recall, the cabin variable is almost all missing values, so we can just drop all of these.

# In[ ]:

titanic_data = titanic.drop(['PassengerId','Name','Ticket','Cabin'], 1)
titanic_data.head()


# Now we have the dataframe reduced down to only relevant variables, but now we need to deal with the missing values in the age variable.
# 
# #### Imputing missing values
# Let's look at how passenger age is related to their class as a passenger on the boat.

# In[ ]:

sb.boxplot(x='Pclass', y='Age', data=titanic_data, palette='hls')


# In[ ]:

titanic_data.head()


# Speaking roughly, we could say that the younger a passenger is, the more likely it is for them to be in 3rd class. The older a passenger is, the more likely it is for them to be in 1st class. So there is a loose relationship between these variables. So, let's write a function that approximates a passengers age, based on their class. From the box plot, it looks like the average age of 1st class passengers is about 37, 2nd class passengers is 29, and 3rd class pasengers is 24.
# 
# So let's write a function that finds each null value in the Age variable, and for each null, checks the value of the Pclass and assigns an age value according to the average age of passengers in that class.

# In[ ]:

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# When we apply the function and check again for null values, we see that there are no more null values in the age variable.

# In[ ]:

titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)
titanic_data.isnull().sum()


# There are 2 null values in the embarked variable. We can drop those 2 records without loosing too much important information from our dataset, so we will do that.

# In[ ]:

titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()


# ### Converting categorical variables to a dummy indicators
# The next thing we need to do is reformat our variables so that they work with the model. Specifically, we need to reformat the Sex and Embarked variables into numeric variables.

# In[ ]:

gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
gender.head()


# In[ ]:

embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embark_location.head()


# In[ ]:

titanic_data.head()


# In[ ]:

titanic_data.drop(['Sex', 'Embarked'],axis=1,inplace=True)
titanic_data.head()


# In[ ]:

titanic_dmy = pd.concat([titanic_data,gender,embark_location],axis=1)
titanic_dmy.head()


# Now we have a dataset with all the variables in the correct format!
# 
# ### Checking for independence between features

# In[ ]:

sb.heatmap(titanic_dmy.corr())  


# Fare and Pclass are not independent of each other, so I am going to drop these.

# In[ ]:

titanic_dmy.drop(['Fare', 'Pclass'],axis=1,inplace=True)
titanic_dmy.head()


# ### Checking that your dataset size is sufficient
# We have 6 predictive features that remain. The rule of thumb is 50 records per feature... so we need to have at least 300 records in this dataset. Let's check again.

# In[ ]:

titanic_dmy.info()


# Ok, we have 889 records so we are fine.

# In[ ]:

X = titanic_dmy.ix[:,(1,2,3,4,5,6)].values
y = titanic_dmy.ix[:,0].values


# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)


# ### Deploying and evaluating the model

# In[ ]:

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[ ]:

y_pred = LogReg.predict(X_test)


# In[ ]:

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# The results from the confusion matrix are telling us that 137 and 69 are the number of correct predictions. 34 and 27 are the number of incorrect predictions.

# In[ ]:

print(classification_report(y_test, y_pred))


# In[ ]:




# In[ ]:



