#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[82]:


df=pd.read_csv(r"C:\Users\shivi\Documents\SH1007.csv")
print(df)


# # Preprocessing Of Data

# In[83]:


df.head()


# In[84]:


df.tail(10)


# In[85]:


df.describe()


# In[86]:


df.describe().sum()


# # FEMALE ARE REPRESENTED BY 1

# In[87]:


df1=df.loc[df["GENDER"]==1]
print(df1)


# # MEN ARE REPRESENTED BY 0

# In[88]:


df2=df.loc[df["GENDER"]==0]
print(df2)


# In[89]:


df3=df.loc[df["INCOME"]>150000]
print(df3)


# In[90]:


df4=df.loc[df["INCOME"]<150000]
print(df4)


# In[91]:


df5=df.loc[df["BANK_ACCOUNT"]==1]
print(df5)


# In[92]:


df6=df.loc[df["BANK_ACCOUNT"]==0]
print(df6)


# In[93]:


df7=df.loc[df["AGE"]>45]
print(df7)


# In[94]:


df8=df[(df["AGE"]>18) & (df["AGE"]<45)]
print(df8)


# In[95]:


display(df1,df2,df3,df4,df5,df6)


# In[96]:


new_df1=pd.concat([df1,df4,df5],axis=0)
print(new_df1)


# In[97]:


new_df2=pd.concat([df1,df4],axis=0)
print(new_df2)


# In[99]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[100]:


X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[101]:


print(X_train)


# In[105]:


print(df[:20])


# In[106]:


print(X_test)


# In[107]:


print(y_train)


# In[108]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))


# In[109]:


print(confusion_matrix(y_test,y_pred))


# In[110]:


from sklearn.metrics import accuracy_score
print("accuracy is",accuracy_score(y_pred,y_test))


# In[111]:


from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))


# In[112]:


print(confusion_matrix(y_test,y_pred))


# In[113]:


from sklearn.metrics import accuracy_score
print("accuracy is",accuracy_score(y_pred,y_test))


# In[114]:


from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))


# In[115]:


print(confusion_matrix(y_test,y_pred))


# In[116]:


from sklearn.metrics import accuracy_score
print("accuracy is",accuracy_score(y_pred,y_test))


# In[117]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))


# In[118]:


print(confusion_matrix(y_test,y_pred))


# In[119]:


from sklearn.metrics import accuracy_score
print("accuracy is",accuracy_score(y_pred,y_test))


# In[120]:


pip install scikit learn


# In[121]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[122]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))


# In[123]:


print(confusion_matrix(y_test,y_pred))


# In[124]:


from sklearn.metrics import accuracy_score
print("accuracy is",accuracy_score(y_pred,y_test))


# In[125]:


from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[126]:


from sklearn.naive_bayes import ComplementNB
classifier = ComplementNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[127]:


from sklearn.metrics import accuracy_score, log_loss
import pandas as pd
classifiers = [
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB(),
    ComplementNB(),               
                  ]
 
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
 
for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    log_entry = pd.DataFrame([[name, acc*100, 11]], columns=log_cols)
    log = log.append(log_entry)
    
    print("="*30)


# In[129]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()


# In[ ]:




