#!/usr/bin/env python
# coding: utf-8

# # 1. Import dataset

# In[11]:


#Import các thư viện
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve


# In[12]:


#Đọc dữ liệu từ file
df = pd.read_csv('data/botnetData.csv')


# In[13]:


df.head()


# In[14]:


df.shape


# #in ra kiểu dữ liệu của mỗi dòng

# In[15]:


df.info()


# In[16]:


#kiểm tra độ chênh lệch của các nhãn trong bộ dữ liệu
print(df['botnet'].value_counts())


# # 2.Preprocessing
# 

# ## 2.1xóa cột thứ 6

# In[17]:


df.drop(columns=df.columns[6], inplace=True)


# In[18]:


df.head()


# ## Kiểm tra giá trị null

# In[19]:


print(df.isnull().sum())


# # 3.Training model LogisticRegression

# In[20]:


# Tách dữ liệu
X = df.drop(columns=['botnet'])
y = df['botnet']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[21]:


#Chuẩn hóa dữ liệu dựa theo công thức của phương pháp chuẩn standardization
sc = StandardScaler()
#Huấn luyện mô hình trên tập dữ liệu train
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[22]:


#Huấn luyện mô hình Logistic Regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[23]:


y_pred = classifier.predict(X_test)


# In[24]:


#Đánh giá mô hình thông qua mức độ chính xác (accuracy), ma trận nhầm lẫn (confusion matrix) và đường cong ROC-AUC
cm = confusion_matrix(y_test, y_pred)


# In[25]:


def base_rate_model(X):
    y = np.zeros(X.shape[0])
    return y


# In[26]:


y_base_rate = base_rate_model(X_test)
print("Mức độ chính xác của Base model là %2.2f" % accuracy_score(y_test, y_base_rate))


# In[27]:


print("Mức độ chính xác của Logistic model là %2.2f" % accuracy_score(y_test, y_pred))


# In[28]:


print("---Base Model---")
base_roc_auc = roc_auc_score(y_test, base_rate_model(X_test))
print("Base Rate AUC = %2.2f" % base_roc_auc)
print(classification_report(y_test, base_rate_model(X_test)))


# In[29]:


print("---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, y_pred)
print("Logistic Rate AUC = %2.2f" % base_roc_auc)
print(classification_report(y_test, y_pred))


# In[30]:


#Vẽ biểu đồ ROC để trực quan hóa hiệu suất mô hình
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label = 'ROC Curve (area = %0.2f)' % logit_roc_auc)
plt.plot([0,1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.xlim([0.0, 1.05])
plt.xlabel('Tỉ lệ False Positive')
plt.ylabel('Tỉ lệ True Positive')
plt.title('Mối quan hệ giữa False Positive và True Positives')
plt.legend(loc = 'lower right')
plt.show()


# In[ ]:





# In[ ]:




