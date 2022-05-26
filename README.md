# Ex-07-Feature-Selection
# AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
Developed by: Sithi Hajara
Register number: 212221230102
```
```
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()

df.columns = boston['feature_names']
df.head()

df['PRICE']= boston['target']
df.head()

df.info()

plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()

#FILTER METHODS

X=df.drop("PRICE",1)
y=df["PRICE"]

from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape

#1.Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)

#2.Information gain/Mutual Information
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

#3.SelectKBest Model
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2) 
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))

#4.Correlation Coefficient
cor=df.corr()
sns.heatmap(cor,annot=True)

#5.Mean Absolute Difference
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='purple')

#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)

#6.Chi Square Test
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

#7.SelectPercentile method
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape

#WRAPPER METHOD

#1.Forward feature selection

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X, y)
sfs.k_feature_names_

#2.Backward feature elimination

sbs = SFS(LinearRegression(),
         k_features=10,
         forward=False,
         floating=False,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#3.Bi-directional elimination

sffs = SFS(LinearRegression(),
         k_features=(3,7),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_

#4.Recursive Feature Selection

from sklearn.feature_selection import RFE
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=7)
rfe.fit(X, y)
print(X.shape, y.shape)
rfe.transform(X)
rfe.get_params(deep=True)
rfe.support_
rfe.ranking_

#EMBEDDED METHOD

#1.Random Forest Importance

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X,y_transformed)
importances=model.feature_importances_

final_df=pd.DataFrame({"Features":pd.DataFrame(X).columns,"Importances":importances})
final_df.set_index("Importances")
final_df=final_df.sort_values("Importances")
final_df.plot.bar(color="purple")
```

# OUPUT
<img width="502" alt="170067199-7b40efaf-584f-4e65-9e13-d709ef9758e4" src="https://user-images.githubusercontent.com/94219582/170406876-4f501af9-1392-45e0-83ce-d0ea5ba766fd.png">

### Analyzing the boston dataset:
<img width="639" alt="170067266-a19d67ed-41cd-46d5-b47f-35ea04fb2a2d" src="https://user-images.githubusercontent.com/94219582/170406885-21e6ea56-29ef-439f-9d46-a96bb61b6ae3.png">



<img width="593" alt="170067355-f6714ede-d9fd-482b-ba73-1ce6626b8bd3" src="https://user-images.githubusercontent.com/94219582/170406970-5238ea97-7c9e-404e-8082-360ffdba72c0.png">



<img width="572" alt="170067397-00397dfa-0f5d-4ea3-a9fa-40b8adae86c9" src="https://user-images.githubusercontent.com/94219582/170407080-98984587-c64a-45a1-b3d0-aa8ebeece0f4.png">

### Analyzing dataset using Distplot:
<img width="658" alt="170067462-7e910061-9e75-4615-a69a-9bfe7c8b1205" src="https://user-images.githubusercontent.com/94219582/170407091-4e7d41a7-7a97-40d7-98c6-2f6c9cf3851a.png">

### Filter Methods
### Variance Threshold:
![170067587-0d4e96e6-f6c7-4db2-9a56-58dd7063093d](https://user-images.githubusercontent.com/94219582/170407137-5b963186-1a93-4894-b1ca-e2ab4d2cbc6b.png)

### Information Gain:
![170067700-5f6ea7e3-4410-4d27-a96b-eaa0cd053fdc](https://user-images.githubusercontent.com/94219582/170407271-6cdb2602-82a9-4f1c-b473-98a8c9e701ce.png)

### SelectKBest Model:
![170067807-eb1e447c-b64a-41ef-b616-365f222d11e1](https://user-images.githubusercontent.com/94219582/170407324-234ab496-afae-4baa-824a-d2cf3e8e50f1.png)

### Correlation Coefficient:
![170067879-63b15be1-d71b-472a-83f5-063ec99200a0](https://user-images.githubusercontent.com/94219582/170407373-c596950f-8952-4f35-af6c-9b4b18748771.png)

### Mean Absolute difference:
![170067959-f6e3aa39-63cc-4650-8774-4c49a6f6175f](https://user-images.githubusercontent.com/94219582/170407413-1e294837-bae6-4d90-b676-79d20143b4aa.png)

### Chi Square Test:
![170068046-db80a17b-31c6-4451-8675-87a61b3b5c98](https://user-images.githubusercontent.com/94219582/170407455-b507d923-f69c-4bdd-b6ad-34835db6bab2.png)
![170068078-7ee4b1b8-5934-48b1-b0a1-aa9f42e5a3fd](https://user-images.githubusercontent.com/94219582/170407513-fc009724-5bf1-4693-94dd-5bf6d03d3c1c.png)

### SelectPercentile Method
![170068149-ffc835a0-1280-4aab-a21c-fed9e35edf93](https://user-images.githubusercontent.com/94219582/170407542-5f7e07ec-3bcf-4288-b790-8af03de7f728.png)

### Wrapper Methods:
### Forward Feature Selection:
![170068275-33326db0-b028-4db8-b08f-15953c130372](https://user-images.githubusercontent.com/94219582/170407584-6bd669ca-1aa2-48f4-a2ca-b2dea23fd610.png)

### Backward Feature Selection:
![170068574-03b200fc-ca19-4db3-8d3a-4810f728836a](https://user-images.githubusercontent.com/94219582/170407678-6ec88c35-55ff-4e77-ab42-b615e3ae4596.png)

### Bi-Directional Elimination:
![170068638-59c1a737-efa1-4315-9b98-cd42c18285be](https://user-images.githubusercontent.com/94219582/170407741-e2e34574-e60b-46f0-9094-a4860501e19e.png)

### Recursive Feature Selection:
![170068742-3c7fb989-1fc0-4b34-a3c0-e673b0710a6b](https://user-images.githubusercontent.com/94219582/170407794-5ce35440-b50a-4628-a720-afe68b5458ec.png)

### Random Forest Importance:
![170068787-2a957043-7074-434d-96be-bac00f1f01f3](https://user-images.githubusercontent.com/94219582/170407843-741ffd24-36f4-4f5b-9e56-161276ce12a5.png)

# RESULT:
Hence various feature selection techniques are applied to the given data set successfully and saved the data into a file.
