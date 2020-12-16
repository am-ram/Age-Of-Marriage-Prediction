import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
df = pd.read_csv(r'D:\Deployed Projects\age_of_marriage_data.txt')
df.drop('height',axis=1)
print(df.isnull().sum())
df.dropna(inplace=True)
X = df.loc[:,['gender','religion','caste','mother_tongue','country']]
y = df.age_of_marriage
le = LabelEncoder()
X.loc[:,['gender','religion','caste','mother_tongue','country']]= \
X.loc[:,['gender','religion','caste','mother_tongue','country']].apply(le.fit_transform)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = RandomForestRegressor(n_estimators=80,max_depth=11)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
pickle.dump(model,open('model.pkl','wb'))
print("MAE : ", mean_absolute_error(y_test,y_predict))
print("R square : ",r2_score(y_test,y_predict))