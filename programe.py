import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# this is data
data = pd.read_csv("C:\\Users\\DELL PC\\Downloads\\penguins.csv")

data.dropna(inplace=True)

data.drop_duplicates(inplace=True)

data.rename(columns={"bill_length_mm":"bill length","bill_depth_mm":"bill depth","flipper_length_mm":"flipper length","body_mass_g":"body mass"},inplace=True)

data.drop(["species","island"],inplace=True,axis=1)

o1 = ["bill length","bill depth","flipper length","body mass","sex"]
for item in o1:
    if item=="sex":
        data[item] = data[item].astype("category")
    else:
        data[item] = data[item].astype("int")

gender = pd.get_dummies(data["sex"],drop_first=True)

data = pd.concat([data,gender],axis=1)

data.drop("sex",axis=1,inplace=True)
data.rename(columns={"MALE":"gender"},inplace=True)

x = data.drop("gender",axis=1)
y = data["gender"]

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=.20 , random_state=100)

model = LogisticRegression()
model.fit(x_train , y_train)

prediction = model.predict(x_test)

print("Accuracy")
print("--"*40)
accuracy = round(accuracy_score(y_test , prediction)*100 , 2)
print("Accuracy : " , accuracy)