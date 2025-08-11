import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

raw_mail_data = pd.read_csv('email_origin.csv')
print("All The Raw data : ", raw_mail_data)

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

print("Mail Data Head : ", mail_data.head())
print(mail_data.shape)

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

x = mail_data['Message']
y = mail_data['Category']

print("X Data : ", x)
print("Y Data : ", y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)

print("X Data Shape : ", x.shape)
print("X Train Data Shape : ", x_train.shape)
print("X Test Data Shape : ", x_test.shape)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)


y_train = y_train.astype('int')
y_test = y_test.astype('int')

print("X Train Features : ", x_train_features)

model = LogisticRegression()
model.fit(x_train_features, y_train)

prediction_traning_data = model.predict(x_train_features)
accuracy_traning_data = accuracy_score(y_train, prediction_traning_data)

print("Accuracy Score For Train data : ", accuracy_traning_data)

prediction_test_data = model.predict(x_test_features)
accuracy_test_data = accuracy_score(y_test, prediction_test_data)

print("Accuracy Score For Test Data : ", accuracy_test_data)

input_mail = ["a large he now his is biggest discount fans reflects up in he wound presidential viagra says that to now calls escapenumber pills x escapenumbermg barack escapenumber escapenumber midday escapenumber escapenumber per item wound to see in a escapenumber pills x escapenumbermg or escapenumber escapenumber a large escapenumber escapenumber per item vicious bush massaging escapenumber pills x escapenumbermg in escapenumber escapenumber is escapenumber escapenumber per item for a the that escapenumber pills x escapenumbermg what escapenumber escapenumber his escapenumber escapenumber per item for core what escapenumber pills x escapenumbermg actually escapenumber escapenumber voted escapenumber escapenumber per item might reggie massaging escapenumber pills x escapenumbermg the escapenumber escapenumber that escapenumber escapenumber per item politics he a large escapenumber pills x escapenumbermg up in escapenumber escapenumber to see escapenumber escapenumber per item that that core but calls and now obama viagra st his in diverse in escapenumber escapenumber pills x escapenumbermg george escapenumber escapenumber for escapenumber escapenumber per item medical star he no escapenumber pills x escapenumbermg his escapenumber escapenumber medical escapenumber escapenumber per item crowd could but escapenumber pills x escapenumbermg up in escapenumber escapenumber the escapenumber escapenumber per item calls in a is escapenumber pills x escapenumbermg actually escapenumber escapenumber reno escapenumber escapenumber per item nevada student barack escapenumber pills x escapenumbermg midday escapenumber escapenumber of the escapenumber escapenumber per item crowd vicious passionate escapenumber pills x escapenumbermg crowd escapenumber escapenumber less escapenumber escapenumber per item decency thousands the escapenumber pills x escapenumbermg midday escapenumber escapenumber says escapenumber escapenumber per item mr bush in student other him fans is cialis st his mr mr a large his escapenumber pills x escapenumbermg involve escapenumber escapenumber politics escapenumber escapenumber per item for a the america escapenumber pills x escapenumbermg now escapenumber escapenumber presidential escapenumber escapenumber per item is too less practice escapenumber pills x escapenumbermg crowd escapenumber escapenumber bush escapenumber escapenumber per item park is appetite escapenumber pills x escapenumbermg wound escapenumber escapenumber is too escapenumber escapenumber per item the involve stage escapenumber pills x escapenumbermg in escapenumber escapenumber up in escapenumber escapenumber per item decency other but with star he him the crowd up in cialis student a wound now thousands escapenumber pills x escapenumbermg less escapenumber escapenumber america escapenumber escapenumber per item waiting loving or escapenumber pills x escapenumbermg whets escapenumber escapenumber escapenumber escapenumber escapenumber per item or the escapenumber pills x escapenumbermg such escapenumber escapenumber new escapenumber escapenumber per item a large like midday escapenumber pills x escapenumbermg the escapenumber escapenumber stage escapenumber escapenumber per item escapenumber but escapenumber pills x escapenumbermg the escapenumber escapenumber selfish escapenumber per item is he seductive escapenumber pills x escapenumbermg is escapenumber escapenumber that escapenumber escapenumber per item medical the sun thousands actually appetite the the crowd viagra jelly the crowd core no practice with escapenumber pills x escapenumbermg baritone escapenumber escapenumber wound escapenumber per item and for is late escapenumber pills x escapenumbermg but escapenumber escapenumber appetite escapenumber escapenumber per item escapenumber wonder the escapenumber pills x escapenumbermg him escapenumber escapenumber crowd escapenumber escapenumber per item for seductive barack escapenumber pills x escapenumbermg a large escapenumber escapenumber the escapenumber escapenumber per item medical now says the obama barack wonder less levitra wound star he loving less selfish escapenumber pills x escapenumbermg might escapenumber escapenumber crowd escapenumber escapenumber per item pull crowd wound escapenumber pills x escapenumbermg calls escapenumber escapenumber wound escapenumber escapenumber per item stage such like escapenumber pills x escapenumbermg but escapenumber escapenumber crowd escapenumber escapenumber per item calls new willis escapenumber pills x escapenumbermg such escapenumber escapenumber mr escapenumber escapenumber per item is but of the escapenumber pills x escapenumbermg obama escapenumber escapenumber but escapenumber escapenumber per item mr presidential up in for in to see crowd the soma the under stage thousands might escapenumber pills x escapenumbermg obama escapenumber escapenumber the crowd escapenumber escapenumber per item and says for escapenumber pills x escapenumbermg in escapenumber escapenumber diverse escapenumber escapenumber per item in timid like escapenumber pills x escapenumbermg seductive escapenumber escapenumber reno escapenumber escapenumber per item to in a people is late up in could george special price viagra escapenumber pills x escapenumber mg cialis escapenumber pills x escapenumber mg only escapenumber escapenumber the crowd is too him the mr"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

print("Your Mail Are : ", prediction)

if prediction[0] == 1:
    print("Your mail Are Normal Mail")
else:
    print("Your Mail Are Spam Mail")