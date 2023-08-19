import pandas as pd
import numpy as np 
import sklearn 
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv",sep=";")

data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"

X = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.1)



best = 0
for i in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)

    acc = linear.score(x_test,y_test)

    if acc > best:
        best = acc
        with open("studentgrades.pickle","wb") as f:
            pickle.dump(linear,f)

pickle_in = open("studentgrades.pickle","rb")
linear = pickle.load(pickle_in)

predicted = linear.predict(x_test)
print("Accuracy : {}".format(linear.score(x_test,y_test)))
for i in range(len(predicted)):
    print("Predicted val : {}, Actual val : {}".format(round(predicted[i]),y_test[i]))

plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()

