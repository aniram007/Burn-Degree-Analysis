from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

"""f=open("burn_dataset.csv")
f.readline()
data=np.loadtxt(fname=f,delimiter=',')
x=data[:,0:2]
y=data[:,2]
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 0)
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
"""
def classify_3d(xco,yco,zco):
    x=np.array([[1.689,6.108,-0.054],
            [-4.625,-0.386,1.114],
            [-4.381,0.469,2.082],
            [3.727,1.47,1.65],
            [-4.859,-2.085,-1.353],
            [-4.73,-1.712,-1.429],
            [4.611,0.601,0.339],
            [1.042,2.255,0.281],
            [-4.309,-0.16,0.338],
            [-5.109,-2.097,-0.669],
            [3.975,-1.93,-1.048],
            [4.663,-2.523,-0.637],
            [-0.072,1.535,-1.56],
            [3.463,-0.975,-0.175],
            [-3.357,-1.02,0.659],
            [-1.412,2.84,-1.595],
            [-1.161,2.54,3.489],
            [2.375,1.145,-4.427],
            [3.766,-2.161,0.927],
            [4.705,-3.912,2.068]])

    y=np.array([2,3,3,2,3,3,1,2,3,3,1,1,2,1,3,2,2,2,1,1])
#clf=DecisionTreeClassifier(max_depth = 2)  #Decision tree classifier
    clf2 = svm.SVC(kernel='linear', C = 1.0)    #SVM Classifier
#clf.fit(x,y)
    clf2.fit(x,y)
#gnb = GaussianNB()
#gnb.fit(x,y)
    #knn = KNeighborsClassifier(n_neighbors = 3)
    #knn.fit(x,y)
#rf = RandomForestClassifier(n_estimators=20, random_state=0)
#rf.fit(x,y)
    z=np.array([xco,yco,zco])
    z=z.reshape(1,-1)
#print(clf.predict(z))
    return(clf2.predict(z))
#print(gnb.predict(z))
    #return(knn.predict(z))
#print(rf.predict(z))




