import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris['data'][:,[2,3]]
y = iris['target']
X = pd.DataFrame(X)
y = pd.DataFrame(y)
data = pd.merge(X,y,left_index=True,right_index=True,how='outer')
data.columns=['x1','x2','y']
h = 0.002
x_min, x_max = data.x1.min() - 0.2, data.x1.max() + 0.2
y_min, y_max = data.x2.min() - 0.2, data.x2.max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
# sns.scatterplot(data.x1, y=data.x2,hue=data.y)
X = data[['x1','x2']]
y = data.y

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2) #80%和20%划分X和y
clf = SVC(C=0.1,kernel='linear')
# clf.fit(X_train,y_train)
# y_pre = clf.predict(X)
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# # sns.scatterplot(data.x1, y=data.x2,hue=y_pre)
# plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.2)
# accuracy_score(data.y,y_pre)

# params = {'C':np.arange(0.1,1,0.1),'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
# gsearch = GridSearchCV(estimator = SVC(kernel='linear'),param_grid = params, scoring='accuracy',iid=False, cv=5)
# gsearch.fit(X,data.y)
# gsearch.best_index_, gsearch.best_params_, gsearch.best_score_


# clf = SVC(C=0.2,kernel='rbf')
# clf.fit(X,y)
# y_pre = clf.predict(X)
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# ax = sns.scatterplot(data.x1, y=data.x2,hue=y_pre)
# ax.legend(loc="lower right")
# plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.2)

scores = []
for m in range(2,X_train.size):#循环2-79
    clf.fit(X_train[:m],y_train[:m])
    y_train_predict = clf.predict(X_train[:m])
    y_val_predict = clf.predict(X_val)
    scores.append(accuracy_score(y_train_predict,y_train[:m]))
plt.plot(range(2,X_train.size),scores,c='green', alpha=0.6)
plt.savefig('./data/ml-data/mkrate.jpg')   # 保存图片
