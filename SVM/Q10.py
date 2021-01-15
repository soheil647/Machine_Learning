import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# #############################################################################
# Generate Input Sample
def y_(x):
    return 5 * np.exp(x) + 3


X = np.sort(np.random.random(100) * 10, axis=0).reshape(-1, 1)
y = y_(X).ravel()

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

# #############################################################################
# Look at the results
lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
    axes[ix].legend(fancybox=True, shadow=True)

    axes[ix].title.set_text('Error: ' + str(mean_absolute_error(svr.predict(X), y)))

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()


# #############################################################################
# #############################################################################
# #############################################################################

import io
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df = []
with io.open("housing_data.txt", mode="r", encoding="utf-8") as f:
    for line in f:
        df.append(line.split())

columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.DataFrame(df, columns=columns)
df.MEDV = df.MEDV.astype(float)

df = df.sort_values('MEDV').reset_index(drop=True)
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# display(X)
# display(y)
# #############################################################################
# Fit regression model
# svr_rbf = SVR(kernel='rbf', C=100,  gamma='auto').fit(X, y)
# svr_lin = SVR(kernel='linear', C=100,  gamma='auto').fit(X, y)
svr_poly = SVR(kernel='poly', C=1, degree=3, gamma='auto').fit(X, y)

svr_rbf = SVR(kernel='rbf', C=1).fit(X, y)
# svr_lin = SVR(kernel='linear', C=1).fit(X, y)
# svr_poly = SVR(kernel='poly', C=1).fit(X, y)
# #############################################################################

# svrs = [svr_rbf, svr_lin, svr_poly]
# kernel_label = ['RBF', 'Linear', 'Polynomial']
# model_color = ['m', 'c', 'g']

# for ix, svr in enumerate(svrs):
y_pred = svr_rbf.predict(X)
plt.figure(figsize=(8,8))
plt.scatter(y, y_pred, color='g', label='RBF model')

plt.title('Error: ' + str(mean_absolute_error(y, y_pred)) )
plt.legend()
plt.show()