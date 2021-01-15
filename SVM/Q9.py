
from sklearn import svm, datasets
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, 2:]

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import copy


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, 2:]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (OneVsRestClassifier(svm.SVC(kernel='linear', C=C)),
          OneVsOneClassifier(svm.SVC(kernel='linear', C=C)),
          OneVsRestClassifier(svm.SVC(kernel='rbf', C=C)),
          OneVsRestClassifier(svm.SVC(kernel='poly', degree=3, C=C)) )
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel, OneVsRest',
          'SVC with linear kernel, OneVsOne',
          'SVC with RBF kernel, OneVsRest',
          'SVC with polynomial (degree 3) kernel, OneVsRest')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2, figsize=(15,15))

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()

C = 1.0  # SVM regularization parameter
models_metrics = (OneVsRestClassifier(svm.SVC(kernel='linear', C=C)),
                  OneVsOneClassifier(svm.SVC(kernel='linear', C=C)),
                  OneVsRestClassifier(svm.SVC(kernel='rbf', C=C)),
                  OneVsRestClassifier(svm.SVC(kernel='poly', degree=3, C=C)))
models_metrics = (clf.fit(X, y) for clf in models_metrics)
for clf, title in zip(models_metrics, titles):
    print(title)
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred))
    plot_confusion_matrix(clf, X, y)
    plt.show()
    print('###################################')
    print()
    print()
