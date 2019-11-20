
from sklearn.datasets import make_circles
from DecisionBoundaryPlots2D import decBoundary
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.05)

from sklearn.svm import SVC
svc = SVC(C=100, gamma='scale')
svc.fit(X, y)
decBoundary(X, y, .1, svc)
