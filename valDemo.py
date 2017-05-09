#coding=utf-8
import gdal
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
class MDC(BaseEstimator):
    def __init__(self):
        pass
    def fit(self,x,y,**kwargs):
        self.classlist=np.unique(y)
        dim=x.shape[1]
        self.means=np.zeros((len(self.classlist),dim))
        for i in range(len(self.classlist)):
            thissample=x[y==self.classlist[i],:]
            self.means[i,:]=np.mean(thissample,axis=0)
    def predict(self, x):
        mins = cdist(x, self.means)
        outvalues = self.classlist[mins.argmin(axis=1)].reshape(-1,1)
        return outvalues
if __name__ == '__main__':
    rawfilename=r'D:\Desktop\ucas\遥感地学分析实验\data.tif'
    roiimgname=r'D:\Desktop\ucas\遥感地学分析实验\roi.tif'
    rawdataset=gdal.Open(rawfilename)
    roidataset=gdal.Open(roiimgname)
    Aroi=roidataset.ReadAsArray(0,0,roidataset.RasterXSize,roidataset.RasterYSize)
    Adata=rawdataset.ReadAsArray(0,0,roidataset.RasterXSize,roidataset.RasterYSize)
    classlist=np.unique(Aroi)
    classlable=Aroi[Aroi!=255]
    samples=Adata[:,Aroi!=255]
    samples=samples.swapaxes(0,1)
    cv=KFold(n_splits=10, shuffle=True)

 #   clf = svm.SVC(kernel='linear', C=1)
 #   scores = cross_val_score(clf, samples, classlable, cv=cv, n_jobs=-1)
#    print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#    gnb = GaussianNB()
#    scores = cross_val_score(gnb,samples,classlable, cv=cv, n_jobs=-1)
#    print("Maximum Likelihood Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    mdc=MDC()
    scores = cross_val_score(mdc, samples, classlable, cv=cv, n_jobs=-1, scoring='accuracy')
    print("Minimum Distance Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))