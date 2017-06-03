#coding=utf-8
import gdal,time
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold,train_test_split
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix

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
    rawfilename=r'J:\work\UCAS\遥感地学分析实验\data.tif'
    roiimgname=r'J:\work\UCAS\遥感地学分析实验\roi.tif'
    rawdataset=gdal.Open(rawfilename)
    roidataset=gdal.Open(roiimgname)
    Aroi=roidataset.ReadAsArray(0,0,roidataset.RasterXSize,roidataset.RasterYSize)
    Adata=rawdataset.ReadAsArray(0,0,roidataset.RasterXSize,roidataset.RasterYSize)
    classlist=np.unique(Aroi)
    classlable=Aroi[Aroi!=255]
    samples=Adata[:,Aroi!=255]
    samples=samples.swapaxes(0,1)
    cv=KFold(n_splits=10, shuffle=True)

    starttime=time.time()
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, samples, classlable, cv=cv, n_jobs=-1)
    endtime=time.time()
    print("SVM Accuracy: %0.2f (+/- %0.2f), Time Cost: %0.2fs" % (scores.mean(), scores.std() * 2,endtime-starttime))

    starttime = time.time()
    gnb = GaussianNB()
    scores = cross_val_score(gnb,samples,classlable, cv=cv, n_jobs=-1)
    endtime = time.time()
    print("Maximum Likelihood Accuracy: %0.2f (+/- %0.2f), Time Cost: %0.2fs" % (scores.mean(), scores.std() * 2,endtime-starttime))

    mdc=MDC()
    cores = cross_val_score(mdc, samples, classlable, cv=cv, n_jobs=-1, scoring='accuracy')
    print("Minimum Distance Accuracy: %0.2f (+/- %0.2f), Time Cost: %0.2fs" % (scores.mean(), scores.std() * 2,endtime-starttime))

    X_train, X_test, y_train, y_test = train_test_split(samples,classlable, test_size = 0.3, random_state = 0)
    mdc = MDC()
    mdc.fit(X_train,y_train)
    ypredict=mdc.predict(X_test)
    conmatrix=confusion_matrix(y_test, ypredict,labels=[1,2,3,4,5])
    print conmatrix
    # y_test=y_test.reshape(-1,1)
    # scores=float(sum(y_test==ypredict))/float(len(y_test))