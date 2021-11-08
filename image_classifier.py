import numpy as np

import os


from pathlib import Path
from keras.preprocessing import image



p=Path("dataset/images")
dir=p.glob("*")
labels_dict={'cats':0,'dogs':1,'horses':2,'Humans':3}
image_data=[]
labels=[]

for f in dir:
    label=str(f).split('/')[-1][:]
    for img_path in f.glob("*.jpg"):
        img=image.load_img(img_path,target_size=(32,32))
        img_array=image.img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[label])


image_data=np.array(image_data,dtype='float32')/255.0
image_data.shape


# therfore we have 808 images of 100x100 in 3 ie RGB standard

labels=np.array(labels)

import matplotlib.pyplot as plt



def showimg(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    

# whenever you develop any ml model's data make sure to shuffle the data otherwise in extreme situation
# it will behave as a overfitting


import random


combined=list(zip(image_data,labels))



random.shuffle(combined)
image_data[:],labels[:]=zip(*combined)



m=image_data.shape[0]
image_data=image_data.reshape(m,-1)

image_data.shape

classes=np.unique(labels)


def classwisedata(x,y):
    data={}
    for i in range(len(classes)):
        data[i]=[]
    for i in range(x.shape[0]):
        data[y[i]].append(x[i])
    for k in data.keys():
        data[k]=np.array(data[k])
    return data

data=classwisedata(image_data,labels)


def getdatapairfrom(d1,d2):
    l1,l2=d1.shape[0],d2.shape[0]
    samples=l1+l2
    features=d1.shape[1]
    data_pair=np.zeros((samples,features))
    data_labels=np.zeros((samples,))
#     now we will copy the refined one into new ds
    data_pair[:l1,:]=d1
    data_pair[l1:,:]=d2
    data_labels[:l1]=-1
    data_labels[:l1]=+1
    
    return data_pair,data_labels

from sklearn import svm
from sklearn.model_selection import GridSearchCV



params=[
    {'kernel':['linear','rbf','poly','sigmoid'],
     'C':[0.1,0.2,0.5,1.0,2.0,5.0]
    }
]

import multiprocessing

cpus=multiprocessing.cpu_count()


datax,datay=getdatapairfrom(data[0],data[1])

datax.shape

datay.shape


gs=GridSearchCV(estimator=svm.SVC(),param_grid=params,scoring="accuracy",cv=5,n_jobs=cpus)

gs.fit(datax,datay)


gs.best_estimator_

svc=svm.SVC(kernel='linear')


def trainsvm(x,y):
    svm_classifier={}
    for i in range(len(classes)):
        svm_classifier[i]={}
        for j in range(i+1,len(classes)):
            xpair,ypair=getdatapairfrom(data[i],data[j])
            svc.fit(xpair,ypair)
            svm_classifier[i][j]=(svc.coef_,svc.intercept_)
    return svm_classifier


svm_classifier=trainsvm(image_data,labels)

svm_classifier[0][1]



def binaryclassifier(x,w,b):
    z=np.dot(x,w.T)+b
    if z>=0:
        return 1
    else:
        return -1




def predict(x):
    
    count=np.zeros((len(classes),))
    for i in range(len(classes)):
        for j in range(i+1,len(classes)):
            w,b=svm_classifier[i][j]
            z=binaryclassifier(x,w,b)
            if z==1:
                count[j]+=1
            else:
                count[i]+=1
    
    v=np.argmax(count)
    return v 


for key,val in labels_dict.items():
    t=predict(image_data[100])
    if val==t:
        print(key)
        break






