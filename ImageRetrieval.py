#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage.io import imread_collection
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from scipy.stats import kendalltau


# # Importing images for trainset and testset

# In[ ]:


trainset= np.array(imread_collection('/home/myron/Desktop/vision/Μέρος Β-2020-2021/DataBase/*.jpg'),dtype=np.float64)/255
testset= np.array(imread_collection('/home/myron/Desktop/vision/Μέρος Β-2020-2021/test/*.jpg'),dtype=np.float64)/255


# # Creating Vectors of 100x100x3 and normalize

# In[ ]:


trainset_n=trainset.reshape(100,3*100*100)
testset_n=testset.reshape(11,3*100*100)


# In[ ]:


scaler=StandardScaler()
trainset_norm = scaler.fit_transform(trainset_n)


# # PCA with 100,50,10 components

# In[ ]:


pca100 = PCA(n_components=100)
dataset100=pca100.fit_transform(trainset_norm)
test100=pca100.transform(testset_n)

pca50 = PCA(n_components=50)
dataset50=pca50.fit_transform(trainset_norm)
test50=pca50.transform(testset_n)

pca10 = PCA(n_components=10)
dataset10=pca10.fit_transform(trainset_norm)
test10=pca10.transform(testset_n)


# # Variance kept for each PCA aproach

# In[ ]:


print('Variance kept with 100 principal components:{:.5f}'.format(sum(pca100.explained_variance_ratio_)))
print('Variance kept with 50 principal components:{:.5f}'.format(sum(pca50.explained_variance_ratio_)))
print('Variance kept with 10 principal components:{:.5f}'.format(sum(pca10.explained_variance_ratio_)))


# # Finding the images

# In[ ]:


def FindImages(target,dataset):
    cor={}
    for i,img in  enumerate(dataset):
        #corr, _ = kendalltau(target, img)
        corr, _ = spearmanr(target,img)
        cor[i]=corr
    return sorted(cor.items(), key=lambda x: x[1], reverse=True)[0][0]


# In[ ]:


pairs=[(pca100,dataset100,test100),(pca50,dataset50,test50),(pca10,dataset10,test10)]
for p in pairs:
    for k in range(11):
        output=FindImages(p[2][k],p[1])
        img=p[0].inverse_transform(p[1][output])
        img=(scaler.inverse_transform(img).reshape(100,100,3) * 255).astype(np.uint8)
        plt.imshow(img)
        plt.show()


# # Grayscale PCA

# In[ ]:


from skimage.color import rgb2gray

gray_trainset=np.array([rgb2gray(img)for img in trainset])
gray_testset=np.array([rgb2gray(img) for img in testset])

trainset_g_n=gray_trainset.reshape(100,-1)
testset_g_n=gray_testset.reshape(11,-1)

scaler_g=StandardScaler()
trainset_g_norm = scaler_g.fit_transform(trainset_g_n)

pca100g = PCA(n_components=100)
dataset100g=pca100g.fit_transform(trainset_g_norm )
test100g=pca100g.transform(testset_g_n)

pca50g = PCA(n_components=50)
dataset50g=pca50g.fit_transform(trainset_g_norm)
test50g=pca50g.transform(testset_g_n)

pca10g = PCA(n_components=10)
dataset10g=pca10g.fit_transform(trainset_g_norm)
test10g=pca10g.transform(testset_g_n)

print('Variance kept with 100 principal components:{:.5f}'.format(sum(pca100.explained_variance_ratio_)))
print('Variance kept with 50 principal components:{:.5f}'.format(sum(pca50.explained_variance_ratio_)))
print('Variance kept with 10 principal components:{:.5f}'.format(sum(pca10.explained_variance_ratio_)))


# In[ ]:


for i in gray_testset:
    plt.imshow(i,cmap='gray')
    plt.show()


# In[ ]:


pairs=[(pca100g,dataset100g,test100g),(pca50g,dataset50g,test50g),(pca10g,dataset10g,test10g)]
for p in pairs:
    for k in range(11):
        output=FindImages(p[2][k],p[1])
        img=p[0].inverse_transform(p[1][output])
        img=(scaler_g.inverse_transform(img).reshape(100,100) * 255).astype(np.uint8)
        plt.imshow(img,cmap='gray')
        plt.show()


# # Autoencoder 

# In[ ]:


import torch
from torch import nn
from torch.utils.data import DataLoader


# In[ ]:


input_size=3*100*100
learning_rate=1e-4
batch_size=5


# In[ ]:


Trainset=torch.tensor(trainset_n,dtype=torch.float)
Testset=torch.tensor(testset_n,dtype=torch.float)
dl_train=DataLoader(Trainset,batch_size,shuffle=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[ ]:


class AutoEncoder(nn.Module):
    def __init__(self,SpaceSize):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size,SpaceSize),
            nn.LeakyReLU(0.20))
            
        self.decoder = nn.Sequential(
            nn.Linear(SpaceSize,input_size),
            nn.Tanh()
            )
    def forward(self, x):
      x = self.encoder(x)
      #print(x.shape)
      x = self.decoder(x)
      #print(x.shape)
      return x

criterion = nn.MSELoss()
model100=AutoEncoder(100).to(device)
model50=AutoEncoder(50).to(device)
model10=AutoEncoder(10).to(device)
optimizer100 = torch.optim.Adam(model100.parameters(), lr=1e-4)
optimizer50= torch.optim.Adam(model50.parameters(), lr=1e-4)
optimizer10 = torch.optim.Adam(model10.parameters(), lr=1e-4)
Error100,Error50,Error10=[],[],[]


# In[ ]:


Epochs=2000
for epoch in range(Epochs):
    for data in dl_train:
        data=data.to(device)
        # ===================forward=====================
        output = model100(data)
        loss = criterion(output,data)
        # ===================backward====================
        optimizer100.zero_grad()
        loss.backward()
        with torch.no_grad():
            optimizer100.step()
    # ===================log========================
    if epoch%20==0:
        Error100.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, Epochs, loss.item()))


# In[ ]:


dataset100=model100.encoder(Trainset.to(device))
for k in range(11):
    target=model100.encoder(Testset.to(device))[k]
    p=FindImages(target.detach().cpu(),dataset100.detach().cpu())
    img=model100.decoder(dataset100[p]).detach().cpu().reshape(100,100,3)
    img[img>1]=0
    img[img<0]=0
    plt.imshow(img)
    plt.show()


# In[ ]:


Epochs=2000
for epoch in range(Epochs):
    for data in dl_train:
        data=data.to(device)
        # ===================forward=====================
        output = model50(data)
        loss = criterion(output,data)
        # ===================backward====================
        optimizer50.zero_grad()
        loss.backward()
        with torch.no_grad():
            optimizer50.step()
    # ===================log========================
    if epoch%20==0:
        Error50.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch , Epochs, loss.item()))


# In[ ]:


dataset50=model50.encoder(Trainset.to(device))
for k in range(11):
    target=model50.encoder(Testset.to(device))[k]
    p=FindImages(target.detach().cpu(),dataset50.detach().cpu())
    img=model50.decoder(dataset50[p]).detach().cpu().reshape(100,100,3)
    img[img>1]=0
    img[img<0]=0
    plt.imshow(img)
    plt.show()


# In[ ]:


Epochs=2000
for epoch in range(Epochs):
    for data in dl_train:
        data=data.to(device)
        # ===================forward=====================
        output = model10(data)
        loss = criterion(output,data)
        # ===================backward====================
        optimizer10.zero_grad()
        loss.backward()
        with torch.no_grad():
            optimizer10.step()
    # ===================log========================
    if epoch%20==0:
        Error10.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, Epochs, loss.item()))


# In[ ]:


dataset10=model10.encoder(Trainset.to(device))
for k in range(11):
    target=model10.encoder(Testset.to(device))[k]
    p=FindImages(target.detach().cpu(),dataset10.detach().cpu())
    img=model10.decoder(dataset10[p]).detach().cpu().reshape(100,100,3)
    img[img>1]=0
    img[img<0]=0
    plt.imshow(img)
    plt.show()


# In[ ]:


x=[20*r for r in range(0,100)]
plt.plot(x,Error10,'r',label='10')
plt.plot(x,Error50,'g',label='50')
plt.plot(x,Error100,'b',label='100')
plt.title('Loss functions')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# # Autoencoder with grayscale

# In[ ]:


gray_trainset=np.array([rgb2gray(img)for img in trainset])
gray_testset=np.array([rgb2gray(img) for img in testset])

trainset_g_n=gray_trainset.reshape(100,100*100)
testset_g_n=gray_testset.reshape(11,100*100)

Trainset_g=torch.tensor(trainset_g_n,dtype=torch.float)
Testset_g=torch.tensor(testset_g_n,dtype=torch.float)
dl_train=DataLoader(Trainset_g,batch_size,shuffle=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size=100*100

criterion = nn.MSELoss()
model100g=AutoEncoder(100).to(device)
model50g=AutoEncoder(50).to(device)
model10g=AutoEncoder(10).to(device)
optimizer100g = torch.optim.Adam(model100g.parameters(), lr=1e-4)
optimizer50g= torch.optim.Adam(model50g.parameters(), lr=1e-4)
optimizer10g = torch.optim.Adam(model10g.parameters(), lr=1e-4)
Error100g,Error50g,Error10g=[],[],[]


# In[ ]:


Epochs=2000
for epoch in range(Epochs):
    for data in dl_train:
        data=data.to(device)
        # ===================forward=====================
        output = model100g(data)
        loss = criterion(output,data)
        # ===================backward====================
        optimizer100g.zero_grad()
        loss.backward()
        with torch.no_grad():
            optimizer100g.step()
    # ===================log========================
    if epoch%20==0:
        Error100g.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, Epochs, loss.item()))


# In[ ]:


dataset100=model100g.encoder(Trainset_g.to(device))
for k in range(11):
    target=model100g.encoder(Testset_g.to(device))[k]
    p=FindImages(target.detach().cpu(),dataset100.detach().cpu())
    img=model100g.decoder(dataset100[p]).detach().cpu().reshape(100,100)
    img[img>1]=0
    img[img<0]=0
    plt.imshow(img,cmap='gray')
    plt.show()


# In[ ]:


Epochs=2000
for epoch in range(Epochs):
    for data in dl_train:
        data=data.to(device)
        # ===================forward=====================
        output = model50g(data)
        loss = criterion(output,data)
        # ===================backward====================
        optimizer50g.zero_grad()
        loss.backward()
        with torch.no_grad():
            optimizer50g.step()
    # ===================log========================
    if epoch%20==0:
        Error50g.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch , Epochs, loss.item()))


# In[ ]:


dataset50=model50g.encoder(Trainset_g.to(device))
for k in range(11):
    target=model50g.encoder(Testset_g.to(device))[k]
    p=FindImages(target.detach().cpu(),dataset50.detach().cpu())
    img=model50g.decoder(dataset50[p]).detach().cpu().reshape(100,100)
    img[img>1]=0
    img[img<0]=0
    plt.imshow(img,cmap='gray')
    plt.show()


# In[ ]:


Epochs=2000
for epoch in range(Epochs):
    for data in dl_train:
        data=data.to(device)
        # ===================forward=====================
        output = model10g(data)
        loss = criterion(output,data)
        # ===================backward====================
        optimizer10g.zero_grad()
        loss.backward()
        with torch.no_grad():
            optimizer10g.step()
    # ===================log========================
    if epoch%20==0:
        Error10.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, Epochs, loss.item()))


# In[ ]:


dataset10=model10g.encoder(Trainset_g.to(device))
for k in range(11):
    target=model10g.encoder(Testset_g.to(device))[k]
    p=FindImages(target.detach().cpu(),dataset10.detach().cpu())
    img=model10g.decoder(dataset10[p]).detach().cpu().reshape(100,100)
    img[img>1]=0
    img[img<0]=0
    plt.imshow(img,cmap='gray')
    plt.show()

