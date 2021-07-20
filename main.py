
import numpy as np
import cv2
import pandas as pd


img = cv2.imread('./Train_Sandstone_Versa0000.tif')


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

df = pd.DataFrame()

# add original pizel values to the data fram eas feature #1
img2 = img.reshape(-1)
df['Original Image'] = img2

#print(df.head())

# add other features

########################################################################################
# First set - gabor features
num = 1 # to count numbers up in order to give Gabos features a label in the data
kernels = []
for theta in range(2):              # define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1,3):             # sigma with 1 and 3
         for lamda in np.arange(0, np.pi, np.pi/4):     # range of wavelengths
             for gamma in (0.05, 0.5):                   # gamma values of 0.05 and 0.5
                
                gabor_label = 'Gabor' + str(num)       # label Gabor columns as Gabor1, Gabor2 etc
                ksize = 5
                kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)

                # now filter the image and add values to a new column
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img
                num += 1

#print(df.head())

########################################################################################

# canny edge
edges = cv2.Canny(img, 100, 200)
edges1 = edges.reshape(-1)
df['Canny Edge'] = edges1

#print(df.head())

from skimage.filters import roberts, sobel, scharr, prewitt

edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts1

edge_sobel = sobel(img)
edge_sobel1 = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel1
 
edge_scharr  = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr1

edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt1

# Gaussian with sigma=3
from scipy import ndimage as nd
gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1


# Gaussian with sigma=7
from scipy import ndimage as nd
gaussian_img2 = nd.gaussian_filter(img, sigma=7)
gaussian_img3 = gaussian_img2.reshape(-1)
df['Gaussian s7'] = gaussian_img3

# median with sigma=3
median_img = nd.median_filter(img, size=3)
median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1

# variance with size=3
variance_img = nd.generic_filter(img, np.var, size=3)
variance_img1 = variance_img.reshape(-1)
df['Variance s3'] = variance_img1

#print(df.head())

labeled_img = cv2.imread('./Sandstone_Versa0000.tif')

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Label'] = labeled_img1

#print(df.head())
##################################################################

# define dependent and independent variables
Y = df['Label'].values
X = df.drop(labels = ['Label'], axis=1)

# splitting train_test_data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)


# train a random forest classifiers using the above DataFrame
# import ML algorithm and train the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=42)

model.fit(X_train, Y_train)

prediction_test = model.predict(X_test)

from sklearn import metrics
print('Accuracy = ', metrics.accuracy_score(Y_test, prediction_test))

# 65 feature ranking
importances = list(model.feature_importances_)
print(importances)

features_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)

#############################################################################33
# 66 saving trained model (pickling) 

import pickle
filename = 'sandstone_model'
pickle.dump(model, open(filename, 'wb'))

load_model = pickle.load(open(filename, 'rb'))
result = load_model.predict(X)
segmented = result.reshape(img.shape)

from matplotlib import pyplot as plt
plt.imshow(segmented, cmap='jet')
plt.show()
plt.imsave('segmented_rock.jpg', segmented, cmap='jet')
        





