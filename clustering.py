from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import imageio
import numpy as np


#load image &
img = imageio.imread("trees.png")

#convert image into numpy array of floating-point representations of the pixels, in the range of [0,1]
img = np.array(img, dtype = np.float64)
img /= 255

#reshape image from 3D to 2D
w, h, d = img.shape
img_2d = np.reshape(img, (w * h, d))

#pick the number of clusters/distinct colors.
k_values = [2,10,50]

#fit k-means to the subset of the data and the specified number of clusters. I used random_state=0
kmeans1 = KMeans(n_clusters = k_values[0]).fit(img_2d)
#Get actual labels from all the data
labels1 = kmeans1.predict(img_2d)

kmeans2 = KMeans(n_clusters = k_values[1]).fit(img_2d)
labels2 = kmeans2.predict(img_2d)

kmeans3 = KMeans(n_clusters= k_values[2]).fit(img_2d)
labels3 = kmeans3.predict(img_2d)



def reconstruct_image(palette, labels, w, h, d):
    #reconstruct the image by assigning colors from the produced color palette to a new, w*h image.
    image = np.zeros((w, h, d))
    label = 0

    for i in range(w):
        for j in range(h):
            image[i][j] = palette[labels[label]]
            label += 1

    return image


# Display original image & reproduced images, if you'd like to check it :
'''
plt.figure(1)
plt.title('Original image')
plt.imshow(img)

plt.figure(2)
plt.title('Reconstructed image using {colors} colors'.format(colors=k_values[0]))
plt.imshow(reconstruct_image(kmeans1.cluster_centers_, labels1, w, h, d))

plt.figure(3)
plt.title('Reconstructed image using {colors} colors'.format(colors=k_values[1]))
plt.imshow(reconstruct_image(kmeans2.cluster_centers_, labels2, w, h, d))

plt.figure(4)
plt.title('Reconstructed image using {colors} colors'.format(colors=k_values[2]))
plt.imshow(reconstruct_image(kmeans3.cluster_centers_, labels3, w, h, d))
plt.show()'''