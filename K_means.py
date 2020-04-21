import numpy as np
from skimage import io, img_as_float
import os

def k_means_clustering(image_vectors, k, num_iterations):
    val = np.full((image_vectors.shape[0],), -1)
    clstr_proto = np.random.rand(k, 3)
    for i in range(num_iterations):
        print('Iteration: ' + str(i + 1))
        points_label = [None for k_i in range(k)]


        for rgb_i, rgb in enumerate(image_vectors):
           
            rgb_row = np.repeat(rgb, k).reshape(3, k).T

          
            closest_label = np.argmin(np.linalg.norm(rgb_row - clstr_proto, axis=1))
            val[rgb_i] = closest_label

            if (points_label[closest_label] is None):
                points_label[closest_label] = []

            points_label[closest_label].append(rgb)

       
        for k_i in range(k):
            if (points_label[k_i] is not None):
                new_cluster_prototype = np.asarray(points_label[k_i]).sum(axis=0) / len(points_label[k_i])
                clstr_proto[k_i] = new_cluster_prototype

    return (val, clstr_proto)

def closest_centroids(X,c):
    K = np.size(c,0)
    idx = np.zeros((np.size(X,0),1))
    arr = np.empty((np.size(X,0),1))
    for i in range(0,K):
        y = c[i]
        temp = np.ones((np.size(X,0),1))*y
        b = np.power(np.subtract(X,temp),2)
        a = np.sum(b,axis = 1)
        a = np.asarray(a)
        a.resize((np.size(X,0),1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr,0,axis=1)
    idx = np.argmin(arr, axis=1)
    return idx

def compute_centroids(X,idx,K):
    n = np.size(X,1)
    centroids = np.zeros((K,n))
    for i in range(0,K):
        ci = idx
        ci = ci.astype(int)
        total_number = sum(ci);
        ci.resize((np.size(X,0),1))
        total_matrix = np.matlib.repmat(ci,1,n)
        ci = np.transpose(ci)
        total = np.multiply(X,total_matrix)
        centroids[i] = (1/total_number)*np.sum(total,axis=0)
    return centroids

if __name__ == '__main__':
    
    
    img = ["Koala.jpg","Penguins.jpg"]
    itr = 5
    
    for J in img:
        print("Working with " + str(J))
        
        image = io.imread(J)[:, :, :3] 
        image = img_as_float(image)
    
        image_dimensions = image.shape
        image_name = image
    
        image_vectors = image.reshape(-1, image.shape[-1])
        
        k = [2,5,10,15,20]
    
        for j in k:
            
            print("\nK Value is " + str(j) + "\n")
            
            lbs, color_centroids = k_means_clustering(image_vectors, k=j, num_iterations=itr)
        
            output_image = np.zeros(image_vectors.shape)
            
            for i in range(output_image.shape[0]):
                output_image[i] = color_centroids[lbs[i]]
        
            output_image = output_image.reshape(image_dimensions)
            print('\nSaving the Compressed Image')
            io.imsave('Compressed_'+str(J)+'_'+str(j)+'.jpg' , output_image)
            info_original = os.stat(J)        
            info_compress = os.stat('Compressed_'+str(J)+'_'+str(j)+'.jpg')
            
            compressionrate = info_original.st_size/info_compress.st_size
            print(round(compressionrate,2))          