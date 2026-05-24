# ============================================================================
# File: util.py
# Date: 2026-03-27
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: list):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []
    for path in img_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = img.resize((16, 16))  # Resize to 16x16
        img = np.array(img).astype(np.float32)  # Convert to numpy array
        feat = img.flatten()  # Flatten the image
        feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-8) # Normalize
        tiny_img_feats.append(feat)

    tiny_img_feats = np.array(tiny_img_feats) # Convert list to ndarray

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################
    # print(tiny_img_feats.shape)
    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 400
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    NOTE:
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better
           (to a point) but be slower to compute,
           you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    #   5. If you have trouble installing cyvlfeat, you can use cv2 for sift & 
    #      sklearns for kmeans
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################
    all_descriptors = []
    for path in img_paths:
        img = Image.open(path).convert('L')
        img = np.array(img).astype(np.float32)
        _, descriptors = dsift(img, step=[5, 5], fast=True) # Get SIFT features with a step size of 5
        all_descriptors.append(descriptors)
    
    all_descriptors = np.vstack(all_descriptors).astype(np.float32) # Stack all descriptors into a single array
    vocab = kmeans(all_descriptors, num_centers=vocab_size) # Perform k-means clustering to get the vocabularies (cluster centers)
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    # print(vocab.shape)
    return vocab

###### Step 1-b-2

# histogram of vocabularies (cluster centers) on this image
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []
    vocab_size = vocab.shape[0]
    for path in img_paths:
        # 1. load image and get SIFT feature
        img = Image.open(path).convert('L')
        img = np.array(img).astype(np.float32)
        _, descriptors = dsift(img, step=[5, 5], fast=True)

        # 2. calculate the distances between these features and cluster centers
        dists = cdist(descriptors, vocab)

        # 3. assign each local feature to its nearest cluster center
        word_indices = np.argmin(dists, axis=1)

        # 4. build a histogram indicating how many times each cluster presents
        hist, _ = np.histogram(word_indices, bins=np.arange(vocab_size + 1))

        # 5. normalize the histogram by number of features
        hist = hist.astype(np.float32)
        hist = hist / (np.sum(hist) + 1e-8)

        img_feats.append(hist)
        
    img_feats = np.array(img_feats)
    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of training images
        train_labels (N): list of string of ground truth category for each 
                          training image
        test_img_feats (M, d): ndarray of feature of testing images
    Returns:
        test_predicts (M): list of string of predict category for each 
                           testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################
    test_predicts = []
    k = 5  # number of nearest neighbors to consider

    # 1. 計算距離矩陣 (M, N)
    dists = cdist(test_img_feats, train_img_feats, metric='minkowski', p=1)

    # 2. 對每個 test image
    for i in range(dists.shape[0]):

        # 3. 找到距離最小的 k 個 train index
        nearest_idx = np.argsort(dists[i])[:k]

        # 4. 取出這 k 個 neighbors 的 labels
        nearest_labels = [train_labels[idx] for idx in nearest_idx]

        # 5. vote for the most common label among the neighbors
        counts = {}
        for label in nearest_labels:
            counts[label] = counts.get(label, 0) + 1

        # 找最多票的 label
        pred_label = max(counts, key=counts.get)

        test_predicts.append(pred_label)


    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################
    
    return test_predicts
