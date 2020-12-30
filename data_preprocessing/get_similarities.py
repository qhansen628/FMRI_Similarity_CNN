from preprocess_fmri import get_image_by_voxels_array
import numpy as np
import pandas as pd

#get_image_by_voxels_array(subject,PCA=False)

def get_rsm(img_by_vox_array):
    '''
    input:  array of shape (#images,#voxels) where the index of 
    each row corresponds to the index of the image the subject was looking
    at and the row contains the fmri data averaged over each time the subject
    looked at the image

    output: an array of shape (#images,#images) representing the similarity matrix
    '''

    num_imgs, _ = img_by_vox_array.shape
    #get array to store the representational similarity matrix
    rsm = np.zeros((num_imgs,num_imgs))

    for i in range(num_imgs):
        for j in range(num_imgs):
            #get array of voxels from image i
            i_voxs = img_by_vox_array[i]
            #get array of voxels from image j
            j_voxs = img_by_vox_array[j]
            #compute cosine distance between i and j
            cos_sim_ij = get_cosine_distance(i_voxs,j_voxs)
            #insert into rsm
            rsm[i][j] = cos_sim_ij
    
    return rsm

def get_mean_rsm():
    
    rsm_list = []

    for i in range(1,7):
        subject = 'subject_'+str(i)
        img_vox_array = get_image_by_voxels_array(subject)
        subject_rsm = get_rsm(img_vox_array)
        rsm_list.append(subject_rsm)
    
    rsm_array = np.array(rsm_list)
    mean_rsm = np.mean(rsm_array, axis=0)

    return mean_rsm


def get_cosine_distance(A,B):
    '''
    Input: Two vectors
    Returns: cosine distance between these two vectors'''
    #compute cosine distance
    cos_sim = 1 - np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return cos_sim

def make_training_txt():
    print('Getting image paths...')
    img_paths = pd.read_csv('FMRI STUFF/image_paths.csv',header=None)
    img_path_list = []
    for i,row in img_paths.iterrows():
        img_path_list.append(row[0])
    print('making representational similarity matrix...')
    mean_rsm = get_mean_rsm()
    print('makeing text file...')
    img_combo_similarities = []
    for i in range(mean_rsm.shape[0]):
        for j in range(mean_rsm.shape[1]):
            if i != j:
                img_path_i = '../' + img_path_list[i]
                img_path_j = '../' + img_path_list[j]
                distance = str(mean_rsm[i,j])
                img_combo_str = '{} {} {}'.format(img_path_i,img_path_j,distance)

                img_combo_similarities.append(img_combo_str)
    with open('ROI_train_distance.txt','w') as file:
        for line in img_combo_similarities:
            file.write(line + '\n')
    print('DONE')

if __name__ == '__main__':
    
    make_training_txt()