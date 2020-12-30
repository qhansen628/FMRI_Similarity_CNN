import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_image_by_voxels_array(subject):
    '''
    input str: subject e.g 'subject_1' or 'subject_2'
    originally the data will have 10 timesteps by num_voxels blocks for each label
    averages blocks of fmri scans for each image label across timesteps
    returns: a np array of shape #images,#voxels where the indicies are in the
    correspond the the label, e.g label 0 will be index 0
    '''
    #get subject subdirectory
    subject_dir = 'FMRI STUFF/'+subject+'/'
    #get labels pandas df
    labels_path = subject_dir + subject + '_indexLabelsCleanROI.csv'
    labels_df = pd.read_csv(labels_path, header=None)
    #get subject clean data (no rest or control scans)
    data_path = subject_dir + subject + '_dataMatrixCleanROI.csv'
    data_df = pd.read_csv(data_path, header=None)
    #center the fmri data
    centered_data_df = center_fmri_data(data_df,subject)


    unique_labels = sorted(labels_df[0].unique())
    assert unique_labels == list(range(len(unique_labels))), 'Missing label'
    #list of lists used to create np array
    list_of_averages = []
    for label in unique_labels:
        #get indices of dataframe
        indices = labels_df[labels_df[0] == label].index
        #get block from data (row,cols) = (num_scans,num_voxels)
        block = centered_data_df.iloc[indices]
        
        #average across time shape = (num_voxels,)
        mean_block = block.mean(axis=0)
        
        #turn to list
        mean_list = mean_block.to_list()
        #add to list_of_averages
        list_of_averages.append(mean_list)
    #return array of shape (num_images,num_voxels)
    return np.array(list_of_averages)
     

def center_fmri_data(data,subject):
    '''
    input: data, might be with or without any rest scans
    output: data - mean of all scans (including rest scans)
    '''
    subject_dir = 'FMRI STUFF/'+subject+'/'
    #get subject full data matrix (including rest scans and control scans)
    full_data_path = subject_dir + subject + '_dataMatrix.csv'
    full_data = pd.read_csv(full_data_path, header=None)
    #get mean actross trials
    mean = full_data.mean(axis=0)
    centered_data = data.subtract(mean,axis=1)
    #print(np.max(full_data),np.max(centered_data))
    return centered_data
     

def get_subject_x_image_voxel_array(do_pca=False,components=16):
    '''
    returns an array of shape(num_subjects x num_images, num_voxels)
    '''
    scaler = StandardScaler()
    res = []
    for sub_num in range(1,7):
        #get subject name e.g subject_1
        subject = 'subject_{}'.format(str(sub_num))
        print('making {}\'s array'.format(subject))
        #get (num_images,num_voxels) array
        subject_data = get_image_by_voxels_array(subject)
        scaled_array = scaler.fit_transform(subject_data)
        if do_pca:
            #scale using standard scaler
            
            print(np.mean(scaled_array[:,0]),np.var(scaled_array[:,0]))
            pca = PCA(n_components=components)
            a = pca.fit_transform(scaled_array)
            res.append(a)
        else:
            res.append(scaled_array)

    print('Concatenating arrays...')
    #res is len num_subjects 6 to be exact
    new_array = np.array(res) #shape num_images x 6, num_voxels or components

    print('Saving array npy ...')
    if do_pca:
        np_file_name = 'ROI_fmri_pca'+str(components)+'.npy'
    else:
        np_file_name = 'ROI_fmri.npy'
    np.save(np_file_name, np.array(new_array))
    print('DONE!')

def make_ROI_cca_text():
    img_paths = pd.read_csv('FMRI STUFF/image_paths.csv',header=None)
    img_path_list = []
    for i,row in img_paths.iterrows():
        img_path_list.append(row[0])
    img_path_list = img_path_list*6
    with open('ROI_train_cca.txt','w') as f:
        i = 0
        for path in img_path_list:
            f.write('../'+path)
            f.write('\n')
            i+=1
        


if __name__ == '__main__':
    
    make_ROI_cca_text()
    
    
