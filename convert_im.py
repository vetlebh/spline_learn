import nibabel as nib
import numpy as np
import os

def read_nii(image_path, plot=False):

    # Get nibabel image object
    image = nib.load(image_path)

    # Get data from nibabel image object (returns numpy memmap object)
    image_data = image.get_fdata()

    # Convert to numpy ndarray (dtype: uint16)
    image_data_arr = np.asarray(image_data)

    return image_data_arr



folder_path = 'training.nosync'

walk = next(os.walk(folder_path))[1]

for ids in walk:

    patient_path = os.path.join(folder_path, ids)

    image_name = ids + '_frame01.nii.gz'
    image_path = os.path.join(patient_path, image_name)

    try:
        image = read_nii(image_path)
        image_save_name = 'image'
        image_save_path = os.path.join(patient_path, image_save_name)
        np.save(image_save_path, image)
        print(f'{image_save_path} saved successfully')
    except:
        print(f'{image_path} does not exist')

    gt_name = ids + '_frame01_gt.nii.gz'
    gt_path = os.path.join(patient_path, gt_name)

    try:
        gt = read_nii(gt_path)
        gt_save_name = 'gt'
        gt_save_path = os.path.join(patient_path, gt_save_name)
        np.save(gt_save_path, gt)
    except:
        print(f'{gt_path} does not exist')
