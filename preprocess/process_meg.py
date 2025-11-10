import os
import mne
import pickle
import glob
import torch
import shutil
import csv
import pandas as pd
import numpy as np
from collections import Counter
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('train', add_help=False)
    parser.add_argument('--subject', type=int)
    return parser.parse_args()

args = get_args_parser()

sub = args.subject
project_dir = 'data/things-meg'


def save_data(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file, protocol=4)

fif_file = os.path.join(project_dir,f"ds004212-download/derivatives/preprocessed/preprocessed_P{sub}-epo.fif")

# output_dir = os.path.join(project_dir,"ds004212-download/derivatives/preprocessed_npy")

concept_csv_file_path = os.path.join(project_dir,'../things/THINGS/Metadata/Concept-specific/image_concept_index.csv')
csv_img_file_path = os.path.join(project_dir,"../things/THINGS/Metadata/Image-specific/image_paths.csv")

origin_img_dir = os.path.join(project_dir,"../things/THINGS/Images/")

training_images_dir = os.path.join(project_dir,"Image_set/training_images")
test_images_dir = os.path.join(project_dir,"Image_set/test_images")

save_dir = os.path.join(project_dir,f"Preprocessed_data/sub-{format(sub,'02')}")

def read_and_crop_epochs(fif_file):
    epochs = mne.read_epochs(fif_file, preload=True)
    cropped_epochs = epochs.crop(tmin=0, tmax=1.0)
    return cropped_epochs

epochs = read_and_crop_epochs(fif_file)    

sorted_indices = np.argsort(epochs.events[:, 2])
epochs = epochs[sorted_indices]

print(len(epochs.events))


image_concept_df = pd.read_csv(concept_csv_file_path, header=None)
print(image_concept_df)

def filter_valid_epochs(epochs, exclude_event_id=999999):
    return epochs[epochs.events[:, 2] != exclude_event_id]
valid_epochs = filter_valid_epochs(epochs)

def identify_zs_event_ids(epochs, num_repetitions=12):
    event_ids = epochs.events[:, 2]
    unique_event_ids, counts = np.unique(event_ids, return_counts=True)
    zs_event_ids = unique_event_ids[counts == num_repetitions]
    return zs_event_ids

zs_event_ids = identify_zs_event_ids(valid_epochs)
# Verify the zero-shot event IDs
print("Zero-shot Event IDs:", zs_event_ids)

# Separate and process datasets
training_epochs = valid_epochs[~np.isin(valid_epochs.events[:, 2], zs_event_ids)]
# Verify the number of events in the training set
print("Number of events in the training set:", len(training_epochs.events))
print(len(training_epochs.events))

# Extract event IDs from the filtered training epochs
training_event_ids = np.unique(training_epochs.events[:, 2])

zs_test_epochs = valid_epochs[np.isin(valid_epochs.events[:, 2], zs_event_ids)]
zs_test_epochs.events
print(len(zs_test_epochs.events))
# zs_test_epochs.events


zs_event_to_category_map = {}
for i, event_id in enumerate(zs_event_ids):
    # Using the row index (i) to map to the image category index
    # Assuming the first event_id corresponds to the first row, second event_id to the second row, and so on
    image_category_index = image_concept_df.iloc[event_id-1, 0]  # Accessing the first (and only) column at row i
    zs_event_to_category_map[event_id] = image_category_index

test_set_categories = []
# Iterate over the event IDs in the test set
for event_id in zs_event_ids:
    if event_id in zs_event_to_category_map:
        # Get the category index from the mapping
        category_index = zs_event_to_category_map[event_id]
        test_set_categories.append(category_index)


event_to_category_map = {}
for i, event_id in enumerate(training_event_ids):
    # Using the row index (i) to map to the image category index
    # Assuming the first event_id corresponds to the first row, second event_id to the second row, and so on
    image_category_index = image_concept_df.iloc[event_id-1, 0]  # Accessing the first (and only) column at row i
    event_to_category_map[event_id] = image_category_index

train_set_categories = []
# Extract event IDs from the training set
training_event_ids = training_epochs.events[:, 2]
# Iterate over the event IDs in the training set
for event_id in training_event_ids:
    if event_id in event_to_category_map:
        # Get the category index from the mapping
        category_index = event_to_category_map[event_id]        
        train_set_categories.append(category_index)

train_set_categories_filtered = [item for item in train_set_categories if item not in test_set_categories]

# Create a mask for epochs to keep in the training set
keep_epochs_mask = [category not in test_set_categories for category in train_set_categories]
# Apply the mask to filter out epochs from training_epochs
training_epochs_filtered = training_epochs[keep_epochs_mask]

def reshape_meg_data(epochs, num_concepts, num_imgs, repetitions):
    data = epochs.get_data()
    reshaped_data = data.reshape((num_concepts * num_imgs, repetitions, data.shape[1], data.shape[2]))
    return reshaped_data

training_data = reshape_meg_data(training_epochs_filtered, num_concepts=1654, num_imgs=12, repetitions=1)
print(training_data.shape)

zs_test_data = reshape_meg_data(zs_test_epochs, num_concepts=200, num_imgs=1, repetitions=12)
print(zs_test_data.shape)





image_df = pd.read_csv(csv_img_file_path, header=None)


image_concept_df = pd.read_csv(concept_csv_file_path, header=None)


img_path_list_training = []
img_path_list_test = []

for index, row in image_df.iterrows():
    source_image_path = row[0]
    event_id = index + 1

    category_index = image_concept_df.iloc[event_id - 1, 0]
    path_parts = source_image_path.split('/')
    source_image_path = '/'.join(path_parts[1:])

    if len(path_parts) > 2:
        formatted_index = str(category_index).zfill(5)
        path_parts[1] = f"{formatted_index}_{path_parts[1]}"

    image_path = '/'.join(path_parts[1:])
    
    if event_id in training_epochs_filtered.events[:, -1]:
        target_dir = os.path.join(training_images_dir)
    elif event_id in zs_test_epochs.events[:, -1]:
        target_dir = os.path.join(test_images_dir)
    else:
        continue

    src_file = os.path.join(origin_img_dir, source_image_path)
    dest_file = os.path.join(target_dir, image_path)

    if sub==1:
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy(src_file, dest_file)
    
    if event_id in training_epochs_filtered.events[:, -1]:
        img_path_list_training.append(dest_file)
    elif event_id in zs_test_epochs.events[:, -1]:
        img_path_list_test.append(dest_file)
    else:
        continue



img_path_list_training = [path.split('data/things-meg/Image_set/')[1] for path in img_path_list_training]
img_path_list_test = [path.split('data/things-meg/Image_set/')[1] for path in img_path_list_test]

print('train img', len(img_path_list_training))
print('test_img', len(img_path_list_test))

os.makedirs(save_dir, exist_ok=True)
test_dict = {
    'eeg': zs_test_data.astype(np.float16),
    'img':img_path_list_test,
}
torch.save(test_dict, os.path.join(save_dir,'test.pt'),pickle_protocol=5)

print(img_path_list_test)

train_dict = {
    'eeg': training_data.astype(np.float16),
    'img':img_path_list_training,
}
torch.save(train_dict, os.path.join(save_dir,'train.pt'),pickle_protocol=5)
