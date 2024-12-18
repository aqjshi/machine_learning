#open data/qm9filtered.npy

import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import re






def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    df = pd.DataFrame(data.tolist() if data.dtype == 'O' and isinstance(data[0], dict) else data)
    return df

def npy_preprocessor(filename):
    df = read_data(filename)
    return df['index'].values, df['inchi'].values, df['xyz'].values, df['chiral_centers'].values, df['rotation'].values









def filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    if task == 1:
        #return only chiral_length <2
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) < 2]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array

    elif task == 2:
        #only return chiral legnth < 5
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) < 5]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    elif task == 3: 
        # Step 1: Filter indices where the length of chiral_centers_array is exactly 1
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) == 1]
        
        # Step 2: Create filtered arrays for index_array, xyz_arrays, chiral_centers_array, and rotation_array
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        
        # Step 3: Modify chiral_centers_array based on the 'R' or 'S' casting rule
        # Cast 'R' if 'R' or 'Tet_CW' is found, else cast 'S' if 'S' or 'Tet_CCW' is found
        def cast_chiral(rs):
            return 'R' if re.search(r'\bR\b|\bTet_CW\b', rs[1], re.IGNORECASE) else 'S' if re.search(r'\bS\b|\bTet_CCW\b', rs[1], re.IGNORECASE) else rs[1]
        
        # Step 4: Apply the casting function to the filtered chiral_centers_array
        filtered_chiral_centers_array = [cast_chiral(chiral_centers_array[i][0]) for i in filtered_indices]
        
        # Step 5: Filter the rotation_array accordingly
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    
    elif task == 4:
        # only return chiral_length == 1
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) == 1]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    else:
        return index_array, xyz_arrays, chiral_centers_array, rotation_array


def generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    # Task 0 or Task 1: Binary classification based on the presence of chiral centers
    if task == 0 or task == 1:
        return [1 if len(chiral_centers) > 0 else 0 for chiral_centers in chiral_centers_array]
    
    # Task 2: Return the number of chiral centers
    elif task == 2:
        return [len(chiral_centers) for chiral_centers in chiral_centers_array]
    
    # Task 3: Assuming that the task is to return something from chiral_centers_array, not rotation_array
    elif task == 3:
        return [1 if 'R' in chiral_centers else 0 for chiral_centers in chiral_centers_array]
    
    # Task 4 or Task 5: Binary classification based on posneg value in rotation_array
    elif task == 4 or task == 5:
        return [1 if posneg[0] > 0 else 0 for posneg in rotation_array]

def generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    # Fix to directly return the output of generate_label
    return generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)


def rotate_xyz(xyz, angles):
    """ Rotate the xyz coordinates of molecules based on provided angles (in degrees). """
    theta_x, theta_y, theta_z = np.radians(angles)  # Convert degrees to radians
    
    # Define rotation matrices around x, y, and z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx  # Matrix multiplication (rotation is applied in the order z, y, x)
    
    # Apply the rotation matrix to the xyz coordinates (shape: n_atoms, 3)
    rotated_xyz = np.dot(xyz, R.T)  # Transpose rotation matrix for proper application
    
    return rotated_xyz
            
def reflect_wrt_plane(xyz, plane_normal=[0, 0, 1]):
    # Normalize the plane normal vector to ensure it's a unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Calculate the distance of each point to the plane along the normal
    d = np.dot(xyz, plane_normal)  # Projects xyz onto the normal

    # Reflect each point by moving it twice the distance from the plane
    reflected_xyz = xyz - 2 * np.outer(d, plane_normal)
    
    return reflected_xyz

def augment_dataset(xyz_array, label_array, rotation_angles=[(30, 45, 60), (45, 60, 75)], task=0):
    """
    Augment the dataset by applying rotation and concatenating it with the original dataset.
    
    Parameters:
    - xyz_array: Array of shape (num_samples, n_atoms, features).
    - label_array: Labels corresponding to xyz_array.
    - rotation_angles: List of tuples for rotation angles to apply.
    - task: Specifies additional augmentation, e.g., reflection.
    
    Returns:
    - Augmented xyz data and labels.
    """
    augmented_xyz = []
    augmented_labels = []

    # For each molecule, apply rotation
    for i, xyz in enumerate(xyz_array):
        augmented_xyz.append(xyz)  # Original data
        augmented_labels.append(label_array[i])  # Original label
        
        # Apply each rotation
        # for angles in rotation_angles:
        #     rotated_xyz = rotate_xyz(xyz[:, :3], angles)  # Rotate only xyz coordinates
        #     rotated_data = np.concatenate([rotated_xyz, xyz[:, 3:]], axis=1)  # Concatenate rotated xyz with features
        #     augmented_xyz.append(rotated_data)
        #     augmented_labels.append(label_array[i])  # Label remains the same

        # # If task is 4, apply reflection
        # if task == 4:
        #     # Reflect the last rotated data
        #     reflected_xyz = reflect_wrt_plane(rotated_xyz)  # Reflect xyz coordinates
        #     reflected_data = np.concatenate([reflected_xyz, xyz[:, 3:]], axis=1)  # Concatenate with original features
        #     augmented_xyz.append(reflected_data)
        #     augmented_labels.append(1 - label_array[i])  # Label change if needed for task 4

    return np.array(augmented_xyz), np.array(augmented_labels)


def augment_1d(xyz_array, label_array, n_atoms=27, rotation_angles=[(30, 45, 60)], task=0):
    augmented_xyz = []
    augmented_labels = []
    
    n_features = xyz_array.shape[1] // n_atoms  # Number of features per atom (xyz + atom type)

    # For each molecule, reshape the flattened array, apply rotation, and flatten back
    for i, xyz in enumerate(xyz_array):
        # Reshape the flattened 1D array back to [n_atoms, n_features]
        xyz_reshaped = xyz.reshape(n_atoms, n_features)
        
        # Add the original data
        augmented_xyz.append(xyz)  # Original data (already flattened)
        augmented_labels.append(label_array[i])  # Original label
        
        # Apply rotation to the xyz coordinates (first 3 columns)
        for angles in rotation_angles:
            rotated_xyz = rotate_xyz(xyz_reshaped[:, :3], angles)  # Rotate only the first 3 columns
            rotated_data = np.hstack([rotated_xyz, xyz_reshaped[:, 3:]])  # Concatenate rotated xyz with atom features
            rotated_data_flat = rotated_data.flatten()  # Flatten back to 1D
            augmented_xyz.append(rotated_data_flat)
            augmented_labels.append(label_array[i])  # Same label as the original

        if task == 4:
            reflected_xyz = reflect_wrt_plane(rotated_xyz)
            reflected_data = np.hstack([reflected_xyz, xyz_reshaped[:, 3:]])
            reflected_data_flat = reflected_data.flatten()
            augmented_xyz.append(reflected_data_flat)
            augmented_labels.append(1 - label_array[i])

    
    return np.array(augmented_xyz), np.array(augmented_labels)

def evaluate_with_f1(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for patches, labels in test_loader:
            outputs = model(patches)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = running_loss / len(test_loader)
    
    return avg_loss, accuracy, f1

