import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
import sys
import re
from sklearn.metrics import f1_score, precision_score, recall_score


def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    df = pd.DataFrame(data.tolist() if data.dtype == 'O' and isinstance(data[0], dict) else data)
    return df


def npy_preprocessor(filename):
    df = read_data(filename)
    return df['index'].values, df['inchi'].values, df['xyz'].values, df['chiral_centers'].values, df['rotation'].values


def filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    if task == 0:
        filtered_indices = [i for i in range(len(index_array))]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array

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
        # Step 1: Filter indices where the length of chiral_centers_array is exactly 1 and the first tuple contains 'R' or 'S'
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) == 1 and ('R' == chiral_centers_array[i][0][1] or 'S' == chiral_centers_array[i][0][1])]
        # Step 2: Create filtered arrays for index_array, xyz_arrays, chiral_centers_array, and rotation_array
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        
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
    elif task == 5:
        filtered_indices = [i for i in range(len(index_array))]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array


def generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    # Task 0 or Task 1: Binary classification based on the presence of chiral centers
    if task == 0 or task == 1:
        return [1 if len(chiral_centers) > 0 else 0 for chiral_centers in chiral_centers_array]
    
    # Task 2: Return the number of chiral centers
    elif task == 2:
        return [len(chiral_centers) for chiral_centers in chiral_centers_array]
    
    # Task 3: Assuming that the task is to return something from chiral_centers_array, not rotation_array
    elif task == 3:
        return [
            1 if chiral_centers and len(chiral_centers[0]) > 1 and 'R' == chiral_centers[0][1] else 0
            for chiral_centers in chiral_centers_array
        ]

    # Task 4 or Task 5: Binary classification based on posneg value in rotation_array
    elif task == 4 or task == 5:
        return [1 if posneg[0] > 0 else 0 for posneg in rotation_array]

def generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    # Fix to directly return the output of generate_label
    return generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)

# 121416 item, each associated with a 27 row, 8 col matrix, apply global normalization to col 0,1,2 Rescaling data to a [0, 1]

def reflect_wrt_plane(xyz, plane_normal=[0, 0, 1]):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    d = np.dot(xyz, plane_normal)
    return xyz - 2 * np.outer(d, plane_normal)

def rotate_xyz(xyz, angles):
    theta_x, theta_y, theta_z = np.radians(angles)
    Rx = np.array([[1,0,0],
                   [0,np.cos(theta_x),-np.sin(theta_x)],
                   [0,np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[ np.cos(theta_y),0,np.sin(theta_y)],
                   [0,1,0],
                   [-np.sin(theta_y),0,np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z),-np.sin(theta_z),0],
                   [np.sin(theta_z), np.cos(theta_z),0],
                   [0,0,1]])
    R = Rz @ Ry @ Rx
    return np.dot(xyz, R.T)

def split_data(index_array, xyz_arrays, chiral_centers_array, rotation_array):
    train_idx, test_idx = train_test_split(range(len(index_array)), test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.05, random_state=42)

    def subset(indices):
        return ([index_array[i] for i in indices],
                [xyz_arrays[i] for i in indices],
                [chiral_centers_array[i] for i in indices],
                [rotation_array[i] for i in indices])
    return subset(train_idx), subset(val_idx), subset(test_idx)

def normalize_xyz_train(xyz_arrays):
    x_array = np.array([xyz[:,0] for xyz in xyz_arrays])
    y_array = np.array([xyz[:,1] for xyz in xyz_arrays])
    z_array = np.array([xyz[:,2] for xyz in xyz_arrays])
    min_val = min(np.min(x_array), np.min(y_array), np.min(z_array))
    max_val = max(np.max(x_array), np.max(y_array), np.max(z_array))
    return min_val, max_val, [((xyz[:,:3]-min_val)/(max_val-min_val)) for xyz in xyz_arrays]

def apply_normalization(xyz_arrays, min_val, max_val):
    return [((xyz[:,:3]-min_val)/(max_val-min_val)) for xyz in xyz_arrays]

def augment_dataset(index_array, xyz_arrays, chiral_centers_array, rotation_array, label_array, task):
    aug_idx, aug_xyz, aug_chiral, aug_rot, aug_label = list(index_array), list(xyz_arrays), list(chiral_centers_array), list(rotation_array), list(label_array)
    for i in range(len(index_array)):
        if len(chiral_centers_array[i]) == 1:
            reflected_xyz = xyz_arrays[i].copy()
            reflected_xyz[:, :3] = reflect_wrt_plane(xyz_arrays[i][:, :3], [0,0,1])
            reflected_label = label_array[i]
            if task == 3: reflected_label = 1 - reflected_label
            elif task in [4,5]: reflected_label = -reflected_label
            aug_idx.append(index_array[i])
            aug_xyz.append(reflected_xyz)
            aug_chiral.append(chiral_centers_array[i])
            aug_rot.append(rotation_array[i])
            aug_label.append(reflected_label)
    return aug_idx, aug_xyz, aug_chiral, aug_rot, aug_label

# take in argument from shell task
task = int(sys.argv[1])


# Neural Network Model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        # Move layers to the specified device
        self.fc1 = nn.Linear(input_size, hidden_size).to(device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size).to(device)
        self.device = device
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# Accuracy and Metrics Calculation
def evaluate_model(model, test_loader, criterion, task, test):
    model.eval()
    all_labels = []
    all_predictions = []
    running_loss = 0.0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # Reshape to Nx1x27x8 for CNN
            data = data.unsqueeze(1)  # Add channel dimension

            outputs = model(data)

            if task == 2:
                loss = criterion(outputs, labels.long())
                predictions = torch.argmax(outputs, dim=1)
            else:
                # Binary classification
                outputs = torch.sigmoid(outputs)
                
                # Ensure the output and labels have the same shape
                outputs = outputs.view(-1)  # Flatten model output to match labels
                labels = labels.view(-1)   # Flatten labels for consistency
                
                loss = criterion(outputs, labels)
                predictions = (outputs > 0.5).float()

            running_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    average_type = 'macro' if task == 2 else 'binary'
    precision = precision_score(all_labels, all_predictions, average=average_type, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=average_type, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average=average_type, zero_division=0)
    if test:
        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)
    return avg_loss, accuracy, precision, recall, f1

# Main Training Function
def train_model(train_data, val_data, test_data, task, num_epochs=50, batch_size=8, learning_rate=0.0001):
    # Unpack and normalize data
    train_idx, train_xyz, _, _ = train_data
    val_idx, val_xyz, _, _ = val_data
    test_idx, test_xyz, _, _ = test_data

    min_val, max_val, train_xyz = normalize_xyz_train(train_xyz)
    val_xyz = apply_normalization(val_xyz, min_val, max_val)
    test_xyz = apply_normalization(test_xyz, min_val, max_val)



    # Convert to tensors
    train_labels = torch.tensor(generate_label_array(train_idx, [], train_data[2], train_data[3], task), dtype=torch.float32)
    val_labels = torch.tensor(generate_label_array(val_idx, [], val_data[2], val_data[3], task), dtype=torch.float32)
    test_labels = torch.tensor(generate_label_array(test_idx, [], test_data[2], test_data[3], task), dtype=torch.float32)


    train_tensor = torch.tensor(np.array(train_xyz), dtype=torch.float32)
    val_tensor = torch.tensor(np.array(val_xyz), dtype=torch.float32)
    test_tensor = torch.tensor(np.array(test_xyz), dtype=torch.float32)


    train_tensor = train_tensor.view(train_tensor.size(0), -1)  # Shape: (N, 27*3)
    val_tensor = val_tensor.view(val_tensor.size(0), -1)
    test_tensor = test_tensor.view(test_tensor.size(0), -1)


    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset = TensorDataset(val_tensor, val_labels)
    test_dataset = TensorDataset(test_tensor, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    input_size = train_tensor.shape[1]
    output_size = 5 if task == 2 else 1
    model = FFNN(input_size, 256, output_size).to(device)

    criterion = nn.CrossEntropyLoss() if task == 2 else nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation on validation data
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, task, test=False)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

    # Final Test Evaluation
    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, task, test=True)
    print(f"Final Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    return model

# Load and preprocess data
index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor('qm9_filtered.npy')
if task == 3 or task == 4 or task == 5:
    #print distribution of labels as a ratio
    print("Distribution of Labels:")
    print(pd.Series(generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)).value_counts(normalize=True))

# shell arg for task
task = int(sys.argv[1])

print("\nTASK:", task)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
filtered_index, filtered_xyz, filtered_chiral, filtered_rotation = filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)
train_data, val_data, test_data = split_data(filtered_index, filtered_xyz, filtered_chiral, filtered_rotation)

model = train_model(train_data, val_data,  test_data, num_epochs=1, task=task)