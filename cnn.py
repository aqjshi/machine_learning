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
    # xyz_arrays: list of arrays each with shape (27, 8)
    # Normalize only columns 0,1,2 globally
    all_xyz = np.concatenate([xyz[:, :3] for xyz in xyz_arrays], axis=0) 
    min_val = all_xyz.min()
    max_val = all_xyz.max()

    norm_xyz = []
    for xyz in xyz_arrays:
        xyz_copy = xyz.copy()
        xyz_copy[:, :3] = (xyz_copy[:, :3] - min_val) / (max_val - min_val)
        norm_xyz.append(xyz_copy)
    return min_val, max_val, norm_xyz

def apply_normalization(xyz_arrays, min_val, max_val):
    norm_xyz = []
    for xyz in xyz_arrays:
        xyz_copy = xyz.copy()
        xyz_copy[:, :3] = (xyz_copy[:, :3] - min_val) / (max_val - min_val)
        norm_xyz.append(xyz_copy)
    return norm_xyz


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

# Define device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model
class CNN(nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        
        # Assuming input shape: N x 1 x 27 x 8
        # Conv layer 1 with padding to keep spatial size
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # After this: from (1,27,8) to (64, 13,4)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        # After first conv+pool: Height: floor(27/2)=13, Width: floor(8/2)=4
        # Second conv keeps size the same (padding=2, kernel=5) -> (128,13,4)
        flatten_size = 128 * 13 * 4

        self.fc_layer = nn.Sequential(
            nn.Linear(flatten_size, output_size),
        )

    def forward(self, x):
        x = self.cnn_layers(x)   # N,128,13,4
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def weight_decay(cnn_model, l2_lambda, device):
    _reg = 0.0
    for param in cnn_model.parameters():
        _reg += torch.norm(param, 2)**2
    _reg = (l2_lambda / 2) * _reg
    return _reg

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

# Parse task from command line
task = int(sys.argv[1])


def train_model(train_data, val_data, test_data, task, num_epochs=50, batch_size=8, learning_rate=0.0001, l2_lambda=1e-4):
    # Unpack data
    train_idx, train_xyz, train_chiral, train_rot = train_data
    val_idx, val_xyz, val_chiral, val_rot = val_data
    test_idx, test_xyz, test_chiral, test_rot = test_data

    # Normalize data
    min_val, max_val, train_xyz = normalize_xyz_train(train_xyz)
    val_xyz = apply_normalization(val_xyz, min_val, max_val)
    test_xyz = apply_normalization(test_xyz, min_val, max_val)

    train_labels_np = np.array(generate_label_array(train_idx, train_xyz, train_chiral, train_rot, task))
    val_labels_np = np.array(generate_label_array(val_idx, val_xyz, val_chiral, val_rot, task))
    test_labels_np = np.array(generate_label_array(test_idx, test_xyz, test_chiral, test_rot, task))

    # Convert to tensors
    train_tensor = torch.tensor(np.array(train_xyz), dtype=torch.float32)
    val_tensor = torch.tensor(np.array(val_xyz), dtype=torch.float32)
    test_tensor = torch.tensor(np.array(test_xyz), dtype=torch.float32)

    
    train_labels = torch.tensor(train_labels_np, dtype=torch.float32 if task != 2 else torch.long)
    val_labels = torch.tensor(val_labels_np, dtype=torch.float32 if task != 2 else torch.long)
    test_labels = torch.tensor(test_labels_np, dtype=torch.float32 if task != 2 else torch.long)

    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset = TensorDataset(val_tensor, val_labels)
    test_dataset = TensorDataset(test_tensor, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine output size
    output_size = 5 if task == 2 else 1
    model = CNN(output_size).to(device)

    if task == 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Reshape for CNN
            inputs = inputs.unsqueeze(1) # (N,1,27,8)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task == 2:
                loss = criterion(outputs, labels)
            else:
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs.squeeze(), labels)
            
            # Add weight decay
            loss = loss + weight_decay(model, l2_lambda, device)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, task, test=False)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

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

print("\nTASK:", task)
print(device)

filtered_index, filtered_xyz, filtered_chiral, filtered_rotation = filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)
train_data, val_data, test_data = split_data(filtered_index, filtered_xyz, filtered_chiral, filtered_rotation)


model = train_model(train_data, val_data, test_data, task=task, num_epochs=100)