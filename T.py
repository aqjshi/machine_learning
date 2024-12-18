import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
import sys
import pandas as pd

# implementation "https://arxiv.org/pdf/2112.01898" linear algebra transformer
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
        return [1 if 'R' == chiral_centers[0][1] else 0 for chiral_centers in chiral_centers_array]
    
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

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_size, output_size):
        super(TransformerClassifier, self).__init__()
        # input_size should match the number of features per token, which is 4 in this case
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_size))  # Assume 32 max tokens here, update based on your data
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        if x.size(-1) != self.embedding.in_features:
            # print("unmatched")
            x = x.transpose(1, 2)

        
        # Ensure positional encoding matches the input size dynamically
        seq_len = x.size(1)  # Get the sequence length (number of tokens/features)
        pos_encoding = self.positional_encoding[:, :seq_len, :]  # Adjust positional encoding to the input size
        x = self.embedding(x) + pos_encoding
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Pooling over sequence dimension
        x = self.fc_out(x)
        return x



task = 3  # Change this variable to switch between tasks

# Load and process data
index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor('qm9_filtered.npy')

index_array, xyz_arrays, chiral_centers_array, rotation_array = filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)

# print(xyz_arrays[0])
xyz_arrays = np.array([np.array(x) for x in xyz_arrays], dtype=np.float32)

print(f"Number of samples {xyz_arrays.shape[0]} Size of each sample {xyz_arrays.shape[1]}  number of features {xyz_arrays.shape[2]}")

total_elements = xyz_arrays.shape[0] * xyz_arrays.shape[1] * xyz_arrays.shape[2] # 2481084 elements
samples = xyz_arrays.shape[0] # Number of samples
sequence_length = xyz_arrays.shape[1]  # Number of tokens/features
num_features = xyz_arrays.shape[2]  # Number of features

xyz_arrays = xyz_arrays.reshape(samples, sequence_length, num_features)


# Generate appropriate classification labels (binary or multi-class)
label_array = generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)

# Convert labels to a tensor
label_array = torch.tensor(label_array, dtype=torch.float32 if task != 2 else torch.long)

# Split the original data before augmentation
train_size = int(0.9 * len(xyz_array))
test_size = len(xyz_array) - train_size
train_xyz, test_xyz, train_labels, test_labels = train_test_split(xyz_arrays, label_array, test_size=test_size, train_size=train_size)

# Augment the training dataset
train_xyz_augmented, train_labels_augmented = augment_dataset(train_xyz, train_labels,task=task)

print(f'Train size: {train_xyz_augmented.shape[0]}, Test size: {test_xyz.shape[0]}')  

# Convert to tensors
train_xyz_tensor = torch.tensor(train_xyz_augmented, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels_augmented)
test_xyz_tensor = torch.tensor(test_xyz, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels)

# Create DataLoader for training
train_dataset = TensorDataset(train_xyz_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Hyperparameters
input_size = train_xyz.shape[1]  # Number of features
learning_rate = 0.0001
num_epochs = 250

# Initialize the Transformer model
hidden_size = 256
num_heads = 16
num_layers = 2
output_size = 5 if task == 2 else 1  # 5-class for task 2, binary otherwise

# Correct: This sets input_size to the number of features per token


model = TransformerClassifier(
    input_size=input_size,  # Now correctly set to 4
    num_heads=num_heads,
    num_layers=num_layers,
    hidden_size=hidden_size,
    output_size=output_size
).to(device)





if task == 2:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
# Open the log file in write mode
with open("log.txt", "a") as log_file:
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)

            if task == 2:
                loss = criterion(outputs, target)
            else:
                loss = criterion(outputs.squeeze(), target)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log the loss for this epoch to both the console and the file
        log_line = f'TASK {task} Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n'
        print(log_line)
        log_file.write(log_line)
    
        # Evaluate the model every 5 epochs and log the results
        if (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_xyz_tensor.to(device)).squeeze()

                if task == 2:
                    test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
                else:
                    test_preds = torch.sigmoid(test_outputs).round().cpu().numpy()

                f1 = f1_score(test_labels, test_preds, average='macro' if task == 2 else 'binary')
                cm = confusion_matrix(test_labels, test_preds)
                
                # Log evaluation metrics to both the console and the file
                eval_log = f'TASK {task} Test F1 Score: {f1:.4f}\nConfusion Matrix:\n{cm}\n'
                print(eval_log)
                log_file.write(eval_log)
                
            model.train()

# Load and preprocess data
index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor('qm9_filtered.npy')
if task == 4 or task == 5:
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

model = train_model(train_data, val_data,  test_data, num_epochs=50, task=task)