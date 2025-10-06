import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scaler import init_scaler_x, init_scaler_y, scaler_x_encode, scaler_y_decode, scaler_y_encode
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score
from eda import  outlier, plot_3d_scatter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset

def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    return pd.DataFrame(data.tolist())

def npy_preprocessor(filename):
    df = read_data(filename)
    # return df[df['chiral_centers'].apply(lambda x: len(x) == 1)].reset_index(drop=True)
    return df


def apply_rotation_molecule(matrix, rotation_angle_deg=15.0):
    matrix_copy = matrix.copy()
    
    is_padded_atom_mask = np.all(matrix_copy[:, 3:] == 0, axis=1)
    real_atom_mask = ~is_padded_atom_mask

    coords = matrix_copy[real_atom_mask, :3]
    if coords.shape[0] == 0:
        return matrix_copy 

    angle_rad = np.radians(rotation_angle_deg)
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)
    
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1 - c
    x, y, z = axis
    
    rotation_matrix = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])
    
    rotated_coords = coords @ rotation_matrix.T
    matrix_copy[real_atom_mask, :3] = rotated_coords
    
    return matrix_copy

def apply_translation_molecule(matrix, translation_vector):
    """Applies a given translation vector to the molecule's coordinates."""
    matrix_copy = matrix.copy()
    
    is_padded_atom_mask = np.all(matrix_copy[:, 3:] == 0, axis=1)
    real_atom_mask = ~is_padded_atom_mask

    matrix_copy[real_atom_mask, :3] += translation_vector
    
    return matrix_copy

def apply_nuclear_uncertainty_molecule(matrix, coord_std, strength=0.05):
    """Applies random noise to simulate nuclear uncertainty."""
    matrix_copy = matrix.copy()
    
    is_padded_atom_mask = np.all(matrix_copy[:, 3:] == 0, axis=1)
    real_atom_mask = ~is_padded_atom_mask

    coords = matrix_copy[real_atom_mask, :3]
    if coords.shape[0] == 0:
        return matrix_copy

    noise = np.random.randn(*coords.shape)
    scaled_noise = noise * coord_std * strength
    perturbed_coords = coords + scaled_noise
    
    matrix_copy[real_atom_mask, :3] = perturbed_coords
    
    return matrix_copy

def augmented_dataset(fold_train_df):
    np.random.seed(42) 
    augmented_dfs = [] 
    train_coord_std = np.array([1.661206, 1.997469, 1.440860])

    # --- 1. Rotation Augmentation ---
    rot_df = fold_train_df.copy()
    rotation_angle = np.random.uniform(0, 300)
    rot_df['xyz'] = rot_df['xyz'].apply(lambda m: apply_rotation_molecule(m, rotation_angle_deg=rotation_angle))
    augmented_dfs.append(rot_df)

    # --- 2. Translation Augmentation ---
    trans_df = fold_train_df.copy()
    translation_vector = np.random.uniform(-0.5, 0.5, size=3)
    trans_df['xyz'] = trans_df['xyz'].apply(lambda m: apply_translation_molecule(m, translation_vector=translation_vector))
    augmented_dfs.append(trans_df)

    # --- 3. Nuclear Uncertainty Augmentation ---
    uncert_df = fold_train_df.copy()
    uncertainty_strength = np.random.uniform(0, 0.001)
    uncert_df['xyz'] = uncert_df['xyz'].apply(lambda m: apply_nuclear_uncertainty_molecule(m, train_coord_std, strength=uncertainty_strength))
    augmented_dfs.append(uncert_df)
    
    # --- 4. Conditional Reflection Augmentation ---
    reflect_df = fold_train_df.copy()
    chiral_mask = reflect_df['chiral_centers'].apply(lambda x: len(x) == 1)
    
    if chiral_mask.any():
        # Apply reflection to coordinates (xyz) for chiral molecules
        reflect_df.loc[chiral_mask, 'xyz'] = reflect_df.loc[chiral_mask, 'xyz'].apply(lambda xyz: xyz * -1)
        
        # Multiply the rotation vector by -1 for the same chiral molecules
        reflect_df.loc[chiral_mask, 'rotation'] = reflect_df.loc[chiral_mask, 'rotation'].apply(lambda rot: [val * -1 for val in rot])
        
    augmented_dfs.append(reflect_df)

    return pd.concat(augmented_dfs, ignore_index=True)



class ViTMultiHead(nn.Module):
    def __init__(self, img_shape=(27, 8), in_channels=1, patch_size=(9, 4), 
                 embed_dim=128, num_layers=4, nhead=4, dropout_rate=0.1):
        super().__init__()
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_h, patch_w = patch_size
        img_h, img_w = img_shape
        
        num_patches = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = in_channels * patch_h * patch_w

        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout_rate,
            batch_first=True, 
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.head_rot_reg = nn.Linear(embed_dim, 3)


    def forward(self, x):
        N, C, H, W = x.shape
        patch_h, patch_w = self.patch_size

        # This part of your model (the shared backbone) is correct.
        x = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        x = x.contiguous().view(N, -1, C * patch_h * patch_w)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)

        # Extract the CLS token output to feed into the heads
        cls_output = x[:, 0]

        pred_rot_reg = self.head_rot_reg(cls_output)

        return  pred_rot_reg


class QMDataset(Dataset):
    def __init__(self, df):
        self.xyz = [
            torch.tensor(arr, dtype=torch.float32) 
            for arr in tqdm(df['xyz'], desc="Processing molecules")
        ]
 
        rots_np = np.vstack(df['rotation'].values).astype(float)
        self.y_rot_reg = torch.tensor(rots_np, dtype=torch.float32)

    def __len__(self):
        return len(self.xyz)
        
    def __getitem__(self, idx):
        x_unpadded = self.xyz[idx]
        padded_x = torch.zeros(27, 8, dtype=torch.float32)
        padded_x[:x_unpadded.shape[0], :] = x_unpadded
        
        return (
            padded_x, 
            self.y_rot_reg[idx]
        )
def run_validation(model, train_df, val_indices, device, loaded_scaler_x, loaded_scaler_y, fold, epoch, test=False, output_filepath="test"):
    if not test:
        fold_val_df = train_df.iloc[val_indices]
    else:
        fold_val_df = train_df
    print(f" Validation subset size: {len(fold_val_df)}")

    model.eval()

    all_preds = []
    all_trues = []

    val_ds = QMDataset(fold_val_df)
    validation_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, loaded_scaler_x, loaded_scaler_y, device=device)
    )
    # 1. Gather all predictions and true labels from the validation set
    with torch.no_grad():
        loop = tqdm(validation_loader, desc=f"Fold {fold+1} Validation", unit="batch", leave=False)
        for X_encoded, y_encoded in loop:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(X_encoded)
            
            batch_preds = torch.cat([logits], dim=1)
            
            all_preds.append(batch_preds.cpu())
            all_trues.append(y_encoded.cpu())

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_trues_tensor = torch.cat(all_trues, dim=0)

    decoded_preds_tensor = scaler_y_decode(loaded_scaler_y[0], loaded_scaler_y[1], all_preds_tensor)
    decoded_trues_tensor = scaler_y_decode(loaded_scaler_y[0], loaded_scaler_y[1], all_trues_tensor)

    decoded_preds = decoded_preds_tensor.numpy()
    decoded_trues = decoded_trues_tensor.numpy()


    rot_0_mae = mean_absolute_error(decoded_trues[:, 0], decoded_preds[:, 0])
    rot_1_mae = mean_absolute_error(decoded_trues[:, 1], decoded_preds[:, 1])
    rot_2_mae = mean_absolute_error(decoded_trues[:, 2], decoded_preds[:, 2])
   
    absolute_errors_total = np.abs(decoded_trues - decoded_preds).flatten()
    error_dist_total = {
        "mean": np.mean(absolute_errors_total),
        "median": np.median(absolute_errors_total),
        "iqr": np.percentile(absolute_errors_total, 75) - np.percentile(absolute_errors_total, 25)
    }

    results_per_threshold = []

    y_pred_binary = (decoded_preds > 0).astype(int)
    y_true_binary = (decoded_trues > 0).astype(int)
    
 

    cm_rot0 = confusion_matrix(y_true_binary[:, 0], y_pred_binary[:, 0], labels=[0, 1])
    cm_rot1 = confusion_matrix(y_true_binary[:, 1], y_pred_binary[:, 1], labels=[0, 1])
    cm_rot2 = confusion_matrix(y_true_binary[:, 2], y_pred_binary[:, 2], labels=[0, 1])

    print("cm_rot0:\n",cm_rot0)
    print("cm_rot1:\n",cm_rot1)
    print("cm_rot2:\n",cm_rot2)
    print(f"rot0: {accuracy_score(y_true_binary[:, 0].flatten(), y_pred_binary[:, 0].flatten()):.4f}")
    print(f"rot1: {accuracy_score(y_true_binary[:, 1].flatten(), y_pred_binary[:, 1].flatten()):.4f}")
    print(f"rot2: {accuracy_score(y_true_binary[:, 2].flatten(), y_pred_binary[:, 2].flatten()):.4f}")


    validation_output = {
        "fold": fold + 1,
        "epoch": epoch,
        "regression_metrics": {
  
            "rot_0_MAE": rot_0_mae,
            "rot_1_MAE": rot_1_mae,
            "rot_2_MAE": rot_2_mae
        },
        "error_distribution": {
            "total": error_dist_total
        },
        "threshold_analysis": results_per_threshold,
        "distributions": {
            "true_total": decoded_trues.flatten().tolist(), "true_rot0": decoded_trues[:, 0].tolist(),
            "true_rot1": decoded_trues[:, 1].tolist(), "true_rot2": decoded_trues[:, 2].tolist(),
            "pred_total": decoded_preds.flatten().tolist(), "pred_rot0": decoded_preds[:, 0].tolist(),
            "pred_rot1": decoded_preds[:, 1].tolist(), "pred_rot2": decoded_preds[:, 2].tolist()
        }
    }



    return validation_output


def custom_collate_fn(batch, scaler_x_continuous, scaler_y, device):
    X_samples, y_samples = zip(*batch)
    
    X_batch_unscaled = torch.stack(X_samples, dim=0)
    y_batch_unscaled = torch.stack(y_samples, dim=0)


    X_continuous = X_batch_unscaled[..., :3] 
    X_categorical = X_batch_unscaled[..., 3:] 
    # turn on to remove one hot encoding

    # X_categorical = torch.ones_like(X_categorical)
   
    original_shape = X_continuous.shape
    reshaped_X_continuous = X_continuous.reshape(-1, 3)
    scaled_X_continuous_flat = scaler_x_encode(scaler_x_continuous[0], scaler_x_continuous[1], reshaped_X_continuous)
    scaled_X_continuous = scaled_X_continuous_flat.reshape(original_shape)
    scaled_X_tensor = torch.cat([scaled_X_continuous, X_categorical], dim=-1).float()
    scaled_Y_tensor = scaler_y_encode(scaler_y[0], scaler_y[1], y_batch_unscaled).float()
    
    scaled_X_tensor = scaled_X_tensor.unsqueeze(1)  # Inserts C=1 at dim 1 (N, C, H, W)
    
    return scaled_X_tensor.to(device, non_blocking=True), scaled_Y_tensor.to(device, non_blocking=True)

def run_training_epoch(model, current_train_df, optimizer, scaler,
                         loaded_scaler_x, loaded_scaler_y, device, desc, num_epochs, warmup_epochs,batch_size =128):

    augmented_dfs =  augmented_dataset(current_train_df)
    train_df_augmented_fold = augmented_dfs

    train_ds = QMDataset(train_df_augmented_fold)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, loaded_scaler_x, loaded_scaler_y, device=device)
    )
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = warmup_epochs * len(train_loader)
    # Warmup phase
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps
    )
    # Decay phase
    decay_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=(num_training_steps - num_warmup_steps)
    )
    
    # Combine them sequentially
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[num_warmup_steps]
    )
    criterion =  torch.nn.HuberLoss(reduction='mean', delta=1)



    model.train()
    total_train_loss = 0.0
    
    loop = tqdm(train_loader, desc=desc, unit="batch")
    
    for X_encoded, y_encoded in loop:
        X_encoded, y_encoded = X_encoded.to(device), y_encoded.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(X_encoded)
            
        
            loss =  criterion(logits, y_encoded)
        # Perform the backward pass and update the optimizer step outside of autocast
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        batch_loss = loss.detach().item()
        total_train_loss += batch_loss * len(X_encoded)
        loop.set_postfix(loss=f"{batch_loss:.4f}")
        

    return total_train_loss / len(train_loader.dataset)


def main():
    output_filepath = sys.argv[1]
    filename    = 'qm9_filtered.npy'

    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df       = npy_preprocessor(filename)
    sub_df = df[df['chiral_centers'].apply(len) == 1]
    

    train_df, test_df = train_test_split(sub_df, test_size=0.2, random_state=42)
    rest_df = df[df['chiral_centers'].apply(len) != 1]
    
    train_df = pd.concat([sub_df, rest_df], ignore_index=False)
    
    print(df.head())
 
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    outlier(df, filter_z=1, output_path="scalar_all_stats.json")
    outliers_in_train = outlier(train_df, filter_z=1, output_path="scalar_train_stats.json", null=False)

    df_for_scaling = train_df.drop(index=outliers_in_train) 

    scale_ds = QMDataset(df_for_scaling)
    scale_loader = DataLoader(scale_ds, batch_size=128) 
    
    print("\nCollecting all training samples to fit scalers...")
    all_x_for_scaling = []
    all_y_for_scaling = []
    for x_batch, y_batch in scale_loader:
        all_x_for_scaling.extend([x for x in x_batch])
        all_y_for_scaling.extend([y for y in y_batch])
    print(f"Collected {len(all_x_for_scaling)} samples.")
    

    loaded_scaler_x = init_scaler_x(all_x_for_scaling) # Returns (mean, std)
    loaded_scaler_y = init_scaler_y(all_y_for_scaling) # Returns (mean, std)

 



    print("\nVerifying loaded scalers by checking their learned means:")
    print("Loaded Scaler X Mean:  ", loaded_scaler_x[0])
    print("Loaded Scaler X STD:  ", loaded_scaler_x[1])
    print("Loaded Scaler Y Mean:  ", loaded_scaler_y[0])
    print("Loaded Scaler Y STD:  ", loaded_scaler_y[1])
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)



    NUM_EPOCHS = 1
    WARMUP_EPOCHS = 0
    LEARNING_RATE = 1e-4 # A good starting point for AdamW
    WEIGHT_DECAY = 1e-2  # A common value for AdamW
    img_shape=(27, 8)
    patch_size=(1, 8)
    embed_dim=1024      
    num_layers=4      
    nhead=64         
    dropout_rate=0.2
    batch_size = 64
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))


    final_model = ViTMultiHead(
                    img_shape=img_shape,
                    patch_size=patch_size,
                    embed_dim=embed_dim,       
                    num_layers=num_layers,       
                    nhead=nhead,            
                    dropout_rate=dropout_rate
                ).to(device)

    optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, NUM_EPOCHS + 1):

        desc = f"Final Training, Epoch {epoch}/{NUM_EPOCHS}"
        avg_final_train_loss = run_training_epoch(
                model=final_model,
                current_train_df=train_df,
                optimizer=optimizer,
                scaler=scaler,
                loaded_scaler_x=loaded_scaler_x,
                loaded_scaler_y=loaded_scaler_y,
                device=device,
                desc=desc,
                num_epochs=NUM_EPOCHS,
                warmup_epochs=WARMUP_EPOCHS, 
                batch_size =batch_size
            )
        print(f"Final Training Epoch {epoch}: Average Training Loss = {avg_final_train_loss:.4f}")

    test_output = run_validation(final_model, test_df,  [], device, loaded_scaler_x, loaded_scaler_y, 0, 0, test=True,output_filepath=output_filepath)
    print("Testing complete.")
    plot_3d_scatter(test_output,output_filepath)
 
if __name__ == "__main__":
    main()


