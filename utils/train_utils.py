import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    """
    Train the model and validate after each epoch.
    
    Args:
        model (nn.Module): The CNN model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to use for training (e.g., cuda).
    
    Returns:
        dict: Training and validation losses for visualization.
    """
    model.to(device)
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        
        # Training loop
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return {"train_losses": train_losses, "val_losses": val_losses}
