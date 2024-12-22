import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

def plot_losses(history):
    """
    Plots training and validation losses over epochs.
    
    Args:
        history (dict): Dictionary containing 'train_losses' and 'val_losses'.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_losses']) + 1)
    plt.plot(epochs, history['train_losses'], label='Training Loss')
    plt.plot(epochs, history['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data and generate metrics.
    
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to use for evaluation.
    
    Returns:
        tuple: Confusion matrix and classification report.
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=[
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ])
    
    return cm, report

def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix as a heatmap.
    
    Args:
        cm (array): Confusion matrix.
        class_names (list): List of class names.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
