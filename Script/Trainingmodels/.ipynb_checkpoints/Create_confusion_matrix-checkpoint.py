
import torch  # 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow

def log_confusion_matrix(model, dataloader, device, class_names, n_classes, base_dir):
    #                                                            
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # --- Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes), normalize="true")
    
    # --- Save CSV version
    cm_path = "confusion_matrix.csv"
    np.savetxt(cm_path, cm, delimiter=",", fmt="%.4f")  # ← CORRIGER
    mlflow.log_artifact(cm_path)
    
    # --- Plot confusion matrix
    fig, ax = plt.subplots(figsize=(22, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format='.2f')
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Confusion Matrix", fontsize=16)
    
    # Save PNG
    fig_path = base_dir / "Outputs" / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(fig_path)
    
    print("✅ Confusion matrix logged to MLflow")