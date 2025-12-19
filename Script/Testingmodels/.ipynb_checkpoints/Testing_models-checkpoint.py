import logging
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import mlflow

# prepare loggers
log = logging.getLogger()
log.setLevel(logging.INFO)

    # define the output format for log messages
log_formatter = logging.Formatter('%(asctime)s.%(msecs)03d\t%(message)s',\
                                  datefmt='%Y-%m-%dT%H:%M:%S')

    # log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
log.addHandler(console_handler)

# for model evaluation
import torcheval.metrics.functional as metf
from sklearn.metrics import precision_score, recall_score

def test_model(model, dataloaders, criterion, best_model_path, class_to_idx, classes_to_remove, n_classes, device, num_epochs=1,
                artf_path="Unnamed_model", is_inception=False):
    """
    Teste un modèle sur le test set avec le meilleur checkpoint sauvegardé.
    
    Args:
        model: Le modèle PyTorch à tester
        dataloaders: Dict contenant le dataloader 'test'
        criterion: La fonction de loss
        best_model_path: Chemin vers les poids du meilleur modèle
        num_epochs: Nombre d'époques de test (généralement 1)
        artf_path: Nom du modèle pour les logs
        is_inception: Si True, gère les sorties auxiliaires d'Inception
    """
    # --- Init logging
    log.info("Start testing")
    
    # --- Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # --- Charger le meilleur modèle
    log.info(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()  # Mode évaluation
    
    # --- Init tracking variables
    test_losses = []
    test_accs = []
    test_aucs = []
    test_bal_accs = []
    test_plankton_recalls = []
    test_plankton_precisions = []
    test_f1_scores = []
    
    # --- Plankton classes
    classes_to_keep = [idx for name, idx in class_to_idx.items() if name not in classes_to_remove]
    
    # --- Testing loop
    for epoch in tqdm(range(num_epochs)):
        log.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        phase = 'test'
        
        run_n = 0
        run_loss = 0.0
        run_auc = 0.0
        run_acc = 0.0
        run_bal_acc = 0.0
        all_outputs, all_labels = [], []
        
        # --- Batch iteration (sans gradients)
        with torch.no_grad():
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculer la loss
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                # --- Stats whatever the class 
                batch_size = inputs.size(0)
                run_n += batch_size
                run_loss += loss.item() * batch_size
                run_auc += metf.multiclass_auroc(input=outputs, target=labels, num_classes=n_classes) * batch_size
                run_acc += metf.multiclass_accuracy(input=outputs, target=labels, average='micro') * batch_size
                run_bal_acc += metf.multiclass_accuracy(input=outputs, target=labels, num_classes=n_classes, average='macro') * batch_size
                
                all_outputs.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())
        
        # --- End of epoch stats
        epoch_loss = run_loss / run_n
        epoch_auc = run_auc / run_n
        epoch_acc = run_acc / run_n
        epoch_bal_acc = run_bal_acc / run_n
        
        # Calcul des prédictions discrètes
        outputs_cat = torch.cat(all_outputs)
        labels_cat = torch.cat(all_labels)
        preds_cat = torch.argmax(outputs_cat, dim=1)
        
        epoch_plankton_recall = recall_score(
            y_pred=preds_cat,
            y_true=labels_cat,
            labels=classes_to_keep,
            average='weighted',
            zero_division=0
        )
        epoch_plankton_precision = precision_score(
            y_pred=preds_cat,
            y_true=labels_cat,
            labels=classes_to_keep,
            average='weighted',
            zero_division=0
        )

        epoch_f1 = metf.multiclass_f1_score(input=preds_cat, target=labels_cat, num_classes=n_classes, average='macro') * batch_size
        
        log.info(f'{epoch+1}\t{phase}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_auc:.4f}\t{epoch_bal_acc:.4f}')
        
        # --- Save metrics
        test_losses.append(epoch_loss)
        test_accs.append(epoch_acc)
        test_aucs.append(epoch_auc)
        test_bal_accs.append(epoch_bal_acc)
        test_plankton_recalls.append(epoch_plankton_recall)
        test_plankton_precisions.append(epoch_plankton_precision)
        test_f1_scores.append(epoch_f1)
        
        # --- Log metrics to MLflow
        mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
        mlflow.log_metric(f"{phase}_accuracy", epoch_acc, step=epoch)
        mlflow.log_metric(f"{phase}_AUC", epoch_auc, step=epoch)
        mlflow.log_metric(f"{phase}_balanced_accuracy", epoch_bal_acc, step=epoch)
        mlflow.log_metric(f"{phase}_plankton_recall", epoch_plankton_recall, step=epoch)
        mlflow.log_metric(f"{phase}_plankton_precision", epoch_plankton_precision, step=epoch)
        mlflow.log_metric(f"{phase}_f1_score", epoch_f1, step=epoch)
        
        torch.cuda.empty_cache()
    
    # --- Résumé final
    log.info("="*50)
    log.info("TEST RESULTS SUMMARY")
    log.info("="*50)
    log.info(f"Loss: {test_losses[-1]:.4f}")
    log.info(f"Accuracy: {test_accs[-1]:.4f}")
    log.info(f"AUC: {test_aucs[-1]:.4f}")
    log.info(f"Balanced Accuracy: {test_bal_accs[-1]:.4f}")
    log.info(f"F1 Score: {test_f1_scores[-1]:.4f}")
    log.info(f"Plankton Recall: {test_plankton_recalls[-1]:.4f}")
    log.info(f"Plankton Precision: {test_plankton_precisions[-1]:.4f}")
    log.info("="*50)
    
    return {
        'loss': test_losses[-1],
        'accuracy': test_accs[-1],
        'auc': test_aucs[-1],
        'balanced_accuracy': test_bal_accs[-1],
        'f1_score': test_f1_scores[-1],
        'plankton_recall': test_plankton_recalls[-1],
        'plankton_precision': test_plankton_precisions[-1]
    }