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


def train_model(model, dataloaders, criterion, scheduler, optimizer, best_model_path, class_to_idx, classes_to_remove, n_classes, device, num_epochs=30,
                artf_path="Unammed_model"):
    # --- Init logging
    log.info("Start training")
    # --- Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # --- Prepare checkpointing
    checkpoint_dir = os.path.dirname(best_model_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    torch.save(model.state_dict(), best_model_path)
    best_loss = float("inf") 

    # --- Init tracking variables
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    train_aucs, valid_aucs = [], []
    train_bal_accs, valid_bal_accs = [], []
    train_weighted_accs, valid_weighted_accs = [], []
    train_plankton_recalls, valid_plankton_recalls = [], []
    train_plankton_precisions, valid_plankton_precisions = [], []
    train_f1_score, valid_f1_score = [], []
    lrs = []

    # --- Plankton classes
    classes_to_keep = [idx for name, idx in class_to_idx.items() if name not in classes_to_remove]

    # --- Training loop
    for epoch in tqdm(range(num_epochs)):
        log.info(f"Epoch {epoch + 1}/{num_epochs}")

        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()

            run_n = 0
            run_loss = 0.0
            run_auc = 0.0
            run_acc = 0.0
            run_bal_acc = 0.0
            all_outputs, all_labels = [], []

            # --- Batch iteration
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                #forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

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
            if phase == 'train':
                scheduler.step()
                lrs.append(optimizer.param_groups[0]['lr'])

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
                labels= classes_to_keep,
                average='weighted',
                zero_division=0
            )
            epoch_plankton_precision = precision_score(
                y_pred=preds_cat,
                y_true=labels_cat,
                labels= classes_to_keep,
                average='weighted',
                zero_division=0
            )

            epoch_f1 = metf.multiclass_f1_score(input=preds_cat, target=labels_cat, num_classes=n_classes, average='macro') * batch_size
            
            log.info(f'{epoch+1}\t{phase}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_auc:.4f}\t{epoch_bal_acc:.4f}')

            # --- Save metrics for plotting (optional)
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                train_aucs.append(epoch_auc)
                train_bal_accs.append(epoch_bal_acc)
                train_plankton_recalls.append(epoch_plankton_recall)
                train_plankton_precisions.append(epoch_plankton_precision)
                train_f1_score.append(epoch_f1)
            else:
                valid_losses.append(epoch_loss)
                valid_accs.append(epoch_acc)
                valid_aucs.append(epoch_auc)
                valid_bal_accs.append(epoch_bal_acc)
                valid_plankton_recalls.append(epoch_plankton_recall)
                valid_plankton_precisions.append(epoch_plankton_precision)
                valid_f1_score.append(epoch_f1)

            # --- Log metrics to MLflow
            mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{phase}_accuracy", epoch_acc, step=epoch)
            mlflow.log_metric(f"{phase}_AUC", epoch_auc, step=epoch)
            mlflow.log_metric(f"{phase}_balanced_accuracy", epoch_bal_acc, step=epoch)
            mlflow.log_metric(f"{phase}_plankton_recall", epoch_plankton_recall, step=epoch)
            mlflow.log_metric(f"{phase}_plankton_precision", epoch_plankton_precision, step=epoch)
            mlflow.log_metric(f"{phase}_f1_score", epoch_f1, step=epoch)

            # --- Save best model
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

        torch.cuda.empty_cache()


