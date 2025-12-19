import torch
import torch.nn.functional as F
import pandas as pd
import os

def predict_with_softmax(model, dataloader, device,idx_to_class, pred_dir, output_csv="predictions.csv"):
    """
    Run predictions on a dataloader and save:
        (- object_id)
        - predicted_class
        - probability (softmax)
        - full vector of probabilities (optional)
    """
    model.eval()

    all_object_ids = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, obj_ids in dataloader:   # <<< IMPORTANT : obj_ids
            images = images.to(device)

            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            max_probs = probas.max(dim=1).values

            # accumulate
            all_object_ids.extend(obj_ids)  # <<< FIXED HERE
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(max_probs.cpu().numpy())

    # Save results
    df_out = pd.DataFrame({
        "object_id": all_object_ids,
        "pred_class": all_preds,
        "predicted_class":  [idx_to_class[i] for i in all_preds],
        "confidence": all_probs
    })

    # --- Sauvegarde CSV
    df_out.to_csv(os.path.join(pred_dir ,output_csv), index=False)

    print(f"[OK] Predictions saved in {output_csv}")

    return df_out
