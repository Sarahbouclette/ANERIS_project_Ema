from torch_lr_finder import LRFinder
import numpy as np

def find_optimal_lr(model, train_loader, criterion, optimizer, device, end_lr=10, num_iter=100):
    print("Running LR Finder...")

    # --- Initialize the finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    # --- Run test
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)

    # --- Extract loss & lr history
    lrs = lr_finder.history["lr"]
    losses = lr_finder.history["loss"]

    # --- Smooth the loss curve (optional, to reduce noise)
    losses_smooth = np.array(losses)
    for i in range(2, len(losses)-2):
        losses_smooth[i] = np.mean(losses[i-2:i+3])

    # --- Find lr at the steepest drop in loss
    min_grad_idx = np.gradient(losses_smooth).argmin()
    new_lr = lrs[min_grad_idx]

    print(f"Suggested learning rate: {new_lr:.2e}")

    # --- Plot
    lr_finder.plot(skip_start=10, skip_end=5)
    
    # --- Reset model & optimizer to initial state
    lr_finder.reset()

    return new_lr