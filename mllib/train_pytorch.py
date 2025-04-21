

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crptmidfreq.mllib.iterable_data import ParquetIterableDataset


def train_model(folder_path, model,
                target='forward_fh1',
                filterfile=None,
                epochs=10,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-3,
                batch_up=-1):
    """
    Trains a feedforward network on a 'huge' CSV dataset, streaming from disk.
    """
    dataset = ParquetIterableDataset(folder_path, target=target, filterfile=filterfile)

    # 2) Build DataLoader with our IterableDataset
    #    shuffle is not supported for IterableDataset by default,
    #    but you can implement your own shuffling strategy if needed.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,     # set >0 if you want parallel data loading
        pin_memory=True,   # might help if using GPU
    )
    if hasattr(model, 'fit') and callable(getattr(model, 'fit')):
        # case of torch-ensemble for instance
        # https://ensemble-pytorch.readthedocs.io/en/latest/quick_start.html#choose-the-ensemble
        # https://ensemble-pytorch.readthedocs.io/en/latest/parameters.html#gradientboostingregressor
        model.fit(
            loader,
            epochs=epochs,
            save_model=False,
        )
        model.eval()
        return model

    # 3) Create the model, loss, optimizer
    criterion = nn.MSELoss()  # or nn.BCEWithLogitsLoss() for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    # Optionally move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) Train loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        samples_processed = 0

        for batch_idx, (features, labels) in enumerate(loader):
            # Move to GPU if using it
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            preds = model(features).squeeze()
            loss = criterion(preds, labels.squeeze())

            # Backprop + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features.size(0)
            samples_processed += features.size(0)

            # You could print status or check memory usage every N steps:
            if batch_up > 0 and ((batch_idx+1) % batch_up == 0):
                avg_loss = epoch_loss / samples_processed
                print(f"Epoch [{epoch+1}/{epochs}], Step {batch_idx+1}, Avg Loss: {avg_loss:.4f}")

        # End of epoch
        avg_loss = epoch_loss / max(1, samples_processed)
        if batch_up > 0:
            print(f"Epoch [{epoch+1}/{epochs}] finished. Avg Loss: {avg_loss:.4f}")

    # 5) Done. Save the model if desired
    #save_path = "model.pth"
    #torch.save(model.state_dict(), save_path)
    #print(f"Model saved to {save_path}")
    model.eval()
    return model
