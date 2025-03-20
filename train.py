import torch
import torch.optim as optim
import torch.nn.functional as F

def train_rbm(model, train_loader, epochs=100, lr=0.01, k=1, device='cpu', is_crbm=False):
    """Unified training function for RBM and CRBM."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            if is_crbm:
                v = batch[0].view(-1, model.n_visible).to(device)
                c = F.one_hot(batch[1], num_classes=model.n_cond).float().to(device) 

                # Forward pass with CD-k for CRBM
                h_pos, v_neg = model(v, c=c, k=k)

                # Calculate losses
                free_energy_pos = model.free_energy(v.detach(), c.detach())
                free_energy_neg = model.free_energy(v_neg.detach(), c.detach())
                loss = free_energy_pos - free_energy_neg
            
            else:
                # For RBM: batch contains only data
                v = batch.view(-1, model.n_visible).to(device)

                # Forward pass with CD-k for RBM
                h_pos, v_neg = model(v=v.detach(), k=k)

                # Calculate losses
                free_energy_pos = model.free_energy(v.detach())
                free_energy_neg = model.free_energy(v_neg.detach())
                loss = free_energy_pos - free_energy_neg
            
            # Backpropagation
            optimizer.zero_grad()
            if not torch.isfinite(loss):
                print("Loss exploded: ", loss.item())
                print(f"v: {v.shape}, v_neg: {v_neg.shape}, c: {c.shape}")
                print(f"W: {model.W.shape}, U: {model.U.shape}, v_bias: {model.v_bias.shape}, h_bias: {model.h_bias.shape}")

                print(f"v.device: {v.device}, c.device: {c.device}, model device: {next(model.parameters()).device}")


            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
