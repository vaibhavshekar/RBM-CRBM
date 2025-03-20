import torch
import torch.nn.functional as F

def train_rbm(model, data_loader, epochs, lr, adaptive_cd=False, crbm=False, cond_loader=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        cd_steps = min(1 + epoch // 10, 5) if adaptive_cd else 1
        epoch_loss = 0
        for idx, batch in enumerate(data_loader):
            v = batch[0].view(batch[0].size(0), -1).float().to(model.W.device)
            if crbm:
                c = torch.nn.functional.one_hot(batch[1], num_classes=model.U.shape[1]).float().to(model.W.device)
                v_recon = model.gibbs_sampling(v, c, steps=cd_steps)
                loss = F.mse_loss(v_recon, v)
            else:
                v_recon = model.gibbs_sampling(v, steps=cd_steps)
                loss = F.mse_loss(v_recon, v)


            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(data_loader):.5f}")
