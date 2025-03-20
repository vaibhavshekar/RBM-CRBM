import torch

def generate_samples(model, num_samples, n_visible, device, conditional_vectors=None):
    generated = []
    for idx in range(num_samples):
        v = torch.rand(1, n_visible).to(device)
        if conditional_vectors is not None:
            c = conditional_vectors[idx % conditional_vectors.size(0)].unsqueeze(0).float().to(device)
            v_sample = model.gibbs_sampling(v, c, steps=50)
        else:
            v_sample = model.gibbs_sampling(v, steps=50)
        generated.append(v_sample.detach().cpu())
    return torch.cat(generated)
