import torch

def augment_dataset(real_data, synthetic_data, real_labels, synth_labels):
    augmented_data = torch.cat([real_data, synthetic_data], dim=0)
    augmented_labels = torch.cat([real_labels, synth_labels], dim=0)
    return augmented_data, augmented_labels
