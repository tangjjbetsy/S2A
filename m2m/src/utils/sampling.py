import torch

################################################################################
# Sampling
################################################################################


# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    # Assuming logits shape: [batch_size, sequence_len, num_logits]
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_adjusted = logits - max_logits  # Numerical stability
    exp_logits = torch.exp(logits_adjusted / temperature)
    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
    probs = exp_logits / sum_exp_logits
    return probs


def weighted_sampling(probs):
    # Flatten the batch and sequence dimensions for sampling
    batch_size, sequence_len, num_logits = probs.shape
    probs_flat = probs.view(-1, num_logits)  # Shape: [batch_size * sequence_len, num_logits]
    samples_flat = torch.multinomial(probs_flat, num_samples=1, replacement=False)
    # Reshape samples to match the original batch and sequence structure
    samples = samples_flat.view(batch_size, sequence_len)
    return samples


def nucleus(probs, p):
    batch_size, sequence_len, num_logits = probs.shape
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cusum_sorted_probs = torch.cumsum(sorted_probs, dim=-1)
    valid_probs_mask = cusum_sorted_probs <= p

    # Ensure at least one value is always selected
    valid_probs_mask |= cusum_sorted_probs.cumsum(dim=-1) == 1
    candi_probs = sorted_probs * valid_probs_mask.float()

    # Normalize the probabilities
    candi_probs /= candi_probs.sum(dim=-1, keepdim=True) + 1e-10

    # Adding a small epsilon to all probabilities to avoid zero sums
    candi_probs += 1e-8
    candi_probs /= candi_probs.sum(dim=-1, keepdim=True)

    # Flatten for sampling
    candi_probs_flat = candi_probs.view(-1, num_logits)

    # Handle cases where the entire row might be zero due to filtering
    if not torch.all(candi_probs_flat.sum(dim=1) > 0):
        raise ValueError(
            "Some probability distributions have sum zero. Check the threshold 'p' or input probabilities."
        )

    samples_flat = torch.multinomial(candi_probs_flat, num_samples=1, replacement=False)
    samples = samples_flat.view(batch_size, sequence_len)

    samples = torch.gather(sorted_indices, 2, samples.unsqueeze(-1))
    return samples.squeeze(-1)


# -- main sampling function -- #
def sampling(logits, p=None, t=1.0):
    probs = softmax_with_temperature(logits=logits, temperature=t)
    if p is not None:
        samples = nucleus(probs, p=p)
    else:
        samples = weighted_sampling(probs)
    return samples
