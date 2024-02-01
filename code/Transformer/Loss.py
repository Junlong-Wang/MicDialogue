import torch
import torch.nn
import torch.nn.functional
from typing import Union

def sequence_cross_entropy_with_logits(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: Union[torch.FloatTensor, torch.BoolTensor],
) -> torch.FloatTensor:
    '''
    Copy from allennlp.util.sequence_cross_entropy_with_logits, we use this function to caculate the cross entropy with
    target mask(the parameter:"weights"),and average the token loss.
    '''
    # make sure weights are float
    weights = weights.to(logits.dtype)
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    return negative_log_likelihood.sum() / (
        weights_batch_sum.sum() + tiny_value_of_dtype(negative_log_likelihood.dtype)
    )

def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))