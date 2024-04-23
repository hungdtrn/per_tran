import torch
import torch.nn.functional as F

def sample_from_gumbel_sigmoid(probabilities: torch.Tensor, temperature: float = 1.0):
    """Sample from the Gumbel-Sigmoid distribution.

    Arg(s):
        probabilities - A tensor of shape (batch_size, 1) containing the probabilities of belonging to the positive
            class.
        temperature - Sigmoid temperature to approximate the results to the true binary distribution (temperature
            -> 0) or to smooth it out and make it uniform (temperature -> +Inf).
    Returns:
        A torch tensor of same shape as probabilities, containing the sampled probabilities for each example.
    """
    probabilities = torch.cat([probabilities, 1.0 - probabilities], dim=-1)
    g = torch.distributions.gumbel.Gumbel(0.0, 1.0).sample(probabilities.size()).to(probabilities.device)
    y = torch.log(probabilities + 1e-20) + g
    return torch.softmax(y / temperature, dim=-1)[:, :1]


def straight_through_gumbel_sigmoid(probabilities: torch.Tensor, temperature: float = 1.0, threshold: float = 0.5):
    """Straight-through estimator for binary variable using the Gumbel-Sigmoid distribution.

    Arg(s):
        probabilities - A tensor of shape (batch_size, 1) containing the probabilities of belonging to the positive
            class.
        temperature - Sigmoid temperature to approximate the results to the true binary distribution
            (temperature -> 0) or to smooth it out and make it uniform (temperature -> +Inf).
        threshold - Threshold for hard decision.
    Returns:
        Two tensors of shape (batch_size, 1) containing the estimated hard and soft probabilities, respectively.
    """
    y = sample_from_gumbel_sigmoid(probabilities, temperature=temperature)
    z = (y > threshold).float()
    z = (z - y).detach() + y
    return z, y

# def masked_softmax(inp, mask, dim):
#     # compute max value for stability
#     # inp_max = torch.max(inp, dim=dim, keepdim=True)[0]
#     # inp_exp = torch.exp(inp - inp_max)

#     # inp_exp = (inp_exp * mask).float()
#     # # print("In softmax, mask", mask)
#     # # print("In softmax, exp", inp_exp)
#     # sum_exp = torch.sum(inp_exp, dim=dim, keepdim=True)
#     # sum_exp[sum_exp < 1e-4] = 1
#     masked_inp = torch.where(mask > 0, inp, torch.tensor(float("-inf")).to(inp.device))
#     out = F.softmax(masked_inp, dim=-1)
#     out = torch.where(torch.isnan(out), torch.tensor(float(0)).to(inp.device), out)

#     # print("In softmax, sum exp", inp_exp)

#     return out



def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = True,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
            result = result * mask.float()
            
    return result