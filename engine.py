import torch

def loss_to_magl(loss_tensor, name="L", normalize=True):
    """
    Convert a PyTorch per-sample loss tensor into a MagL+X distribution.

    Args:
        loss_tensor (torch.Tensor): shape (N,) or (N, ...)
        name (str): distribution name in MagL
        normalize (bool): whether to normalize into probability distribution

    Returns:
        str: MagL program snippet
    """

    # Flatten (handles multi-dim losses like per-pixel)
    values = loss_tensor.detach().cpu().flatten().numpy()

    # Avoid negatives (some losses can be slightly < 0)
    values = values.clip(min=0)

    # Normalize if requested
    if normalize:
        total = values.sum()
        if total > 0:
            values = values / total

    # Convert to list string
    values_str = ", ".join(f"{v:.6f}" for v in values)

    magl_code = f"distribution {name} = [{values_str}]"

    return magl_code

def xent_to_magl(target_loss, pred_loss, name_p="P", name_q="Q"):
    """
    Convert two loss tensors into MagL distributions + xent call.
    """

    P = loss_to_magl(target_loss, name=name_p)
    Q = loss_to_magl(pred_loss, name=name_q)

    emit = f"emit xent {name_p} {name_q}"

    return "\n".join([P, Q, emit])

def loss_to_magnitudes(loss_tensor):
    vals = loss_tensor.detach().cpu().numpy()

    lines = []
    for i, v in enumerate(vals):
        lines.append(f"magnitude l{i} = {float(v):.6f}")
        lines.append(f"emit tone l{i}")

    return "\n".join(lines)

class MagLHook:
    def __init__(self):
        self.last_program = None

    def capture(self, loss_tensor):
        self.last_program = loss_to_magl(loss_tensor)

    def get_program(self):
        return self.last_program
