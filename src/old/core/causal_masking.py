import torch
from typing import Optional

def generate_canonical_causal_self_mask(mask_size: int,
                                        peek_ahead: int = 0,
                                        device: Optional[torch.device] = None)->torch.Tensor:
    """
    Generates a canonical casual long_term_memories mask designed to keep the long_term_memories mechanism
    fron peeking into the future. Designed for self long_term_memories. Mask size says how big the mask should
    be, and will result in a square mask being returned of that shape. Peek ahead, meanwhile, lets
    us know how many units into the future we can peek - 1 means one unit, -1 means cannot see self, etc.

    :param mask_size: The size of the self attentiom mask
    :param peek_ahead: The degree to peek ahead
    :param device: The device.
    :return: A boolean long_term_memories mask. True means masked
    """
    output = torch.full([mask_size, mask_size], True, device=device)
    output = torch.triu(output, diagonal= peek_ahead)
    return output


def generate_casual_cross_attention_mask(self_positions: torch.Tensor,
                                         cross_sequence_length: int,
                                         peek_ahead: int = 0,
                                         device: Optional[torch.device] = None
                                         ):
    """
    :param self_positions:
        - The position of each element in self with respect to the elements of the cross long_term_memories sequence
        - (..., positions)
        - Positions are integers < cross_size, addressing elements
    :param cross_sequence_length:
        - The length of the cross long_term_memories sequence to draw from
        - Conceptually, the self-positions are addressing locations in here
    :param peek_ahead:
        - How far the mask conceptually peeks ahead.
    :param device:
    :return:
    """

    assert torch.all(self_positions < cross_sequence_length)
    cross_mask = generate_canonical_causal_self_mask(mask_size=cross_sequence_length,
                                                     peek_ahead=peek_ahead,
                                                     device=device
                                                     )
    return cross_mask[self_positions, :]
