#this code is for asymmetric quantization from float32 to Int8

import torch

def linear_quantization(tensor, scale, zero_point, dtype = torch.int8):
    scaled_shift_tensor = tensor/scale +zero_point

    rounded_tensor = torch.round(scaled_shift_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max


    #we want the tensor to be inbetween this

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)


    return q_tensor


def linear_dequantization(quanitized_tensor, scale, zero_point):
    return  scale*(quanitized_tensor.float() - zero_point)


#getting correct values of score and zero point

def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)

    zero_point = q_min - (r_min / scale)

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        # round and cast to int
        zero_point = int(round(zero_point))
    
    return scale, zero_point

