import torch
from helper import plot_quantization_errors

def linear_quantization(tensor, scale, zero_point, dtype = torch.int8):
    scaled_shift_tensor = tensor/scale +zero_point

    rounded_tensor = torch.round(scaled_shift_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max


    #we want the tensor to be inbetween this

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)


    return q_tensor



def linear_dequantization(quantized_tensor, scale, zero_point):
    """
    Linear de-quantization
    """
    dequantized_tensor = scale * (quantized_tensor.float() - zero_point)

    return dequantized_tensor



#get scale for quantization for symmetric function 

def get_q_scale_symmetric(tensor, dtype = torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    # return the scale
    return r_max/q_max


test_tensor = torch.randn((4, 4))

print(test_tensor)


scale = get_q_scale_symmetric(test_tensor)
print(scale)


#lets do it symmetric quantization 

def  linear_quantization_symmetric(tensor, dtype = torch.int8):
    scale = get_q_scale_symmetric(tensor)

    quantized_tensor = linear_quantization(tensor, scale= scale, zero_point=0, dtype=dtype)

    return quantized_tensor,scale



quantized_T , scale = linear_quantization_symmetric(test_tensor)

dequantized = linear_dequantization(quantized_T, scale= scale, zero_point=0)

print(dequantized)


plot_quantization_errors(
    test_tensor, quantized_T, dequantized)



