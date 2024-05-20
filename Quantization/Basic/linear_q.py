import torch

def linear_q_with_scale_and_zero_point(
    tensor, scale, zero_point, dtype = torch.int8):
    scaled_shift_tensor = tensor/scale +zero_point

    rounded_tensor = torch.round(scaled_shift_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max


    #we want the tensor to be inbetween this

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)


    return q_tensor


    ### a dummy tensor to test the implementation
test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

### these are random values for "scale" and "zero_point"
### to test the implementation
# scale = 3.5
# zero_point = -70
# quantized_tensor = linear_q_with_scale_and_zero_point(
#     test_tensor, scale, zero_point)

# print(quantized_tensor)





#lets get the dequanitzed tensor to see how accuare is it with the original values

# dequnatized_tensor = scale*(quantized_tensor.float() - zero_point)

# print("dequnatized",dequnatized_tensor)

# lets define a function dequantizer

def linear_dequantization(quanitized_tensor, scale, zero_point):
    return  scale*(quanitized_tensor.float() - zero_point)

#overall quantization error 
# er = (dequnatized_tensor - test_tensor).square().mean()

# print("er",er)



# to get the correct values of  Scale and zero point

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



scale , zero_point = get_q_scale_and_zero_point(test_tensor, dtype=torch.int8)

print("scale", scale)
print("zero point", zero_point)


#now lets get the correct quantized values 


correct_q_tensor = linear_q_with_scale_and_zero_point(test_tensor,scale,zero_point)

print("correct q_tensor", correct_q_tensor)


# new deqauntized tensor 

new_de_tensor = linear_dequantization(correct_q_tensor,scale, zero_point)

print('new d quant',new_de_tensor)


er = (new_de_tensor - test_tensor).square().mean()

print("diff",er)
