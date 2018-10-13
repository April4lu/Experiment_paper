# -*- coding: utf-8 -*-
from tensorflow.python import pywrap_tensorflow   
 
reader = pywrap_tensorflow.NewCheckpointReader((r"./nn/train_model.ckpt1")) 
var_to_shape_map = reader.get_variable_to_shape_map()  
for key in var_to_shape_map:  
    print("tensor_name: ", key)  
    print(reader.get_tensor(key))