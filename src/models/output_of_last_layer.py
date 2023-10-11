def output_of_last_layer(pre_trained_model):
    """
    Gets the last layer output of a model
  
    Args:
    pre_trained_model (tf.keras Model): model to get the last layer output from
    
    Returns:
    last_output: output of the model's last layer 
    """
    last_desired_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_desired_layer.output_shape)
    last_output = last_desired_layer.output
    print('last layer output: ', last_output)
    
    return last_output