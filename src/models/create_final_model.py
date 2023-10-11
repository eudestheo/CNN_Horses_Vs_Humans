def create_final_model(pre_trained_model, last_output):
    """
    Create and return a final model for image classification with transfer learning.

    This function takes a pre-trained model and the last output layer as input
    and builds a new model for fine-tuning with a custom output layer.

    Args:
    - pre_trained_model (tf.keras.models.Model): A pre-trained model with trained weights.
    - last_output (tf.Tensor): The last layer of the pre-trained model, to be used as input.

    Returns:
    tf.keras.models.Model: A final model for fine-tuning with custom classification layers.
    """

    x = tf.keras.layers.Flatten()(last_output)
    

    x = tf.keras.layers.Dense(1024, activation='relu')(x)

    x = tf.keras.layers.Dropout(0.2)(x)  

    x = tf.keras.layers.Dense (1, activation='sigmoid')(x)        
    
    model = tf.keras.models.Model(pre_trained_model.input, x)

    model.compile(optimizer=RMSprop(learning_rate=0.0001), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model