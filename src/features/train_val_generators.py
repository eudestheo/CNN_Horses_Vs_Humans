def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
    """
    Creates the training and validation data generators
  
    Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images
    
    Returns:
    train_generator, validation_generator: tuple containing the generators
    """
    
    train_datagen = ImageDataGenerator(
    rescale=1.0/255., #Rescales pixel values (originally 0-256) to 0-1
    rotation_range=0.4, #Rotates the image up to 40 degrees in either direction
    shear_range=0.2, #shears the image up to 20 degrees
    width_shift_range=0.2, #shifts the width by up to 20 %
    height_shift_range=0.2, #shifts the height by up to 20 %
    horizontal_flip=True, #flips the image along the horizontal axis
    fill_mode='nearest' #fills pixels lost during transformations with its nearest pixel
    )
    
    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=32, 
                                                      class_mode='binary',
                                                      target_size=(150, 150))
    
    validation_datagen = ImageDataGenerator(rescale=1.0/255.)
    
    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=32, 
                                                                class_mode='binary',
                                                                target_size=(150, 150))
    
    return train_generator, validation_generator