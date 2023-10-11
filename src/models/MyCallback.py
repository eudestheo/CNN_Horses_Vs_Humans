class MyCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to stop training when accuracy reaches a specified threshold.

    This callback monitors the training process and stops training
    when the accuracy metric reaches the specified threshold.

    Attributes:
    - model (tf.keras.models.Model): The model to which this callback is applied.
    - target_accuracy (float): The target accuracy threshold (default: 0.95).
    """

    def __init__(self, target_accuracy=0.95):
        super(MyCallback, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs={}):
        """
        Called at the end of each training epoch.

        If the accuracy reaches or exceeds the target_accuracy threshold,
        training is stopped.

        Args:
        - epoch (int): The current epoch number.
        - logs (dict): A dictionary of training metrics, including 'accuracy'.
        """
        if logs.get('accuracy') >= self.target_accuracy:
            print(f"\nReached {self.target_accuracy*100:.1f}% accuracy, so cancelling training!")
            self.model.stop_training = True