from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping
from time import time

# Slow down training deeper into dataset
def schedule(epoch):
    if epoch < 5:
        # Warmup model first
        return .0000032
    elif epoch < 8:
        return .01
    elif epoch < 15:
        return .002
    elif epoch < 35:
        return .0004
    elif epoch < 40:
        return .00008
    elif epoch < 45:
        return .000016
    elif epoch < 48:
        return .0000032        
    else:
        return .0000009       


def make_callbacks(weights_file):
    # checkpoint
    filepath = weights_file
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Update info
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # learning rate schedule
    lr_scheduler = LearningRateScheduler(schedule, verbose=1)

    # 
    early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')

    # all the goodies
    return [lr_scheduler, checkpoint, tensorboard, early_stopping]