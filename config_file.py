# UNET configuration file
import segmentation_loss_function as loss


# directory path
path = {'save_dir': {'root_dir': 'Experiment_stone_augmentation/Experiments_2021',
                     'save_folder': 'Partition_dataset/sc+A(0.25sf)'}}

# hyper parameters
params = {'loss_function': loss.tversky_loss(0.7, 1.0),
          'lr_decay': {'method': 0,  # 0:lr_conditional  / 1:lr_decay
                       'conditional_fn': {'monitor': 'val_loss',
                                          'factor': 0.5},
                       'decay_fn': {'decay_val': 0.5}},
          'initial_lr': 0.001,
          'limit_lr': 0.0001,
          'batch_size': 4,
          'epoch': 100,
          'val_split': 0.2}

model_params = {'input_channel': 1,
                'output_channel': 1,
                'first_layer_filters': 32}

image_params = {'image_size': 256}


