# configuration file
import segmentation_loss_function as loss


TRAIN_LOCAL = False


# directory path
if TRAIN_LOCAL:
    root_path = 'C:/Users/Job/Documents/DoctorProject/KidneyStoneProject/'

    dataset_path = {'image_dir': 'data/all_images_full',
                    'gt_dir': 'data/all_groundtruth',
                    'KUB_map_dir': 'data/Full_KUB_map'}

    stones_path = {'k_stone': 'data/cropped_stone_dataset_new/k_stone_dataset',
                   'k_stone_dist': 'data/cropped_stone_dataset_new/k_stone_dist_map',
                   'u_stone': 'data/cropped_stone_dataset_new/u_stone_dataset',
                   'u_stone_dist': 'data/cropped_stone_dataset_new/u_stone_dist_map',
                   'b_stone': 'data/cropped_stone_dataset_new/b_stone_dataset',
                   'b_stone_dist': 'data/cropped_stone_dataset_new/b_stone_dist_map'}

    save_path = {'save_dir': {'root_dir': 'Results2021',
                              'save_folder': 'partition_sc+sf'}}

# work station
else:
    root_path = '/kw_resources/kidney_stone'

    dataset_path = {'image_dir': 'datasets/all_images',
                    'gt_dir': 'datasets/all_groundtruth',
                    'KUB_map_dir': 'datasets/Full_KUB_map'}

    stones_path = {'k_stone': 'datasets/cropped_stone_dataset_new/k_stone_dataset',
                   'k_stone_dist': 'datasets/cropped_stone_dataset_new/k_stone_dist_map',
                   'u_stone': 'datasets/cropped_stone_dataset_new/u_stone_dataset',
                   'u_stone_dist': 'datasets/cropped_stone_dataset_new/u_stone_dist_map',
                   'b_stone': 'datasets/cropped_stone_dataset_new/b_stone_dataset',
                   'b_stone_dist': 'datasets/cropped_stone_dataset_new/b_stone_dist_map'}

    save_path = {'save_dir': {'root_dir': 'exports',
                              'save_folder': 'full_sc_iw_batch2'}}


# hyper parameters
training_params = {'loss_function': loss.iw_tversky_loss(0.7, 2.0),
                   'evaluate_methods': [loss.dice_coef, loss.recall, loss.precision],
                   'lr_decay': {'method': 'condition',  # 0:lr_conditional  / 1:lr_decay
                                'conditional_fn': {'monitor': 'val_loss',
                                                   'factor': 0.5},
                                'decay_fn': {'decay_val': 0.5}},
                   'initial_lr': 0.001,
                   'limit_lr': 0.00005,
                   'batch_size': 4,
                   'epoch': 150,
                   'sf_ratio': 0,
                   'iw_contour': False,
                   'iw_contour_batch': True}

# UNet parameters
model_params = dict(input_channel=1,
                    output_channel=1,
                    first_layer_filters=32)

image_params = dict(image_size=256,               # image size to network
                    full_image_size=1024,         # image size in pre-processing (data augment + partition)
                    partition=False,               # partition // full images input
                    partition_method='kub_map')   # kub_map // fix_region

# data generation methods
datagen = dict(stone_augment=False,
               num_of_stones=[1, 3],
               rotation_range=5,
               zoom_range=False,
               vertical_flip=False,
               horizontal_flip=True,
               brightness_shift=False,
               contrast_change=False)

# experiment parameters
exp_params = dict(exp_method='cross_val',   # cross_val, train/val/test split
                  test_split=0.2,
                  val_split=0.125,
                  save_excel=True,
                  save_results=True,
                  save_augment=False,
                  save_graphs=True,
                  save_train_history=True)

