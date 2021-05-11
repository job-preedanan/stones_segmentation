import numpy as np
from skimage.measure import label
from skimage.morphology import disk, dilation
import tensorflow as tf


def learning_rate_decay(lr_decay_params, initial_lr, limit_lr):

    # (1) learning rate decay function
    def decay_funct(epoch):
        new_lr = initial_lr / (1 + lr_decay_params['decay_fn']['decay_val'] * epoch)
        if new_lr <= limit_lr:
            new_lr = limit_lr

        return new_lr

    # (2) learning rate reduce(half) by condition
    lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=10,
                                                              min_lr=5e-5)

    if lr_decay_params['method'] == 'condition':
        lr_decay_method = lr_reduce_callback
    elif lr_decay_params['method'] == 'decay_function':
        lr_function = tf.keras.callbacks.LearningRateScheduler(decay_funct, 1)
        lr_decay_method = lr_function

    return lr_decay_method


def plot_summary_graph(history, save_path):
    import matplotlib.pyplot as plt

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_loss2.png')
    plt.clf()

    # summarize history for accuracy
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice coefficient')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_acc2.png')
    plt.clf()

    # summarize history for recall
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_recall2.png')
    plt.clf()

    # summarize history for precision
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_precision2.png')
    plt.clf()


# Normalize Image
def normalize_x(image):
    return image / 127.5 - 1


# Normalize GT
def normalize_y(image):
    return image / 255


def denormalize_x(image):
    return (image + 1) * 127.5


# Denormarlize for saving results
def denormalize_y(image):
    return image * 255


def normalize(image):
    image = (image - np.ones_like(image)*np.min(image))/(np.max(image) - np.min(image) + 1)
    return (image * 255).astype(np.uint8)


def cc2weight(image, w_min: float = 1., w_max: float = 50., bias=1.0):

    # dilate contour
    image = dilation(image, selem=disk(1))

    cc = label(image, connectivity=2)

    weight = np.ones_like(cc, dtype='float32')
    cc_items = np.unique(cc)

    # bg region
    bgcc = np.bincount(cc.flatten()).argmax()

    # contours
    N = np.prod(cc.shape) - len(cc[cc == bgcc])    # contour area

    cc_items = np.delete(cc_items, np.argwhere(cc_items == bgcc))     # only stone contours
    K = len(cc_items)            # number of contours
    if K > 0:
        for i in cc_items:
            weight[cc == i] = (N / (K * np.sum(cc == i))) + bias

    return np.clip(weight, w_min, w_max)


def cc2weight_batch(batch_images, sf_flag, w_min: float = 1., w_max: float = 100., bias=1.0):

    images = np.zeros((batch_images.shape[2], batch_images.shape[1], batch_images.shape[0]))
    cc_batch = np.zeros_like(images)

    # dilate contour + find contour in batch
    total_cc = 0
    total_cc_area = 0
    for i in range(batch_images.shape[0]):
        if sf_flag[i] == 0:  # sf
            images[:, :, i] = np.zeros_like(batch_images[i, :, :])
        else:
            images[:, :, i] = dilation(batch_images[i, :, :], selem=disk(1))
        cc = label(images[:, :, i], background=0, connectivity=2)             # find cc in an image
        cc[cc != 0] += total_cc
        total_cc += len(np.unique(cc)) - 1                                    # not count bg cc (==0)
        cc_batch[:, :, i] = cc
        total_cc_area += np.prod(cc.shape) - len(cc[cc == 0])                 # total contour area (no bg)

    # inverse weighting calculation
    cc_items = np.unique(cc_batch)[1:]    # only stone contours
    weight = np.ones_like(images, dtype='float32')

    # print('bg_cnt size = 64655 , w = 1')
    K = len(cc_items)
    for i in cc_items:
        iw1 = bias + (total_cc_area / (K * np.sum(cc_batch == i)))
        iw2 = total_cc_area / (np.sum(cc_batch == i))
        weight[cc_batch == i] = iw2
        # print('cnt[' + str(int(i)) + '] size = ' + str(np.sum(cc_batch == i)) + ', iw = ' + str(iw2))

    weight = np.clip(weight, w_min, w_max)

    # append w_map to batch
    batch_ccweight = np.ones_like(batch_images)
    for i in range(batch_ccweight.shape[0]):
        batch_ccweight[i, :, :] = weight[:, :, i]

    return batch_ccweight


# data spliting function
def split_train_test(samples, split_ratio):
    split_idx = int(round(len(samples) * split_ratio))  # split index
    test = samples[:split_idx]
    train = samples[split_idx:]
    return train, test


# list moving function
def list_index_move(list, split_num):
    split_idx = int(round(len(list) * split_num))  # split index
    new_list = list[split_idx:]
    new_list = new_list + list[:split_idx]
    return new_list


if __name__ == '__main__':
    import cv2
    path = 'C:/Users/Job/Documents/DoctorProject/stones_segmentation/'

    image1 = cv2.resize(cv2.imread(path + 'contour_test.png', cv2.IMREAD_GRAYSCALE),(256, 256))
    # image2 = cv2.resize(cv2.imread(path + 'contour_test2.png', cv2.IMREAD_GRAYSCALE),(256, 256))
    # image3 = cv2.resize(cv2.imread(path + 'contour_test3.png', cv2.IMREAD_GRAYSCALE),(256, 256))
    # image4 = cv2.resize(cv2.imread(path + 'contour_test4.png', cv2.IMREAD_GRAYSCALE),(256, 256))
    # image5 = cv2.imread(path + '6145914_R.png', cv2.IMREAD_GRAYSCALE)
    # image6 = cv2.imread(path + '10166874_R.png', cv2.IMREAD_GRAYSCALE)
    # image7 = cv2.imread(path + '12059625_R.png', cv2.IMREAD_GRAYSCALE)
    # image8 = cv2.imread(path + '12100811_B.png', cv2.IMREAD_GRAYSCALE)
    # image9 = cv2.imread(path + '12100811_B.png', cv2.IMREAD_GRAYSCALE)

    images = np.zeros((1, 256, 256))
    images[0, :, :] = image1
    # images[1, :, :] = image2
    # images[2, :, :] = image3
    # images[3, :, :] = image4
    # images[4, :, :] = image5
    # images[5, :, :] = image6
    # images[6, :, :] = image7
    # images[7, :, :] = image8
    # images[8, :, :] = image9

    w_map = cc2weight_batch(normalize_y(images), [1])
