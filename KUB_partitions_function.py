import numpy as np
import cv2


def find_KUB_bounding_box(full_KUB_map, border_size=80):

    # KUB map preprocessing
    full_KUB_map = cv2.cvtColor(full_KUB_map, cv2.COLOR_BGR2GRAY)
    _, full_KUB_map = cv2.threshold(full_KUB_map, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    full_KUB_map = cv2.dilate(full_KUB_map, kernel, iterations=1)

    # find contour
    cnt_tmp = cv2.findContours(full_KUB_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = cnt_tmp[0] if len(cnt_tmp) == 2 else cnt_tmp[1]
    x, y, w, h = cv2.boundingRect(contour[0])    # parameters for L and R partitions

    # expand bb
    x_top = max(0, x - border_size)
    w_top = min(w + 2*border_size, full_KUB_map.shape[1] - x_top)
    y_top = max(0, y - border_size)
    h_top = round(h/2) + 2*border_size

    # find bladder partition
    low_img_map = full_KUB_map.copy()
    low_img_map[y:y+round(h/2), :] = 0    # remove top map

    # find contour
    cnt_tmp = cv2.findContours(low_img_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    low_contour = cnt_tmp[0] if len(cnt_tmp) == 2 else cnt_tmp[1]
    x_low, y_low, w_low, h_low = cv2.boundingRect(low_contour[0])  # parameters for B partition
    # expand bb
    x_low = x_low - border_size
    w_low = w_low + 2 * border_size
    h_low = min(h_low + border_size, full_KUB_map.shape[0] - y_low)

    # display testing
    # display_img = np.zeros([full_KUB_map.shape[0], full_KUB_map.shape[1], 3])
    # display_img[:, :, 0] = full_KUB_map.copy()
    # display_img[:, :, 1] = full_KUB_map.copy()
    # display_img[:, :, 2] = full_KUB_map.copy()
    #
    # # left partition
    # cv2.rectangle(display_img, (x_top, y_top), (x_top + round(w_top/2), y_top + h_top), (255, 0, 0), 3)
    # # right partition
    # cv2.rectangle(display_img, (1 + x_top + round(w_top/2), y_top), (x_top + w_top, y_top + h_top), (0, 255, 0), 3)
    # # bottom partition
    # cv2.rectangle(display_img, (x_low, y_low), (x_low + w_low, y_low + h_low), (0, 0, 255), 3)
    #
    # cv2.imshow('bb_display', display_img)
    # cv2.waitKey(0)

    return x_top, y_top, w_top, h_top, x_low, y_low, w_low, h_low


def create_KUB_partitions(full_image, full_KUB_map):

    # bb of full KUB map
    x_top, y_top, w_top, h_top, x_low, y_low, w_low, h_low = find_KUB_bounding_box(full_KUB_map)

    # left partition
    L_partition = full_image[y_top:y_top+h_top, x_top:x_top+round(w_top/2)]

    # right partition
    R_partition = full_image[y_top:y_top+h_top, x_top+round(w_top/2):x_top+w_top]
    # cv2.imshow('Right partition', R_partition)
    # cv2.waitKey(0)

    # bottom partition
    B_partition = full_image[y_low:y_low+h_low, x_low:x_low+w_low]
    # cv2.imshow('Bladder partition', B_partition)
    # cv2.waitKey(0)

    return L_partition, R_partition, B_partition


def combine_KUB_partitions(L_image, R_image, B_image, full_KUB_map):

    # bb of full KUB map
    x_top, y_top, w_top, h_top, x_low, y_low, w_low, h_low = find_KUB_bounding_box(full_KUB_map)

    # resize input partitions
    L_image = cv2.resize(L_image, (round(w_top/2), h_top))
    R_image = cv2.resize(R_image, (w_top - round(w_top/2), h_top))
    B_image = cv2.resize(B_image, (w_low, h_low))

    # create image size equal size to full image
    combined_image = np.zeros([full_KUB_map.shape[0], full_KUB_map.shape[1]], np.float32)

    combined_image[y_top:y_top+h_top, x_top:x_top+round(w_top/2)] = L_image            # left partition
    combined_image[y_top:y_top+h_top, x_top+round(w_top/2):x_top+w_top] = R_image    # right partition
    combined_image[y_low:y_low+h_low, x_low:x_low+w_low] = B_image                     # bottom partition

    # cv2.imshow('combined_image', combined_image)
    # cv2.waitKey(0)

    return combined_image


def create_KUB_partitions_old(full_image):

    w, h = full_image.shape

    # left partition
    L_partition = full_image[0:round(h/2), 0:round(w/2)]

    # right partition
    R_partition = full_image[0:round(h/2), 1+round(w/2):w]

    # bottom partition
    B_partition = full_image[1 + round(h/2):h, round(w/4):3 * round(w/4)]

    return L_partition, R_partition, B_partition


def combine_KUB_partitions_old(L_image, R_image, B_image):

    w = L_image.shape[1] + R_image.shape[1]
    h = L_image.shape[0] + B_image.shape[0]

    # create image size equal size to full image
    combined_image = np.zeros([h, w], np.float32)

    combined_image[0:round(h/2), 0:round(w/2)] = L_image  # left partition
    combined_image[0:round(h/2), round(w/2):w] = R_image  # right partition
    combined_image[round(h/2):h, round(w/4):3 * round(w/4)] = B_image  # bottom partition

    return combined_image


if __name__ == '__main__':
    import random

    IMAGE_SIZE = 1024
    full_image = cv2.resize(cv2.imread('data/all_images/T008.jpg', cv2.IMREAD_GRAYSCALE), (1024, 1024))
    full_image = cv2.resize(full_image, (round(full_image.shape[1] / (full_image.shape[0] / 512)), 512))
    full_KUB_map = cv2.resize(cv2.imread('data/Full_KUB_map/T008.png'), (full_image.shape[1], full_image.shape[0]))

    L_partition, R_partition, B_partition = create_KUB_partitions(full_image, full_KUB_map)
    L_partition = cv2.resize(L_partition, (256, 256))
    R_partition = cv2.resize(R_partition, (256, 256))
    B_partition = cv2.resize(B_partition, (256, 256))

    cv2.imshow('Left partition', L_partition)
    cv2.waitKey(0)

    cv2.imshow('Right partition', R_partition)
    cv2.waitKey(0)

    cv2.imshow('Bot partition', B_partition)
    cv2.waitKey(0)
    #full_image = combine_KUB_partitions(L_partition, R_partition, B_partition, full_KUB_map)