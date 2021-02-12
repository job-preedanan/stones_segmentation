import numpy as np
import cv2


def metrics_compute(TP, FN, FP):
    def recall(TP, FN):
        try:
            return round((TP * 100) / (TP + FN), 2)
        except ZeroDivisionError:
            return 0

    def precision(TP, FP):
        try:
            return round((TP * 100) / (TP + FP), 2)
        except ZeroDivisionError:
            return 0

    def f_score(TP, FN, FP, f_score_beta):
        try:
            return ((1 + f_score_beta ** 2) * TP * 100) / (
                        ((1 + f_score_beta ** 2) * TP) + ((f_score_beta ** 2) * FN) + FP)
        except ZeroDivisionError:
            return 0

    return recall(TP, FN), precision(TP, FP), f_score(TP, FN, FP, f_score_beta=1), f_score(TP, FN, FP, f_score_beta=2)


# Post-processing: remove small contours + dilate
def post_processing(y_pred):

    cnt_area_th = 30
    cnt_tmp = cv2.findContours(y_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnt_tmp[0] if len(cnt_tmp) == 2 else cnt_tmp[1]
    new_y_pred = y_pred.copy()
    for c in contours:
        if cv2.contourArea(c) > cnt_area_th:
            cv2.drawContours(new_y_pred, [c], 0, (0, 255, 0), 3)

    kernel = np.ones((5, 5), np.uint8)
    new_y_pred = cv2.dilate(new_y_pred, kernel, iterations=1)

    return new_y_pred


def pixelbased_metric(y_true, y_pred, binary_th=0.5):
    _, y_true = cv2.threshold(y_true, binary_th*255, 255, cv2.THRESH_BINARY)
    _, y_pred = cv2.threshold(y_pred, binary_th*255, 255, cv2.THRESH_BINARY)
    #y_pred = post_processing(y_pred)

    y_true = np.array(y_true)
    y_true = y_true.flatten()
    y_pred = np.array(y_pred)
    y_pred = y_pred.flatten()

    TP = np.sum(np.logical_and(y_true == 255, y_pred == 255))
    FP = np.sum(np.logical_and(y_true == 0, y_pred == 255))
    FN = np.sum(np.logical_and(y_true == 255, y_pred == 0))
    TN = np.sum(np.logical_and(y_true == 0, y_pred == 0))

    return TP, FP, FN


def regionbased_metric(y_true, y_pred, binary_th=0.5, overlap_th=0.5):

    _, y_true = cv2.threshold(y_true, binary_th*255, 255, cv2.THRESH_BINARY)
    _, y_pred = cv2.threshold(y_pred, binary_th*255, 255, cv2.THRESH_BINARY)

    # post processing
    #y_pred = post_processing(y_pred)

    cnt_tmp1 = cv2.findContours(y_true, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_tmp2 = cv2.findContours(y_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    y_true_contours = cnt_tmp1[0] if len(cnt_tmp1) == 2 else cnt_tmp1[1]
    y_pred_contours = cnt_tmp2[0] if len(cnt_tmp2) == 2 else cnt_tmp2[1]

    img = np.zeros([y_true.shape[0], y_true.shape[1], 3])
    img[:, :, 0] = y_true.copy()
    img[:, :, 1] = y_true.copy()
    img[:, :, 2] = y_true.copy()
    # --------------------------------- check GT contours ----------------------------------------------------------
    TP = 0
    FN = 0
    total_stones = len(y_true_contours)
    stone_data = [[0 for x in range(6)] for y in range(total_stones)]      # stone location(x,y), (w,h), size, detect?
    for i, true_cnt in enumerate(y_true_contours):
        tmp_true_cnt = np.zeros(y_true.shape, np.uint8)
        cv2.drawContours(tmp_true_cnt, [true_cnt], -1, 255, -1)

        # stone properties
        x, y, w, h = cv2.boundingRect(true_cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # cv2.imshow('stone_gt', img)
        # cv2.waitKey(0)

        stone_data[i][0] = x + round(w/2)
        stone_data[i][1] = y + round(h/2)
        stone_data[i][2] = w
        stone_data[i][3] = h
        stone_data[i][4] = cv2.contourArea(true_cnt)

        # intersect [pixel-based TP]
        pixel_TP = np.sum(np.logical_and(tmp_true_cnt == 255, y_pred == 255))

        overlap_ratio = pixel_TP / cv2.contourArea(true_cnt)

        if overlap_ratio >= overlap_th:
            TP = TP + 1
            stone_data[i][5] = True
        elif overlap_ratio < overlap_th:
            FN = FN + 1
            stone_data[i][5] = False

    # ---------------------------- check predicted contours --------------------------------------------------------
    FP = 0
    for pred_cnt in y_pred_contours:
        tmp_pred_cnt = np.zeros(y_true.shape, np.uint8)
        cv2.drawContours(tmp_pred_cnt, [pred_cnt], -1, 255, -1)

        # intersect
        pixel_TP = np.sum(np.logical_and(tmp_pred_cnt == 255, y_true == 255))

        overlap_ratio = pixel_TP / cv2.contourArea(pred_cnt)

        if overlap_ratio < overlap_th:
            FP = FP + 1

    return TP, FP, FN, total_stones, stone_data


def segmentation_evaluate_full(y_pred, y_true):
    p_TP, p_FP, p_FN = pixelbased_metric(y_true, y_pred)
    TP, FP, FN, total_stones, stone_results = regionbased_metric(y_true, y_pred)

    # evaluation by each image (pixel based)
    evaluate_pixelbased = [0 for x in range(4)]
    evaluate_pixelbased[0] = total_stones
    evaluate_pixelbased[1] = p_TP
    evaluate_pixelbased[2] = p_FN
    evaluate_pixelbased[3] = p_FP

    # evaluation by each image
    evaluate_regionbased = [0 for x in range(4)]
    evaluate_regionbased[0] = total_stones
    evaluate_regionbased[1] = TP
    evaluate_regionbased[2] = FN
    evaluate_regionbased[3] = FP

    # evaluation by each stone
    stone_data = [[0 for x in range(7)] for y in range(total_stones)]
    for i in range(total_stones):
        stone_data[i][0] = 'stone_' + str(i)
        stone_data[i][1:7] = stone_results[i]

    return evaluate_pixelbased, evaluate_regionbased, stone_data


def segmentation_evaluate(y_pred, y_true, full_KUB_map):

    k_map = full_KUB_map[:, :, 2] / 255
    u_map = full_KUB_map[:, :, 1] / 255
    b_map = full_KUB_map[:, :, 0] / 255

    # kidneys region
    k_true = y_true.copy() / 255
    k_true = np.array(255 * k_true * k_map, dtype=np.uint8)
    k_pred = y_pred.copy() / 255
    k_pred = np.array(255 * k_pred * k_map, dtype=np.uint8)
    k_TP, k_FP, k_FN, k_total_stones, k_stone_data = regionbased_metric(k_true, k_pred)

    # ureter region
    u_true = y_true.copy() / 255
    u_true = np.array(255 * u_true * u_map, dtype=np.uint8)
    u_pred = y_pred.copy() / 255
    u_pred = np.array(255 * u_pred * u_map, dtype=np.uint8)
    u_TP, u_FP, u_FN, u_total_stones, u_stone_data = regionbased_metric(u_true, u_pred)

    # bladder region
    b_true = y_true.copy() / 255
    b_true = np.array(255 * b_true * b_map, dtype=np.uint8)
    b_pred = y_pred.copy() / 255
    b_pred = np.array(255 * b_pred * b_map, dtype=np.uint8)
    b_TP, b_FP, b_FN, b_total_stones, b_stone_data = regionbased_metric(b_true, b_pred)

    # evaluation by each image
    evaluated_data = [0 for x in range(12)]
    evaluated_data[0] = k_total_stones
    evaluated_data[1] = k_TP
    evaluated_data[2] = k_FP
    evaluated_data[3] = k_FN
    evaluated_data[4] = u_total_stones
    evaluated_data[5] = u_TP
    evaluated_data[6] = u_FP
    evaluated_data[7] = u_FN
    evaluated_data[8] = b_total_stones
    evaluated_data[9] = b_TP
    evaluated_data[10] = b_FP
    evaluated_data[11] = b_FN

    # evaluation by each stone
    stone_data = [[0 for x in range(8)] for y in range(k_total_stones + u_total_stones + b_total_stones)]
    for i in range(k_total_stones):
        stone_data[i][0] = 'stone_' + str(i)
        stone_data[i][1] = 'K'
        stone_data[i][2:8] = k_stone_data[i]

    for i in range(u_total_stones):
        stone_data[k_total_stones + i][0] = 'stone_' + str(k_total_stones + i)
        stone_data[k_total_stones + i][1] = 'U'
        stone_data[k_total_stones + i][2:8] = u_stone_data[i]

    for i in range(b_total_stones):
        stone_data[k_total_stones+u_total_stones+i][0] = 'stone_' + str(k_total_stones + u_total_stones + i)
        stone_data[k_total_stones+u_total_stones+i][1] = 'B'
        stone_data[k_total_stones+u_total_stones+i][2:8] = b_stone_data[i]

    return evaluated_data, stone_data


if __name__ == '__main__':

    evaluation_results = {'image_name': [],
                          'kidney_stones': [],
                          'k_TP': [],
                          'k_FP': [],
                          'k_FN': [],
                          'k_recall': [],
                          'k_precision': [],
                          'k_F1': [],
                          'k_F2': [],
                          'ureter_stones': [],
                          'u_TP': [],
                          'u_FP': [],
                          'u_FN': [],
                          'u_recall': [],
                          'u_precision': [],
                          'u_F1': [],
                          'u_F2': [],
                          'bladder_stones': [],
                          'b_TP': [],
                          'b_FP': [],
                          'b_FN': [],
                          'b_recall': [],
                          'b_precision': [],
                          'b_F1': [],
                          'b_F2': []}

    stone_results = {'stone_name': [],
                     'stone_type': [],
                     '(x,y)': [],
                     '(w,h)': [],
                     'stone_size': [],
                     'detect': []}

    image_name = '16181956'
    y_pred = cv2.resize(cv2.imread('16181956_Result.png', cv2.IMREAD_GRAYSCALE), (1024, 1024))
    y_true = cv2.resize(cv2.imread('16181956L.png', cv2.IMREAD_GRAYSCALE), (1024, 1024))
    KUB_map = cv2.resize(cv2.imread('16181956.png'), (1024, 1024))
    evaluated_data, stone_data = segmentation_evaluate(y_pred, y_true, KUB_map)

    # KIDNEYS STONE EVALUATION
    evaluation_results['image_name'].append(image_name)
    evaluation_results['kidney_stones'].append(evaluated_data[0])
    evaluation_results['k_TP'].append(evaluated_data[1])
    evaluation_results['k_FP'].append(evaluated_data[2])
    evaluation_results['k_FN'].append(evaluated_data[3])

    # URETER STONE EVALUATION
    evaluation_results['ureter_stones'].append(evaluated_data[4])
    evaluation_results['u_TP'].append(evaluated_data[5])
    evaluation_results['u_FP'].append(evaluated_data[6])
    evaluation_results['u_FN'].append(evaluated_data[7])

    # BLADDER STONE EVALUATION
    evaluation_results['bladder_stones'].append(evaluated_data[8])
    evaluation_results['b_TP'].append(evaluated_data[9])
    evaluation_results['b_FP'].append(evaluated_data[10])
    evaluation_results['b_FN'].append(evaluated_data[11])

    # STONE DATA
    for i in range(len(stone_data)):
        stone_results['stone_name'].append(image_name + '_' + stone_data[i][0])
        stone_results['stone_type'].append(stone_data[i][1])
        stone_results['(x,y)'].append((stone_data[i][2], stone_data[i][3]))
        stone_results['(w,h)'].append((stone_data[i][4], stone_data[i][5]))
        stone_results['stone_size'].append(stone_data[i][6])
        stone_results['detect'].append(stone_data[i][7])

    print(stone_results)