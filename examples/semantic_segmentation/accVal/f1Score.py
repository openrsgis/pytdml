"""
f1 to F1 score is used for evaluation the performance of the model
"""

import os
import cv2 as cv


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def sumnum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum


"""Calculate precision, recall, f1-score of a single category, except 0 category"""


def cal_p_r_each(pre_img, gt_img, class_type):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    precision = 0
    for i in range(pre_img.shape[0]):
        for j in range(pre_img.shape[1]):
            if gt_img[i][j] == 0:
                break
            else:
                if pre_img[i][j] == class_type and gt_img[i][j] == class_type:
                    TP += 1
                elif not pre_img[i][j] == class_type and not gt_img[i][j] == class_type:
                    TN += 1
                elif not pre_img[i][j] == class_type and gt_img[i][j] == class_type:
                    FN += 1
                elif pre_img[i][j] == class_type and not gt_img[i][j] == class_type:
                    FP += 1
    return TP, TN, FN, FP


"""Calculate precision, recall, f1-score for all categories, except 0"""


def get_all_p_r(pre_img, gt_img):
    list_TP = []
    list_TN = []
    list_FN = []
    list_FP = []
    for i in range(1, 6):
        TP, TN, FN, FP = cal_p_r_each(pre_img, gt_img, i)
        list_TP.append(TP)
        print(TP)
        list_TN.append(TN)
        print(TN)
        list_FN.append(FN)
        list_FP.append(FP)

    total_TP = sumnum(list_TP)
    total_TN = sumnum(list_TN)
    total_FN = sumnum(list_FN)
    total_FP = sumnum(list_FP)
    total_p = 0
    total_r = 0
    micro_f = 0
    try:
        total_p = total_TP / (total_TP + total_FP)
        total_r = total_TN / (total_TN + total_FN)
        micro_f = 2 * total_p * total_r / (total_p + total_r)
    except Exception as e:
        print(e)

    return total_p, total_r, micro_f


def import_img(img_path):
    img_path_list = os.listdir(img_path)
    img_path_list.sort()
    img_list = []
    for i in range(len(img_path_list)):
        img = cv.imread(img_path + img_path_list[i], flags=cv.IMREAD_UNCHANGED)
        t = img.shape
        img_list.append(img)
    return img_list


if __name__ == "__main__":
    pre_path = "/home/pySampleCube/RSTask/DeepLabV32/result"
    gt_path = "/home/RSdata/LandUse/GID5class/test/label"
    pre_img_list = import_img(pre_path)
    gt_img_list = import_img(gt_path)
    precision = []
    recall = []
    f1Score = []
    for i in range(len(pre_img_list)):
        print(str(i + 1))
        total_p, total_r, micro_f = get_all_p_r(pre_img_list[i], gt_img_list[i])
        precision.append(total_p)
        recall.append(total_r)
        f1Score.append(micro_f)

    m_precision = averagenum(precision)
    print("precision: " + str(m_precision))

    m_recall = averagenum(recall)
    print("recall: " + str(m_recall))

    m_f1Score = averagenum(f1Score)
    print("f1Score: " + str(m_f1Score))

    print("finish!")
