# RICE_SORTING-BW-COLOR-
"""
Created on 2025/2/19  14:11 

@author: YANG FENG RUI
"""
# -*- coding: utf-8 -*-
"""
camera_capture.py

功能说明：
1. capture_single_image() 函数：打开相机拍摄单张图像，以 16 位格式保存
2. capture_continuously() 函数：以指定帧率连续采集并保存为 16 位 .npy 文件
3. 附带几个辅助函数 (analyze_frame, show_16bit_with_matplotlib)，可选使用

使用方法：
- 在 PyCharm 中打开这个脚本，直接运行后可在 main() 中调用你想执行的函数。
- 或者在其他脚本中 from camera_capture import capture_single_image/capture_continuously 来调用。
"""

import os
import time
import numpy as np
import pylablib as pll
from pylablib.devices import Thorlabs
import matplotlib.pyplot as plt
import cv2


# 初始化(可查看可用设备等)
pll.list_backend_resources("serial")


def show_16bit_with_matplotlib(frame_16u):
    """
    用 matplotlib 显示 16 位图像。
    frame_16u: dtype=uint16, shape=(height, width) 或 (height, width, channels)
    """
    plt.figure()
    plt.imshow(frame_16u, cmap='gray', vmin=0, vmax=65535)
    plt.colorbar(label='Pixel value')
    plt.title("16-bit image (Matplotlib)")
    plt.show()


def analyze_frame(frame):
    """打印 frame 的一些基本统计信息"""
    print("Shape:", frame.shape)
    print("Dtype:", frame.dtype)
    print("Min:", frame.min())
    print("Max:", frame.max())
    print("Mean:", frame.mean())


def capture_single_image(name='background',exposure_time=0.04):
    """
    功能：打开相机，设置曝光，拍摄单张图像并保存为 16 位 .npy 和 PNG，然后关闭相机。

    使用：
      from camera_capture import capture_single_image
      capture_single_image()
    """
    # 1. 打开相机

    cam = Thorlabs.ThorlabsTLCamera()

    # 2. (可选) 设置 ROI
    # width, height = 1440, 1080
    # cam.set_roi(0, width, 0, height, hbin=1, vbin=1)

    # 3. 设置曝光时间（单位: 秒），例如 10 ms
    cam.set_exposure(exposure_time)

    # 4. 启动采集 (nframes=1 表示只采一帧)
    cam.setup_acquisition(nframes=1)
    cam.start_acquisition()

    try:
        # 5. 等待并读取图像
        cam.wait_for_frame(timeout=10.0)
        frame = cam.read_newest_image()
        analyze_frame(frame)

        # 6. 可视化（可注释）
        #show_16bit_with_matplotlib(frame)

        # 7. 保存为 .npy (16 位原始数据)
        np.save(name+'.npy', frame)
        print(f"[INFO] 16 位图像已保存为：{name}.npy")
        '''
        # 8. 也保存为 16 位 PNG（OpenCV 支持写入 16 位深度）
        cv2.imwrite('single_frame_16bit.png', frame)
        print("[INFO] 16 位图像已保存为：single_frame_16bit.png")
        '''

    finally:
        # 9. 关闭相机
        cam.close()
        print("[INFO] 相机已关闭。")




def capture_continuously(func, model, bg_gray=[0],
                         min_area=2500,
                         morph_kernel_size=3,
                         threshold_val=4,
                         exposure_time=0.05, frame_num=10):

    """
    连续采集图像，并调用func进行处理(如 classification)，
    然后将 func 返回的 result_img(带框/预测标签的8位图) 做实时显示。
    """

    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(exposure_time)

    cam.setup_acquisition(nframes=frame_num)
    cam.start_acquisition()

    frame_count = 0
    last_time = time.time()
    last_frame_time = last_time
    file_index = 0

    print(f"[INFO] 开始连续采集，每秒 {frame_num} 帧。")

    try:
        while True:
            now = time.time()

            # 当时间足够时（>=1/frame_num），采一帧
            if (now - last_frame_time) >= 1.0 / frame_num:
                frame = cam.read_newest_image()
                if frame is None:
                    continue

                # ---- 调用你的处理函数(如 classification) ----
                flag, result_img = func(
                    str(file_index),
                    model,
                    bg_gray,
                    frame,  # 原始16位灰度
                    min_area,
                    morph_kernel_size,
                    threshold_val
                )

                # 如果返回flag==0, 则停止采集
                if flag == 0:
                    print("[INFO] func返回0，中断循环。")
                    break

                # ---- 实时显示：此时 result_img 已是带框的8位BGR图 ----
                cv2.imshow("Live Preview", result_img)

                # 处理按键退出 (ESC = 27)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] 按下 ESC 键，退出循环")
                    break

                # 记录计数，更新时间
                file_index += 1
                frame_count += 1
                last_frame_time = now

            # 每隔 1 秒计算一次实际 FPS

            if (now - last_time) >= 1.0:
                fps = frame_count / (now - last_time)
                print(f"[INFO] 实际FPS: {fps:.2f}")
                frame_count = 0
                last_time = now


    except KeyboardInterrupt:
        print("[INFO] 捕获 Ctrl+C，准备关闭相机。")
    finally:
        cam.close()
        cv2.destroyAllWindows()  # 关闭窗口
        print("[INFO] 相机已关闭，脚本结束。")



def i(base_name,bg_gray, sample, min_area,morph_kernel_size,threshold_val):
    answer_bg = input("是否需要拍摄背景图像？(y/n): ").strip().lower()
    print(answer_bg)
    return 0

import cv2
import matplotlib.pyplot as plt
import numpy as np
def detect_rice_by_subtraction(
        bg_gray,
        sample_gray,
        min_area=1000,
        morph_kernel_size=2,
        threshold_val=4

):
    """
    利用像素差分的方法检测大米。
    返回:
      final_image: 在 sample_img 上用红色描绘了有效轮廓的结果图(如果需要彩色展示)
      fg_mask:     二值掩码
      valid_contours: 有效轮廓的列表
    """
    # 1. 读取背景灰度图
    #bg_gray = np.load(bg_gray_path)  # 读取灰度背景

    # 2. 读取当前帧（已经是灰度）
    #sample_gray = np.load(sample_path, allow_pickle=True)
    # 3. 作差分
    diff = cv2.absdiff(bg_gray, sample_gray)

    # 4. 二值化
    _, fg_mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

    # 5. 形态学去噪和填洞
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6. 查找轮廓
    fg_mask = fg_mask.astype(np.uint8)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 过滤小面积轮廓，并在图上描绘
    #    如果需要在灰度图上用“红色”描绘，需要先转换成 BGR。
    final_image = cv2.cvtColor(sample_gray, cv2.COLOR_GRAY2BGR)
    tmp = final_image.astype(np.float32)

    # 归一化到 [0,1] 之间
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())

    # 再乘到 [0,255]
    tmp = tmp * 255

    # 最终转换成 uint8
    final_image = tmp.astype(np.uint8)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        valid_contours.append(cnt)
        # 在 BGR 图中用红色(0,0,255)描绘
        cv2.drawContours(final_image, [cnt], -1, (0, 0, 255), 2)

    return final_image, fg_mask, valid_contours
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体为“黑体” (SimHei) 或者其他
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

def label_contours(image, contours):
    """
    在给定“单通道灰度”图像上，为每个轮廓标上唯一的序号。
    """
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # 只需用一个灰度值(例如 255 表示白色)
        cv2.putText(
            image, str(i), (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, 255, 2, cv2.LINE_AA
        )
    return image


def get_training_set(base_name,nothing,bg_gray, sample, min_area,morph_kernel_size,threshold_val):
    """
    1. 做差分得到轮廓(灰度)
    2. 在图上标识
    3. 交互式地让用户选择是否是“正常大米”，或者“轮廓错误”而删除
    4. 保存结果
    """

    # 1. 检测大米轮廓（灰度差分）
    #    请确保 detect_rice_by_subtraction 也已改为灰度版本
    final_image, fg_mask, valid_contours = detect_rice_by_subtraction(
        bg_gray=bg_gray,
        sample_gray=sample,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )

    # 2. 标记序号（方便用户识别）
    labeled_image = label_contours(final_image, valid_contours)

    # 3. 用 Matplotlib 显示图像（灰度）
    # 显示窗口
    cv2.namedWindow('Labeling Window', cv2.WINDOW_NORMAL)
    # 设置窗口大小，例如 800x600
    cv2.resizeWindow('Labeling Window', 800, 600)
    cv2.imshow('Labeling Window', labeled_image)

    normal_flags = []
    filtered_contours = []

    for i, cnt in enumerate(valid_contours):
        # 保证窗口事件被处理，否则窗口会卡住或没反应
        cv2.waitKey(1)

        user_input = input(f"编号 {i} 的大米是否是正常大米？(y/n/e): ").lower().strip()

        if user_input in ["y", "yes"]:
            normal_flags.append(True)
            filtered_contours.append(cnt)
        elif user_input in ["n", "no"]:
            normal_flags.append(False)
            filtered_contours.append(cnt)
        elif user_input in ["e", "error"]:
            print(f"编号 {i} 的轮廓已删除。")
        else:
            print(f"输入 {user_input} 无效，轮廓已删除。")

    cv2.destroyAllWindows()

    # 6. 存储轮廓和标签
    #    只存被保留的轮廓 (filtered_contours)
    conts_as_arrays = [cnt.reshape(-1, 2) for cnt in filtered_contours]
    data_dict = {
        "is_normal": normal_flags,
        "contours": conts_as_arrays,
        "data": sample  # 可以附带保存原始图像
    }

    # 假设 sample_path = "frame_save\\250.npy"
    # 取文件名并加后缀
    save_name = base_name + "_setted"
    save_folder = "train_set"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name)

    np.save(save_path, data_dict, allow_pickle=True)
    print(f"数据已保存到 {save_path}")
    ans = input("是否结束？(y/n): ").strip().lower()
    if ans=='y':
        return 0, 0
    else:
        return 1, 0


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib
import time

import cv2
import numpy as np


def extract_gray_features(image_gray_uint16, contour):
    """
    在 16 位或 8 位灰度图上，对给定轮廓提取 [mean, std] 两维特征。
    如果你训练时是别的特征，就相应修改此函数。
    """
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)

    # 创建掩膜
    h, w = image_gray_uint16.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, color=255, thickness=-1)

    if mask.sum() == 0:
        # 轮廓面积为0时，返回默认特征
        return [0.0, 0.0]

    mean_val, std_val = cv2.meanStdDev(image_gray_uint16, mask=mask)
    gray_mean = float(mean_val[0][0])
    gray_std = float(std_val[0][0])
    return [gray_mean, gray_std]


def build_dataset(train_set_dir):
    """
    遍历 train_set_dir 文件夹下的所有 .npy 文件，
    读取 data_dict，解析出 is_normal(标签)、contours(轮廓) 以及原图 data(灰度, uint16)，
    然后对每个轮廓提取[灰度均值, 灰度标准差]特征，并存入 X, y.
    """
    X = []
    y = []

    # 1. 遍历文件夹里的 .npy
    for filename in os.listdir(train_set_dir):
        if not filename.endswith(".npy"):
            continue  # 跳过非 npy 文件

        npy_path = os.path.join(train_set_dir, filename)
        data_dict = np.load(npy_path, allow_pickle=True).item()

        # data_dict 应该包含 "is_normal", "contours", "data"
        # data 为灰度 uint16 图像，形状 (H, W)
        image_gray_uint16 = data_dict["data"]
        contours = data_dict["contours"]
        is_normal_list = data_dict["is_normal"]  # list of True/False

        # 2. 针对每个轮廓，提取特征
        for cnt, normal_flag in zip(contours, is_normal_list):
            features = extract_gray_features(image_gray_uint16, cnt)
            X.append(features)

            # 将 True/False 转为 1/0 (或其他二分类标签)
            label = 1 if normal_flag else 0
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_model(train_set_dir):
    # ========== 1. 构建训练数据集 ==========

    X, y = build_dataset(train_set_dir)
    print("特征矩阵 X shape:", X.shape)  # 预计 (N, 2)
    print("标签 y shape:", y.shape)  # 预计 (N, )

    # 如果你怀疑数据范围，可以打印几条特征看看
    # print(X[:10], y[:10])

    # ========== 2. 划分训练/测试集 ==========
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ========== 3. 训练随机森林模型 ==========
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # ========== 4. 在测试集上评估 ==========
    y_pred = rf.predict(X_test)

    print("随机森林分类报告:")
    print(classification_report(y_test, y_pred, target_names=["异常大米", "正常大米"]))

    # ========== 5. (可选) 保存模型 ==========
    model_save_path = "rice_rf_model.pkl"
    joblib.dump(rf, model_save_path)
    print(f"模型已保存: {model_save_path}")





def classification(base_name,
                   model,  # 训练好的模型对象，而非模型路径
                   bg_gray,  # 背景(16位灰度数组)
                   sample,   # 当前帧(16位灰度数组)
                   min_area,
                   morph_kernel_size,
                   threshold_val
                   ):
    """
    对单帧图像 sample 进行前景检测 + 随机森林分类，并在图像上画出分类结果（绿框/红框）。
    最终返回 (flag, result_img)，其中：
      - flag: 0 or 1，用于告诉采集循环是否继续(1=继续, 0=中断)。
      - result_img: 带有识别/分类结果的 8位 BGR 图，用于实时显示。
    """
    # 1) 做差分，得到轮廓
    final_image, fg_mask, valid_contours = detect_rice_by_subtraction(
        bg_gray=bg_gray,
        sample_gray=sample,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )
    # final_image 是基于 sample_gray 转换后的 BGR 8位图（已画红色轮廓）

    # 2) 对n每个轮廓提取特征并做预测
    pred_results = []
    for contour in valid_contours:
        features = extract_gray_features(sample, contour)
        # 转成 (1,2)
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        pred_label = model.predict(features)[0]  # 0 / 1
        pred_results.append(pred_label)

    # 3) 根据预测结果重新画“矩形框”与文字：
    result_img = final_image.copy()
    for contour, pred_label in zip(valid_contours, pred_results):
        # 预测标签：0=异常(红), 1=正常(绿)
        color = (0, 255, 0) if pred_label == 1 else (0, 0, 255)
        label_text = "Normal" if pred_label == 1 else "Abnormal"

        # 获取矩形框
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_img, label_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4) 将结果保存到文件（可选）

    out_dir = "results_set"
    os.makedirs(out_dir, exist_ok=True)
    out_filename = base_name + "_inference.png"
    out_path = os.path.join(out_dir, out_filename)
    cv2.imwrite(out_path, result_img)
    
    # 5) 同步保存预测标签（可选，用于记录）
    pred_data = {
        "valid_contours": valid_contours,
        "pred_labels": pred_results
    }
    out_npy_path = os.path.join(out_dir, base_name + "_pred.npy")
    np.save(out_npy_path, pred_data, allow_pickle=True)
    
    # 6) 如果你需要某些条件触发停止，可以在此检查并 return (0, None)
    #    这里示例：一直返回 1, 表示循环继续

    return 1, result_img




if __name__ == '__main__':

        exposure_time = 0.05
        frame_num = int(0.5 / exposure_time)

        # -----------------------
        # Step 1:拍摄背景图像
        # -----------------------

        answer_bg = input("是否需要拍摄背景图像？(y/n): ").strip().lower()
        if answer_bg == 'y':

            print("开始拍摄背景图像...")
            capture_single_image(name='background', exposure_time=exposure_time)

            bg_gray_path = "background.npy"
        elif answer_bg == 'n':
            # 如果不需要拍摄，直接使用已有的背景图bg_gray=np.load("background.npy")像
            bg_gray_path = "background.npy"
        else:
            bg_gray_path = "background.npy"
        bg_gray = np.load("background.npy")
        print(f"已加载背景，使用的背景路径为：{bg_gray_path}")

        # -----------------------
        # Step 2: 处理参数
        # -----------------------
        answer_param = input("是否需要调试检测参数？(y/n): ").strip().lower()

        if answer_param == 'n':
            # 直接读取本地文件 detect_parameters.npy
            if not os.path.exists('detect_parameters.npy'):
                print("detect_parameters.npy 文件不存在，无法读取参数。请先进行调试。")

        else:
            # 需要调试
            print("开始参数调试流程...")
            # 先拍摄一张用于调试的图像
            capture_single_image(name='find_parameters', exposure_time=exposure_time)  # 可自行调整曝光时间
            sample_path = 'find_parameters.npy'
            sample_gray = np.load(sample_path)

            # 这里示例用 matplotlib 来展示
            while True:
                # 让用户输入参数
                try:
                    min_area = int(input("请输入 min_area (数字): "))
                    morph_kernel_size = int(input("请输入 morph_kernel_size (数字): "))
                    threshold_val = int(input("请输入 threshold_val (数字): "))
                except ValueError:
                    print("输入格式有误，请重新输入。")
                    continue

                # 调用检测函数
                final_img, fg_mask, valid_contours = detect_rice_by_subtraction(
                    bg_gray=bg_gray,
                    sample_gray=sample_gray,
                    min_area=min_area,
                    morph_kernel_size=morph_kernel_size,
                    threshold_val=threshold_val
                )

                # 转 RGB 以便 matplotlib 正常展示颜色
                show_final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

                # 展示结果
                plt.figure(figsize=(8, 6))
                plt.imshow(show_final_img)
                plt.title(f"min_area={min_area}, morph_kernel_size={morph_kernel_size}, threshold_val={threshold_val}")
                plt.show()

                # 再询问是否满意
                ans_satisfied = input("是否满意当前检测结果？(y/n): ").strip().lower()
                if ans_satisfied == 'y':
                    print("好的，将参数保存到 detect_parameters.npy。")
                    params_dict = {
                        'min_area': min_area,
                        'morph_kernel_size': morph_kernel_size,
                        'threshold_val': threshold_val
                    }
                    np.save('detect_parameters.npy', params_dict)
                    break
                else:
                    print("不满意，请重新输入参数。")

        params_dict = np.load('detect_parameters.npy', allow_pickle=True).item()
        min_area = params_dict.get('min_area', 1000)
        morph_kernel_size = params_dict.get('morph_kernel_size', 2)
        threshold_val = params_dict.get('threshold_val', 4)
        print("已从 detect_parameters.npy 中读取到如下参数：")
        print(f"min_area = {min_area}, morph_kernel_size = {morph_kernel_size}, threshold_val = {threshold_val}")

        # -----------------------
        # Step 3:设置训练集
        # -----------------------

        answer_st = input("是否需要设置训练集？(y/n): ").strip().lower()
        if answer_st == 'y':
            capture_continuously(get_training_set, 0, bg_gray=bg_gray,
                                        min_area=min_area,
                                        morph_kernel_size=morph_kernel_size,
                                        threshold_val=threshold_val,
                                        exposure_time=exposure_time, frame_num=frame_num)
            train_set_dir = 'train_set'
        else:
            train_set_dir = 'train_set'
        print(f"已加载训练集，使用的训练集路径为：{train_set_dir}")

        # -----------------------
        # Step 4:训练模型
        # -----------------------

        answer_st = input("是否需要训练模型？(y/n): ").strip().lower()
        if answer_st == 'y':
            train_model(train_set_dir)
            model_path = "rice_rf_model.pkl"
        else:
            model_path = "rice_rf_model.pkl"
        print(f'已完成模型训练，路径为：{model_path}')
        model = joblib.load(model_path)
        print(f'已加载模型')

        capture_continuously(classification, model, bg_gray=bg_gray,
                                    min_area=min_area,
                                    morph_kernel_size=morph_kernel_size,
                                    threshold_val=threshold_val,
                                    exposure_time=exposure_time, frame_num=frame_num)








