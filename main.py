import numpy as np

import tracker
from detector import Detector
import cv2

if __name__ == '__main__':

    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1530, 2720), dtype=np.uint8)
    # 初始化4个撞线polygon
    list_pts_blue = [[1743, 517], [1548, 253], [1513, 263], [1718, 532]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((1530, 2720), dtype=np.uint8)
    list_pts_yellow = [[1091, 584], [857, 790], [786, 779], [777, 814], [862, 828], [1110, 604]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 填充第三个polygon
    mask_image_temp = np.zeros((1530, 2720), dtype=np.uint8)
    list_pts_red = [[1029, 1231], [1193, 1369], [1193, 1500], [1229, 1502], [1231, 1355], [1055, 1210]]
    ndarray_pts_red = np.array(list_pts_red, np.int32)
    polygon_red_value_3 = cv2.fillPoly(mask_image_temp, [ndarray_pts_red], color=3)
    polygon_red_value_3 = polygon_red_value_3[:, :, np.newaxis]

    # 填充第四个polygon
    mask_image_temp = np.zeros((1530, 2720), dtype=np.uint8)
    list_pts_green = [[1650, 1123], [1862, 924], [1972, 948], [1975, 983], [1878, 960], [1679, 1143]]
    ndarray_pts_green = np.array(list_pts_green, np.int32)
    polygon_green_value_4 = cv2.fillPoly(mask_image_temp, [ndarray_pts_green], color=4)
    polygon_green_value_4 = polygon_green_value_4[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2,3,4），供撞线计算使用
    polygon_mask = polygon_blue_value_1 + polygon_yellow_value_2 + polygon_red_value_3 + polygon_green_value_4

    # 缩小尺寸，1920x1080->960x540
    polygon_mask = cv2.resize(polygon_mask, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 红
    red_color_plate = [0, 0, 255]
    red_image = np.array(polygon_red_value_3 * red_color_plate, np.uint8)

    # 绿
    green_color_plate = [0, 255, 0]
    green_image = np.array(polygon_green_value_4 * green_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image + red_image + green_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # 記錄id
    list_TRN = []
    list_TRW = []
    list_TRS = []
    list_TRE = []

    # 记录帧数
    count = 0

    # 各个进口的车辆数目汇总
    TRN_P = 0
    TRN_B = 0
    TRN_T = 0
    TRW_P = 0
    TRW_B = 0
    TRW_T = 0
    TRS_P = 0
    TRS_B = 0
    TRS_T = 0
    TRE_P = 0
    TRE_B = 0
    TRE_T = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture('./video/test.mp4')

    # 记录结果输出文本
    txt_name = 'E:/traffic/result01' + '.txt'
    txt_file = open(txt_name, 'w')

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break
        count += 1

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            for item_b in bboxes:
                _, _, _, _, lbl, _ = item_b
                if lbl == 'others':
                    bboxes.remove(item_b)

            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，x,y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.5))
                x1_offset = int(x1 + ((x2 - x1) * 0.5))

                # 撞线的点
                y = y1_offset
                x = x1_offset

                if polygon_mask[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_TRN:
                        list_TRN.append(track_id)
                        if label == 'others':
                            pass
                        elif label == 'passenger':
                            TRN_P += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 北进口撞线 | 北进口撞线passenger总数: {TRN_P}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 北进口撞线 | 北进口撞线passenger总数: " + str(TRN_P) + '\n')
                        elif label == 'bus':
                            TRN_B += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 北进口撞线 | 北进口撞线bus总数: {TRN_B}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 北进口撞线 | 北进口撞线bus总数: " + str(TRN_B) + '\n')
                        elif label == 'truck':
                            TRN_T += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 北进口撞线 | 北进口撞线truck总数: {TRN_T}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 北进口撞线 | 北进口撞线truck总数: " + str(TRN_T) + '\n')

                if polygon_mask[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_TRW:
                        list_TRW.append(track_id)
                        if label == 'others':
                            pass
                        elif label == 'passenger':
                            TRW_P += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 西进口撞线 | 西进口撞线passenger总数: {TRW_P}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 西进口撞线 | 西进口撞线passenger总数: " + str(TRW_P) + '\n')
                        elif label == 'bus':
                            TRW_B += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 西进口撞线 | 西进口撞线bus总数: {TRW_B}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 西进口撞线 | 西进口撞线bus总数: " + str(TRW_B) + '\n')
                        elif label == 'truck':
                            TRW_T += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 西进口撞线 | 西进口撞线truck总数: {TRW_T}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 西进口撞线 | 西进口撞线truck总数: " + str(TRW_T) + '\n')

                if polygon_mask[y, x] == 3:
                    # 如果撞 红polygon
                    if track_id not in list_TRS:
                        list_TRS.append(track_id)
                        if label == 'others':
                            pass
                        elif label == 'passenger':
                            TRS_P += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 南进口撞线 | 南进口撞线passenger总数: {TRS_P}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 南进口撞线 | 南进口撞线passenger总数: " + str(TRS_P) + '\n')
                        elif label == 'bus':
                            TRS_B += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 南进口撞线 | 南进口撞线bus总数: {TRS_B}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 南进口撞线 | 南进口撞线bus总数: " + str(TRS_B) + '\n')
                        elif label == 'truck':
                            TRS_T += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 南进口撞线 | 南进口撞线truck总数: {TRS_T}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 南进口撞线 | 南进口撞线truck总数: " + str(TRS_T) + '\n')

                if polygon_mask[y, x] == 4:
                    # 如果撞 绿polygon
                    if track_id not in list_TRE:
                        list_TRE.append(track_id)
                        if label == 'others':
                            pass
                        elif label == 'passenger':
                            TRE_P += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 东进口撞线 | 东进口撞线passenger总数: {TRE_P}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 东进口撞线 | 东进口撞线passenger总数: " + str(TRE_P) + '\n')
                        elif label == 'bus':
                            TRE_B += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 东进口撞线 | 东进口撞线bus总数: {TRE_B}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 东进口撞线 | 东进口撞线bus总数: " + str(TRE_B) + '\n')
                        elif label == 'truck':
                            TRE_T += 1
                            print(f'第 {count} 帧 | 类别: {label} | id: {track_id} | 东进口撞线 | 东进口撞线truck总数: {TRE_T}')
                            txt_file.write("第 " + str(count) + " 帧 | " + "类别：" + str(label) + " | id:" +
                                           str(track_id) + " | 东进口撞线 | 东进口撞线truck总数: " + str(TRE_T) + '\n')
                else:
                    pass
                pass
            pass

            # ----------------------清除无用id----------------------
            list_all = list_TRN + list_TRW + list_TRS + list_TRE
            for id1 in list_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_TRN:
                        list_TRN.remove(id1)
                    pass
                    if id1 in list_TRW:
                        list_TRW.remove(id1)
                    pass
                    if id1 in list_TRS:
                        list_TRS.remove(id1)
                    pass
                    if id1 in list_TRE:
                        list_TRE.remove(id1)
                    pass
                pass
            list_all.clear()
            pass

            # 清空list
            list_bboxs.clear()
            pass

        else:
            # 如果图像中没有任何的bbox，则清空虛擬框list
            list_TRN.clear()
            list_TRS.clear()
            list_TRW.clear()
            list_TRE.clear()
            pass
        pass

        TR = TRN_P + TRE_P + TRS_P + TRW_P
        text_draw = ''
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)

        pass
    pass

    capture.release()
    cv2.destroyAllWindows()
