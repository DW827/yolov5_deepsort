# 基于无人机视频的交叉口的 车辆 检测 跟踪 计数

## 环境

- Win10
- python 3.9.10。
- yolov5 v6.1版。使用的权重文件可以在此下载：链接：https://pan.baidu.com/s/17Wpjq1X2NL33KNY3TNpPlQ 提取码：o0lz

## 功能
- 实现了 交叉口四个进口道车辆类型 分别计数。
- 显示检测类别。
- 要检测不同位置和方向，可在 main.py 文件修改4个polygon的点。
- 默认检测类别：passenger，bus，truck，others。
- 检测类别可在 detector.py 文件第60行修改。

## 运行

1. 下载代码

    ```
    D:\> git clone https://github.com/DW827/yolov5_deepsort.git
    ```

2. 进入目录

    ```
    D:\> cd win10_yolov5_deepsort_counting
    ```

3. 创建 python 虚拟环境

    ```
    D:\win10_yolov5_deepsort_counting> python -m venv venv
    ```

4. 激活虚拟环境

    ```
     D:\win10_yolov5_deepsort_counting> venv\Scripts\activate
    ```

5. 升级pip

    ```
     (venv) D:\win10_yolov5_deepsort_counting> python -m pip install --upgrade pip
    ```

6. 安装软件包

    ```
     (venv) D:\win10_yolov5_deepsort_counting> pip3 install -r requirements.txt
    ```

7. 在 main.py 文件中第97行，设置要检测的视频文件路径，默认为 './video/test.mp4'

    > 400+MB的测试视频可以在这里下载：链接：https://pan.baidu.com/s/1hhnwhIOC0IMH-5WM3axxIA 提取码：dk6h

    ```
    capture = cv2.VideoCapture(r'video\test.mp4')
    ```

8. 运行程序

    ```
    (venv) D:\win10_yolov5_deepsort_counting> python main.py
    ```


## 使用框架

- https://github.com/Sharpiless/Yolov5-deepsort-inference
- https://github.com/ultralytics/yolov5/
- https://github.com/ZQPei/deep_sort_pytorch


## 代码参考
本代码改自 https://github.com/dyh/win10_yolov5_deepsort_counting.git
感谢作者开源。
