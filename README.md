# yolo-face-rknn

Yolo-based face detection model on RK NPU

## 执行

1. 在开发板上安装 Python 执行环境。注意：
   - RKNN-Toolkit Ubuntu 20.04 只支持 Python 3.8 和 3.9。
   - Ubuntu 22.04 支持 Python 3.10 和 3.11。

2. 在开发板上执行以下命令以安装必要的库：
   ```bash
   sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc
   pip install -i https://mirror.baidu.com/pypi/simple opencv_contrib_python
   ```

3. 获得瑞芯微官方的 RKNN-Toolkit：
   ```bash
   git clone https://hub.nuaa.cf/airockchip/rknn-toolkit2.git
   ```

4. 将 RKNN-Toolkit 安装到开发板上：
   ```bash
   cd rknn-toolkit2
   pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cpxx-cpxx-linux_aarch64.whl
   ```
   其中的 `cpxx` 需要替换为 Python 的版本号，请在 `rknn-toolkit-lite2/packages` 目录下查找和你的 Python 版本匹配的 `.whl` 文件。例如：
   ```bash
   pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cp38-cp38-linux_aarch64.whl
   ```

5. 将本项目拷贝至 RK3588 开发板。

6. 在开发板上准备测试视频，例如 `test.mp4`。

7. 编辑 `main.py`，将 `cap = cv2.VideoCapture('./test.mp4')` 中的路径指向你的测试视频路径。

8. 执行：
   ```bash
   python main.py
   ```

## 模型转换

1. 本项目原始模型来自 [Yolo-Face](https://github.com/akanametov/yolo-face)。

2. 如果需要重新量化生成 RKNN 模型，请执行以下步骤：

3. 参考 [RKNN-Toolkit2 快速入门指南](https://hub.nuaa.cf/airockchip/rknn-toolkit2/blob/master/doc/01_Rockchip_RV1106_RV1103_Quick_Start_RKNN_SDK_V2.0.0beta0_CN.pdf)，在 PC 端用 Docker 安装 RKNN-Toolkit2 镜像。

4. 启动镜像后，在镜像中执行以下步骤：

5. 安装 ultralytics YOLO 库：
   ```bash
   pip install --index-url https://pypi.org/simple ultralytics==8.2.69
   ```

6. 下载模型至合适的目录。本项目中使用从 [yolo-face 项目](https://github.com/akanametov/yolo-face) 中下载的 `yolov8n.pt`。

7. 将模型转换为 RKNN-Toolkit 支持的中间格式。对于 `pt` 格式，可以转换为 `torchscript` 或 `onnx`。本项目中测试转换为 `torchscript` 效果更好，故采取此方式。
   1. 获得瑞芯微优化的 YOLO 支持库：
      ```bash
      git clone -b rk_opt_v1 https://github.com/airockchip/ultralytics_yolov8.git
      ```
      注意：必须为 `rk_opt_v1` 分支，主分支为 `onnx` 格式。
   2. 进入库目录：
      ```bash
      cd ultralytics_yolov8
      ```
   3. 编辑 `ultralytics/cfg/default.yaml` 文件，将其中的 `model` 属性指向第 6 步中下载的模型。

   4. 开始转换：
      ```bash
      export PYTHONPATH=./
      python ./ultralytics/engine/exporter.py
      ```
      执行结束后会得到后缀为 `_rknnopt.torchscript` 的模型文件，例如 `yolov8n-face_rknnopt.torchscript`。将后缀 `.torchscript` 修改为 `.pt`，得到 `yolov8n-face_rknnopt.pt`。

8. 利用 `docker cp` 将本项目源码拷贝进容器中。

9. 准备量化样本。本例中使用从 [WIDERFACE 数据集](http://shuoyang1213.me/WIDERFACE/) 下载的人脸图片集，并解压缩。

10. 编辑 `gen_ds.py`，将路径指向样本集根目录。执行：
    ```bash
    python gen_ds.py
    ```
    获得 `dataset.txt` 文件。

11. 编辑 `pt2krnn.py`，将 `dataset` 路径指向第 10 步生成的 `dataset.txt` 文件。

12. 执行：
    ```bash
    python3 pt2rknn.py <model_name>.pt <platform>
    ```
    例如：
    ```bash
    python3 pt2rknn.py yolov8n-face_rknnopt.pt rk3588
    ```
    之后即可获得 `output.rknn` 模型文件。

## ONNX

如果希望通过 ONNX 格式进行模型转换，请在第 7 步时切换到 `main` 分支，并在第 11、12 步时改用 `onnx2rknn.py`。注意：`onnx2rknn.py` 不接收参数，直接通过 `python onnx2rknn.py` 执行。参数请在文件内修改。

## 关于参数

mean_values 和 std_values 是模型在预处理输入图片时用来进行归一化的数据。这些值通常取决于模型在训练时对输入图像进行的归一化方式。因此，为了正确地进行量化，应该使用与模型训练时相同的 mean_values 和 std_values。

### 如何确定 mean_values 和 std_values 的取值

查看模型的预处理要求:

通常，模型的作者会在模型描述或代码库中说明输入数据需要的预处理方式。如果你是在使用开源模型，查看相关文档或代码，找到他们对输入图像进行归一化所使用的均值和标准差。

常见的 mean_values 和 std_values 设定
以下是几种常见情况：

如果图像输入是 RGB 格式，像素值范围在 [0, 255]:

归一化到 [0, 1]：mean_values=[[0, 0, 0]], std_values=[[255.0, 255.0, 255.0]]
归一化到 [−1, 1]：mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]]

如果图像输入已经归一化到 [0, 1]:

无归一化，即不做进一步的归一化：mean_values=[[0]], std_values=[[1]]

如果图像输入已经归一化到 [-1, 1]:

标准算⽰化时：mean_values=[[0]], std_values=[[1]]

## 参考

- 本项目代码基于： [rknn3588-yolov8](http://git.bwbot.org/publish/rknn3588-yolov8)
- 模型转换参考： [RK356x YOLOv8 示例](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolov8.html#id4)
- 更多模型可见： [RKNN 模型库](https://github.com/airockchip/rknn_model_zoo)