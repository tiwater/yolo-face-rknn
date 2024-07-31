# yolo-face-rknn
Yolo based face detection model on rk npu

# 执行

1. 在开发板上安装 python 执行环境。注意 rknn-toolkit ubuntu 20.04 只支持 python 3.8 和 3.9。ubuntu 22.04 支持 python 3.10 和 3.11。
2. 在开发板上执行以下命令以安装必要的库
`sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc`
`pip install -i https://mirror.baidu.com/pypi/simple opencv_contrib_python`
3. 获得瑞芯微官方的 rknn-toolkit
`git clone https://hub.nuaa.cf/airockchip/rknn-toolkit2.git`
4. 将 rknn-toolkit 安装到开发板上
`cd rknn-toolkit2
pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cpxx-cpxx-linux_aarch64.whl`
注意其中的 cpxx 需要替换为 python 的版本号，请在 rknn-toolkit-lite2/packages 目录下查找和你的 python 版本匹配的 whl 文件。例如：
`pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cp38-cp38-linux_aarch64.whl`
5. 将本项目拷贝至 rk3588 开发板
6. 在开发板上准备测试视频，例如 test.mp4
7. 编辑 main.py, 将 `cap = cv2.VideoCapture('./test.mp4')` 中的路径指向你的测试视频路径
8. 执行 `python main.py`

# 模型转换

1. 本项目原始模型来自：https://github.com/akanametov/yolo-face
2. 如果需要重新量化生成 rknn 模型，请执行后续步骤：
3. 参考 https://hub.nuaa.cf/airockchip/rknn-toolkit2/blob/master/doc/01_Rockchip_RV1106_RV1103_Quick_Start_RKNN_SDK_V2.0.0beta0_CN.pdf，在 PC 端用 Docker 安装 RKNN-Toolkit2 镜像
4. 启动镜像后，在镜像中执行以下步骤：
5. 安装 ultralytics YOLO 库：
`pip install --index-url https://pypi.org/simple ultralytics`
6. 下载模型至合适的目录。本项目中为从 yolo-face 项目中下载的 yolov8n.pt
7. 将模型转换为 rknn-toolkit 支持的中间格式。对 pt 格式，可以转换为 torchscript 或 onnx。本项目中测试转换为 torchscript 效果更好，故采取此种方式。
7.1 获得瑞芯微优化的 YOLO 支持库：`git clone -b rk_opt_v1  https://github.com/airockchip/ultralytics_yolov8.git` （必须为 rk_opt_v1 分支，主分支为 onnx 格式）
7.2 `cd ultralytics_yolov8`
7.3 编辑 `ultralytics/cfg/default.yaml` 文件，将其中的 model 属性指向第 6 步中下载的模型
7.4 开始转换：
`export PYTHONPATH=./
python ./ultralytics/engine/exporter.py`
执行结束后会得到后缀为 `_rknnopt.torchscript` 的模型文件，例如 `yolov8n-face_rknnopt.torchscript`，将后缀 `.torchscript` 修改为 `.pt`，得到 `yolov8n-face_rknnopt.pt`
8. 利用 docker cp 将本项目源码拷贝进容器中
9. 准备量化样本。本例中为从 http://shuoyang1213.me/WIDERFACE/ 下载的人脸图片集。解压缩。
10. 编辑 gen_ds.py，将路径指向样本集根目录。执行 `python gen_ds`，获得 `dataset.txt` 文件。
11. 编辑 pt2krnn.py，将 dataset 路径指向第 10 步生成的`dataset.txt` 文件。
12. 执行 `python3 pt2rknn.py <model_name>.pt <platform>` 完成模型转换，例如：
`python3 pt2rknn.py yolov8n-face_rknnopt.pt rk3588`
之后即可获得 output.rknn 模型文件。

## ONNX
如果希望通过 ONNX 格式进行模型转换，在前面 7.1 处 checkout main 分支，并在第 11、12 步处改用 onnx2rknn.py。
注意： onnx2rknn.py 不接收参数，直接通过 `python onnx2rknn.py` 执行。参数在文件内修改。

# 参考

本项目代码基于： http://git.bwbot.org/publish/rknn3588-yolov8
模型转换参考： https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolov8.html#id4
更多模型可见： https://github.com/airockchip/rknn_model_zoo