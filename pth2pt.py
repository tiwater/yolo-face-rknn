# 拷贝本文件至目标模型项目，并修改YourModelClass类为目标模型类

import torch

# 定义模型类
class YourModelClass(torch.nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        # 定义模型架构

    def forward(self, x):
        # 定义前向传播
        return x

# 加载模型权重
model = YourModelClass()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 切换到评估模式

# # 创建一个输入张量作为示例输入
# example_input = torch.randn(1, 3, 112, 112)  # 根据模型输入尺寸调整

# # 将模型转换为 TorchScript
# traced_script_module = torch.jit.trace(model, example_input)

# # 保存 TorchScript 模型
# traced_script_module.save("model_traced.pt")

# 将模型转换为 TorchScript
scripted_module = torch.jit.script(model)

# 保存 TorchScript 模型
scripted_module.save("model_scripted.pt")