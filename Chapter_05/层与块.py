import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov

#自定义层
class CustomLayer(nn.Module):
    def __init__(self,in_features,out_features,dropout_p=0.5):
        super().__init__()
        #手动管理参数
        self.weight = nn.Parameter(torch.randn(in_features,out_features)*0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = dropout_p

    def forward(self,x):
        out = torch.matmul(x,self.weight)+self.bias
        if self.training:
            out = F.dropout(out,self.dropout)
        return out

#自定义块
class CustomBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.layer1 = CustomLayer(dim,dim)
        self.relu=nn.ReLU()
        self.layer2 = CustomLayer(dim,dim)
    def forward(self,x):
        out=self.relu(x+self.layer2(self.relu(self.layer1(x))))
        return out
    
class UltimateModel(nn.Module):
    def __init__(self):
        super().__init__()
        #嵌套块
        self.blocks1 = CustomBlock(64)
        self.blocks2 = CustomBlock(64)
        self.out=nn.Linear(64,10)
    def forward(self,x):
        x=self.blocks1(x)
        x=self.blocks2(x)
        return self.out(x)

def Try_NPU(model):
    # 1. 把 PyTorch 模型转成 OpenVINO 格式
        # 注意：这一步通常用命令行，但这里我们用代码演示思想
        # input=[1, 64] 是关键：NPU 只接受静态形状，光给 example_input 会得到动态形状 [?,?]
        ov_model = ov.convert_model(model, example_input=torch.randn(1, 64), input=[1, 64])
    
        # 2. 初始化 OpenVINO 核心
        core = ov.Core()
    
        # 3. 查看你电脑上可用的硬件设备（装逼时刻）
        available_devices = core.get_available_devices()
        print(f"\n检测到可用硬件: {available_devices}") # 会输出 ['CPU', 'GPU.0', 'NPU'] 之类的
    
        # 4. 强行编译到 NPU 上
        if 'NPU' in available_devices:
            print("正在将计算图编译至 Intel NPU...")
            compiled_model = core.compile_model(ov_model, "NPU")
            
            # 5. NPU 推理
            infer_request = compiled_model.create_infer_request()
            input_tensor = torch.randn(1, 64).numpy()
            result = infer_request.infer({0: input_tensor})
            print("NPU 推理成功！输出形状:", result[0].shape)
        else:
            print("未检测到 NPU，回退到 CPU 执行。")



if __name__=="__main__":
    # 1. 实例化并指定设备 (这里用 CPU，因为后面要导给 OpenVINO 喂给 NPU)
    device = torch.device('cpu')
    model = UltimateModel().to(device)

    # 2. 打印参数树（验证第五章知识点）
    print("=== 参数管理树 ===")
    for name, param in model.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")

    # 3. 假数据训练一轮
    print("=== 假数据训练一轮 ===")
    X_dummy = torch.randn(32, 64).to(device)
    y_dummy = torch.randint(0, 10, (32,)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    loss = criterion(model(X_dummy), y_dummy)
    loss.backward()
    optimizer.step()
    print(f"\n训练完成，Loss: {loss.item():.4f}")

    # 4. 保存模型参数（核心考点：只保存 state_dict）
    torch.save(model.state_dict(), "ultimate_model.pth")
    print("模型参数已保存为 ultimate_model.pth")

    # 5. 重新加载（演示如何加载）
    new_model = UltimateModel()
    new_model.load_state_dict(torch.load("ultimate_model.pth"))
    new_model.eval() # 切到推理模式(Dropout 关闭)

    # 6. 尝试在 NPU 上运行
    print("\n=== 尝试在 NPU 上运行 ===")
    Try_NPU(new_model)
