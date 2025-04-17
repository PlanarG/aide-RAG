import torch
import multiprocessing as mp
from transformers import AutoModel
from torchvision import transforms, models

def child_worker(model_name):
    # 每个子进程独立加载模型
    model = models.efficientnet_b0(pretrained=True)
    # 使用模型...

if __name__ == '__main__':
    # 1. 必须最先设置启动方法
    # mp.set_start_method('spawn')
    
    # 2. 主进程可以加载自己的模型（子进程不会继承）
    main_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")
    
    # 3. 创建子进程
    p = mp.Process(target=child_worker, args=("intfloat/multilingual-e5-large-instruct",))
    p.start()
    p.join()