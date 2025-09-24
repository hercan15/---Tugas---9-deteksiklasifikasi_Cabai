# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2025-07-27 11:40:26
# @Last Modified by:   Your name
# @Last Modified time: 2025-07-27 11:44:52
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")