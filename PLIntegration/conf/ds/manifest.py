#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
    へ　　　　　  ／|
    /＼7　　   ∠＿/
    /　│　　 ／　／               
    │　Z ＿,＜　／　　 /`ヽ     
    │　　　　　ヽ　　 /　　〉    
    Y　　　　　  `  /　　/     
    ｲ ●　､　●　⊂⊃〈　　/       
    ()  へ　　　　 |　＼       
    >ｰ ､_　 ィ　 │ ／／        | @author:yjc
    / へ　　 /　ﾉ＜| ＼＼       | @project:PLIntegration
    ヽ_ﾉ　　(_／　 │／／        | @file: manifest.py.py
    7　　　　　　　|／          | @time: 2021/11/10 上午10:43
====＞―r￣￣`ｰ―＿===========    
^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
from dataclasses import dataclass, MISSING, field
from typing import Dict, Optional, Any


@dataclass
class DSArgs:
    manifest_filepath: str = MISSING
    loader: Dict[Any, Any] = MISSING


@dataclass
class ManifestDataset:
    args: Optional[Dict[Any, Any]] = MISSING
    __target__: str = MISSING

# train_ds:
#     __target__:
#       - PLIntegration.datasets.image.ManifestBase
#     args:
#       - manifest_filepath: /home/yjc/PythonProject/PLIntegration/data/glint_asia_manifest.json
#         loader:
#           batch_size: *batch_size
#           num_workers: 8
#           pin_memory: True
#           shuffle: True
#           drop_last: False
#         transformers:
#           __target__:
# #            - torchvision.transforms.ToPILImage
#             - torchvision.transforms.RandomHorizontalFlip
#             - torchvision.transforms.ToTensor
#             - torchvision.transforms.Normalize
#           args:
# #            - null
#             - p: 0.5
#             - null
#             - mean:
#                 - 0.5
#                 - 0.5
#                 - 0.5
#               std:
#                 - 0.5
#                 - 0.5
#                 - 0.5
