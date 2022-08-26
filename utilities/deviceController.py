# -*- coding: utf-8 -*-
import os
import torch

# import pynvml as p


def get_torch_device(device="0"):
    """
    device '0','1','2' or `cpu`
    """
    if isinstance(device, str):
        if device.lower() == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{str(device)}")
    return torch.device("cpu")


def get_environ_info():
    """
    Obtain current working environment information
    """
    info = dict()
    info["device"] = "cpu"
    info["num"] = int(os.environ.get("CPU", 1))
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        gpu_num = 0
        try:
            gpu_num = torch.cuda.device_count()
        except:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            pass

        if gpu_num > 0:
            info["device"] = "cuda"
            info["num"] = gpu_num
    return info


# def get_single_gpu_memory_usage(index):
#     """
#     get a single gpu memory usage
#     """
#     p.nvmlInit()
#     handler = p.nvmlDeviceGetHandleByIndex(index)
#     mem = p.nvmlDeviceGetMemoryInfo(handler)
#     name = p.nvmlDeviceGetName(handler)
#     attr = {"name": name, "free": mem.free, "used": mem.used, "total": mem.total}
#     p.nvmlShutdown()
#     return attr


# def get_all_gpu_memory_usage():
#     """
#     get all gpus' memory usage
#     """
#     p.nvmlInit()
#     deviceCount = p.nvmlDeviceGetCount()
#     gpu_mem_infos = []
#     for i in range(deviceCount):
#         attr = get_single_gpu_memory_usage(i)
#         gpu_mem_infos.append(attr)
#     p.nvmlShutdown()
#     return gpu_mem_infos
