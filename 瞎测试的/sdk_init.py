import sys
import os
# sys.path.append(os.getenv('MVCAM_COMMON_RUNENV') + "/Sample/python/MvImport")
sys.path.append("C:/Program Files (x86)/MVS/Development" + "/Sample/python/MvImport")
import importlib
mv = importlib.import_module("MvCameraControl_class")
from MvCameraControl_class import *
params = importlib.import_module("CameraParams_header")
err = importlib.import_module("MvErrorDefine_const")

if __name__ == "__main__":
    # 1. 初始化SDK
    MvCamera.MV_CC_Initialize()

    # 2. 进行设备发现，控制，图像采集等操作

    # 3. 反初始化SDK
    MvCamera.MV_CC_Finalize()
