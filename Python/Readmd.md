
# How to train your own model?


This folder contains codes for training VE-Net and IR-Net, which include:

1) Generate "voxel-items" from a PointCloud Map,  
2) Train VE-Net and IR-Net,  
3) Convert a python's .pth model to onnx


## Usage

**Step I. Get training items from a PointCloud**
Run `generate_label.py`, which generate items for further training from a .ply pointcloud.

After run, you'll get voxels with p2v labels in output folder.

**Step II. Train IR-Net and VE-Net**
Run `train.py`, which using training items for training.

After run, you'll get python's .pth model in output folder.


**Step III. Convert .pth to .onnx**
Run `convert_to_onnx.py`, which convert .pth model to .onnx


**Summary**
Give a ply file, run `generate_label` you'll get "training_items" in "Output/" folder.  
Then run `train`, you'll get "joint-model.pth" in "Output/model_output/" folder.  
Finally run `convert_to_onnx`, you'll get ir-net and ve-net .onnx model, which can then be used for C++'s SLAM.  



## Attention
The current code only support import one ply to generate training data.  
If you want use many pointcloud ply, you need to first merge them into a whole ply (by CloudCompare or else) by hand in advance.

You can download a test ply here: [BaiduNet](TODO:)

