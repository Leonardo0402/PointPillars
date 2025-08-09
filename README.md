# Point Pillars Implementation

- [x] config.py
- [x] network.py
- [x] reader.py
- [x] processor.py
- [x] lossfunction.py
- [x] train.py
- [ ] inference.py

# kitti dataset
## label.txt 说明
0. 类别
1. 截断程度
2. 遮挡率
3. 观察角度
45. 2D bounding box左上角坐标
67. 2D bounding box右下角坐标
8910. 3D bounding box 的length，width， height
111213. 3D boudning box 在相机的坐标
14. 相对y轴的旋转角度
## calibration.txt 说明
Tr_velo_to_cam maps a point in point cloud coordinate to reference co-ordinate.
## 文件夹结构(/home/neil/disk/kitti)
```
       |--- testing -- velodyne(000000.bin~007517.bin)
kitti -|
       |--- traning -- label_2(000000.txt~007480.txt)
                  -- velodyne(000000.bin~007480.bin)
```
# make pillars算法
```
输入：包含n个4维点的点云
输出：包含n个9维点的点云
1. 创建一个pillars的字典，key为center，value为所包含的点的list，初始化为空.
2. 对于点云中的点，进行遍历。
（1）判断点是否在范围内。如果是，=>（2），否则跳过进入下一个点。
（2）判断点在哪个pillar内，加入对应pillar的list。
3. 对于已经创建好的pillars的字典内容进行遍历。
（1）如果该list的点含量大于100，随机采样其中的100个点，保留下来。如果该list的点含量小于100，用0填充至100。如果该list的点含量等于100，进入（2）。
（2）对于已经处理好的包含100个点的list进行遍历，将每个点由4维扩展为9维。
（3）将一个list转化为一个numpy矩阵。
4. 将字典中的所有numpy矩阵转化为一个numpy矩阵，输出。
    
```

# pybind编译指令
> c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) point_pillars.cpp -o point_pillars$(python3-config --extension-suffix)


# 🌟Awesome Links
[Kitti介绍（来自medium）](https://medium.com/test-ttile/kitti-3d-object-detection-dataset-d78a762b5a4)

# 常见问题解答
1. **calibration file在哪里？为什么要做一个变换？**  
   标定文件位于 KITTI 数据集的 `training/calib` 与 `testing/calib` 目录，提供从 LiDAR 到相机坐标系的变换矩阵。标签数据在相机坐标系下给出，因此需要利用该矩阵将点云转换到同一坐标系以进行训练和评估。
2. **ground truth是怎么做出来的？如何理解cpp文件的内容？**  
   Ground truth 来自 `label_2` 文件中的 3D 边界框参数，训练时根据这些标注生成锚框目标。仓库中的 `point_pillars.cpp` 和 `make_pillars.cpp` 使用 C++/pybind11 加速点云划分成柱状体（pillar）的过程，是数据预处理的实现。
3. **heading 和 angle 的区别？**  
   `angle` 是相对于锚框的连续偏航角回归量，而 `heading` 是二值分类，区分物体朝向前或后，用于解决 180° 方向模糊的问题。
4. **把focal loss中的BCE换掉了才跑得起来，哪里出错了？**  
   原实现先对输出做 Sigmoid 再使用 `BCELoss`，容易出现数值不稳定。改用 `BCEWithLogitsLoss` 直接接受 logits，无需手动 Sigmoid，使得 Focal Loss 计算更稳健。
