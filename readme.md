# 项目名称  
  
<strong style="font-size:16px">把我刷到墙上</strong> 

## 简单介绍  
  
本项目是一个基于 python 的图像处理项目，主要功能是把图片刷到墙上。

本项目使用 OpenCV 库对图片进行解析和保存, 完全使用 numpy 库进行图像的一系列处理(包括图像的变换, 插值, 多个图像的合并等)。

原图片保存在 ./images/ 目录下, 处理后的图片以 png 格式保存在 ./results/ 目录下。

作者还扩展了对 mp4 视频文件的处理，使用 OpenCV 库将视频中的每一帧提取出来，并分别对其进行处理后, 使用 imageio 库将处理后的图片合成为 gif 动图。虽然工人粉刷出动图很疯狂, 但是 that's interesting 😘。

原视频保存在 ./source.mp4 , 处理后的动图保存在 ./result/ 目录下。

注: 输出格式为 "年月日_时分_by_插值算法.文件格式"。😋
  
## 快速开始  
  
python==3.11 你可以通过pip安装本项目的依赖：  
  
```bash  
pip install -r requirements.txt  
```  
或者分别安装：  

```bash  
pip install numpy==1.25.2
pip install opencv_python==4.10.0.84 
pip install opencv_contrib_python==4.10.0.82
pip install imageio==2.31.2
```    
  
然后运行  
  
```bash  
python main.py  
```  

可以通过改变 main.py 中 src_path 来将你喜欢的图片或者视频刷到墙上。(对于其他格式的处理还有待完善...🥲🥲🥲)

可以通过修改 main.py 中 method 来选择插值算法，默认为 nearest , 可选值有 nearest, bilinear, bicubic。

最后，go to ./results/ 目录下查看处理后的图片或者动图。🎉🎉🎉

## 注意事项  
  
本项目仅供学习交流使用，请勿用于商业用途🚫。
有任何问题欢迎联系作者：email: <EMAIL>hhxai_i33@163.com
 微信：y3320013393