# SelfCap Dataset

Long multi-view videos collected for the SIGGRAPH Asia 2024 (TOG) paper: [Representing Long Volumetric Video with Temporal Gaussian Hierarchy](https://zju3dv.github.io/longvolcap/).

## Content

Camera parameter convensions follow [EasyVolcap](https://github.com/zju3dv/EasyVolcap). Some sequences contain an extra synchronization correction list (`time computed from frame index - sync.json = actual timestamp`).

We also provide a set of point clouds extracted from multiview images using various tools like [COLMAP](https://colmap.github.io) and [RealityCapture](https://www.capturingreality.com/), which were used as initialization for training the Temporal Gaussian Hierarchy model for the paper.

Note that the released dataset are compressed into videos to save bandwidth and space. 
You can extract the images using tools like ffmpeg following scripts like [this](https://github.com/zju3dv/EasyVolcap/blob/main/scripts/preprocess/extract_videos.py).

If you encountered any problems when using the dataset, feel free to contact [Zhen Xu](https://zhenx.me).

- `bar`: 
  - 3540 frames at 60 FPS (~1 min)
  - 2160p
  - 18 cameras
  - dense point clouds (every 1000 frames), sparse (every frame) point clouds
  - no sync.json provided.
- `corgi`: 
  - 3500 frames at 60 FPS (~1 min)
  - 2160p
  - 24 cameras
  - dense point clouds (every 1000 frames), sparse (every frame) point clouds
  - extra synchronization correction provided in `optimized/sync.json`.
- `bike`: 
  - 37377 frames at 60 FPS (~10 min)
  - 1024x1024
  - 22 cameras
  - same as `corgi` but with denser sparse point clouds.
- `hair`:
  - 6500 frames at 60 FPS (~2 min)
  - 2160p
  - 24 cameras
  - same as `corgi` but with denser sparse point clouds.
- `dance`: 
  - 8200 frames at 60 FPS (~2.5 min)
  - 2160p
  - 24 cameras
  - same as `corgi` but with denser sparse point clouds.
- `yoga`: 
  - 10300 frames at 60 FPS (~3 min)
  - 2160p
  - 24 cameras
  - same as `corgi` but with denser sparse point clouds.

For the LongVolcap paper we only performed qualitative analysis and want to achieve the best quality possible (mainly for the realtime rendering demo), thus no extra testing views are held out. We used 0.5x downsampled images for training to make the process faster. We used the videos as their full speed (60 fps) without subsampling. For bike, we used the 15000-21000th frames for the 6000-frame model, 15000-33000th frames for the 18000-frame model. For dance, hair and yoga, we used the 6000-12000th frames. For corgi, we used 5000-12000th frames. The bar model uses all existing frames.



For the FreeTimeGS paper, we summarize the quantitative evaluation protocol as shown in the table below. For scenes with a downsample ratio of 0.5, we first perform COLMAP undistortion with `blank_pixels=0`, and then downsample by a factor of 0.5 using INTER_AREA. For scenes with a downsample ratio of 1.0, we perform COLMAP undistortion with `blank_pixels=0` without downsampling.

| FreeTimeGS Scene | SelfCap Scene    | Test View | Training Views        | Frame Indices | Downsample Ratio |
| ---------------- | ---------------- | --------- | --------------------- | ------------- | ---------------- |
| dance1           | hair-release     | 0015.mp4  | the rest of the views | [4120,4180)   | 0.5              |
| dance2           | hair-release     | 0015.mp4  | the rest of the views | [5530,5590)   | 0.5              |
| corgi1           | corgi-release    | 0007.mp4  | the rest of the views | [200,260)     | 0.5              |
| corgi2           | corgi-release    | 0007.mp4  | the rest of the views | [2950,3010)   | 0.5              |
| bike1            | bike-release     | 0009.mp4  | the rest of the views | [8900,8960)   | 1.0              |
| bike2            | bike-release     | 0009.mp4  | the rest of the views | [30020,30080) | 1.0              |
| dance3           | not released yet |           |                       |               | 0.5              |
| dance4           | not released yet |           |                       |               | 0.5              |


## 相机外参格式与存储方式
坐标系约定：OpenCV 约定（世界到相机变换 w2c = [R|T]）
旋转表示：轴角（Rodrigues 向量），字段名 R_相机名 。
平移表示：3×1 列向量，字段名 T_相机名 。
存储文件：extri.yml，位于数据集根目录（多视图）或 cameras/相机名/（单目。
写入格式：通过自定义 FileStorage 写为 OpenCV YAML 格式，支持 !!opencv-matrix。
读取处理：读取时用 cv2.Rodrigues(Rvec)[0] 将轴角转为 3×3 旋转矩阵，并与 T 拼接为 [R|T]。
示例字段（extri.yml）
names: [00, 01]  
R_00: !!opencv-matrix  
  rows: 3  
  cols: 1  
  dt: d  
  data: [r1, r2, r3]  # 轴角  
T_00: !!opencv-matrix  
  rows: 3  
  cols: 1  
  dt: d  
  data: [tx, ty, tz]

## License

The ***SelfCap*** dataset is released under the non-commercial, research-only custom zju3dv license. Please contact [Prof. Xiaowei Zhou](https://xzhou.me) for any commercial usage inquiries.

## Citation

```bibtex
@Article{xu2024longvolcap,
  author  = {Xu, Zhen and Xu, Yinghao and Yu, Zhiyuan and Peng, Sida and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  title   = {Representing Long Volumetric Video with Temporal Gaussian Hierarchy},
  journal = {ACM Transactions on Graphics},
  number  = {6},
  volume  = {43},
  month   = {November},
  year    = {2024},
  url     = {https://zju3dv.github.io/longvolcap}
}

@Article{xu2023easyvolcap,
  title     = {EasyVolcap: Accelerating Neural Volumetric Video Research},
  author    = {Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle = {SIGGRAPH Asia 2023 Technical Communications},
  year      = {2023}
}

@Inproceedings{xu20234k4d,
  title     = {4K4D: Real-Time 4D View Synthesis at 4K Resolution},
  author    = {Xu, Zhen and Peng, Sida and Lin, Haotong and He, Guangzhao and Sun, Jiaming and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei},
  booktitle = {CVPR},
  year      = {2024}
}

@Inproceedings{wang2025freetimegs,
  author  = {Wang, Yifan and Yang, Peishan and Xu, Zhen and Sun, Jiaming and Zhang, Zhanhua and Chen, Yong and Bao, Hujun and Peng, Sida and Zhou, Xiaowei},
  title   = {FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction},
  booktitle = {CVPR},
  year    = {2025},
  url     = {https://zju3dv.github.io/freetimegs}
}
```