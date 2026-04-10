# OOP Final Report

本项目简单做了一些关于 imu 的研究，包括 cnn-mlp, ar-cnn, lstm, gnn 对比 baseline
（传统算法）下的表现。

## 目录

`code` 为具体的训练推理代码，其中有 conda 环境所需的 python 包，以及生成报告中的图片。

`flake.nix` `flake.lock` 为配置 `typst` 中所需字体环境。

## 运行

```
$ cd code
$ conda env create -f environment.yml
$ conda activate oop
$ python generate_figures.py
```

## 编译 PPT

```
$ nix build
$ open result/main.pdf
```
