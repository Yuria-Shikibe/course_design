# 联邦学习聚合防御方法对比

基于 PyTorch 实现的联邦学习框架，集成 **5 种现有防御聚合方法** + **1 种创新方法 APRA**，支持 MNIST / CIFAR-10 / CIFAR-100 / Tiny-ImageNet-200 数据集。

## 项目结构

```
course_design/
├── requirements.txt
└── src/
    ├── config.py                    # 统一配置
    ├── helper.py                    # 辅助类（设备/模型/存储）
    ├── models.py                    # CNN / CifarCNN / ResNet18
    ├── dataset.py                   # 数据加载 + 非IID 划分
    ├── client.py                    # 客户端 + WeightAccumulator
    ├── utils.py                     # 评估 / 特征提取 / gap统计
    ├── main.py                      # 训练入口
    └── aggregation/
        ├── fedavg.py                # FedAvg 基线
        ├── clip.py                  # 梯度裁剪
        ├── deepsight.py             # DeepSight (DBSCAN 三维聚类)
        ├── foolsgold.py             # FoolsGold (余弦相似度)
        ├── rflbat.py                # RFLBAT (PCA + 多阶段过滤)
        └── apra.py                  # APRA (创新方法)
```

## 环境配置

```bash
pip install -r requirements.txt
```

依赖：`torch`, `torchvision`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `tqdm`

## 聚合方法

| 方法 | 关键参数 | 核心思想 |
|------|---------|---------|
| `avg` | - | 联邦平均，直接对更新求均值 |
| `clip` | `--clip-factor` | 对每个客户端更新的 L2 范数进行上限裁剪 |
| `deepsight` | - | NEUP + DDIF + bias 三维 DBSCAN 聚类，剔除恶意簇 |
| `foolsgold` | - | 基于历史更新的余弦相似度计算信任权重 |
| `rflbat` | - | PCA 降维 + 两阶段欧氏距离过滤 + KMeans 聚类 |
| **`apra`** | `--apra-k-init` `--apra-k-decay` `--apra-pca-components` | **自适应渐进鲁棒聚合：4 阶段融合上述方法** |

## 快速开始

### 基本用法

```bash
# 使用 FedAvg 基线
python src/main.py --agg-method avg --dataset mnist

# 使用 APRA 创新方法
python src/main.py --agg-method apra --dataset mnist

# 使用 DeepSight 防御
python src/main.py --agg-method deepsight --dataset cifar10 --num-malicious 3

# 使用 FoolsGold 防御
python src/main.py --agg-method foolsgold --dataset cifar100
```

### 完整参数

```
核心参数:
  --dataset {mnist,fashion_mnist,cifar10,cifar100,tiny-imagenet-200}
  --model {cnn,resnet18}
  --agg-method {avg,clip,deepsight,foolsgold,rflbat,apra}
  --num-clients 100
  --num-sampled 10
  --num-malicious 0

训练参数:
  --global-epochs 50
  --local-epochs 5
  --batch-size 64
  --lr 0.01
  --momentum 0.9

数据划分:
  --non-iid           # 非 IID 划分（默认）
  --iid               # IID 划分
  --alpha 0.5         # Dirichlet 分布参数

防御参数:
  --clip-factor 1.0   # Clip 方法的裁剪因子

APRA 参数:
  --apra-pca-components 3   # PCA 降维维度
  --apra-base-clip 1.0      # 基础裁剪因子
  --apra-k-init 5.0         # MAD 阈值初始值
  --apra-k-decay 0.1        # MAD 阈值衰减率
  --apra-no-neup-ddif       # 禁用 NEUP/DDIF 特征

系统参数:
  --seed 42
  --folder-path ./results
  --data-dir ./data
```

### 场景示例

```bash
# 场景1: 无攻击 - 对比各方法的基础性能
python src/main.py --agg-method avg --dataset mnist --global-epochs 20
python src/main.py --agg-method apra --dataset mnist --global-epochs 20

# 场景2: 30% 恶意客户端 - 标签翻转攻击
python src/main.py --agg-method avg --dataset cifar10 --num-clients 100 --num-sampled 30 --num-malicious 9
python src/main.py --agg-method apra --dataset cifar10 --num-clients 100 --num-sampled 30 --num-malicious 9

# 场景3: 非IID 环境 - 模拟真实场景
python src/main.py --agg-method apra --dataset fashion_mnist --non-iid --alpha 0.3 --num-malicious 3

# 场景4: APRA 参数调优
python src/main.py --agg-method apra --dataset cifar10 --apra-k-init 8.0 --apra-k-decay 0.2 --apra-base-clip 0.5
```

## APRA 方法原理

**APRA** (Adaptive Progressive Robust Aggregation) 融合四种防御策略的优势：

| 阶段 | 机制 | 设计来源 |
|------|------|---------|
| 1. 多维特征提取 | 分类层权重重塑 + NEUP/DDIF 行为特征 + PCA 降维 | DeepSight + RFLBAT |
| 2. 自适应 MAD 预过滤 | L2 范数的 MAD 鲁棒异常检测，阈值随 epoch 指数衰减 | **创新点** |
| 3. 层次聚类 + 自动选簇 | Ward 聚类 + 轮廓系数自动确定最优 k，选最大×内部相似度最高簇 | 改进 RFLBAT |
| 4. 信任加权 + 自适应裁剪 | FoolsGold 式 logit 信任权重驱动 Clip 阈值 | 融合 FoolsGold + Clip |

MAD 阈值衰减公式：`k(epoch) = max(2.0, k_init * e^(-k_decay * epoch))`

- 训练初期 (k ≈ 5.0)：容忍较大偏差，防御在学习
- 训练后期 (k → 2.0)：严格过滤，精准剔除恶意更新

## 输出

训练结果保存在 `--folder-path` 目录下（默认 `./results/`）：

```
results/
├── best_model.pth          # 最佳模型权重
├── training.log            # 训练日志
├── foolsgold/              # FoolsGold 历史更新
└── saved_updates/          # RFLBAT 客户端更新
```
