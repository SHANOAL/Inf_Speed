# 相关工作调研与分析

## 1. Jenga: Efficient Sparse Attention with Block-Wise Attention Redistribution (2505.16864)

### 核心方法
Jenga 提出了一种基于 **块级稀疏注意力** 的高效推理方案，主要包含三个部分：

1. **3D Space-Filling Curve (3D SFC)**：将 2D/3D token 映射到 1D 序列时，使用 3D 空间填充曲线（类似 Hilbert 曲线）而非传统的 raster scan。这样做的好处是空间上相邻的 token 在 1D 序列中也更接近，使得 block-wise attention 天然保留了更多局部空间信息。

2. **Block-wise Sparse Attention**：将序列分成固定大小的 block，每个 block 只与自身以及若干关键 block 做注意力计算。通过注意力分数统计来确定哪些 block 间的交互是重要的。

3. **Progressive Resolution Strategy**：在推理过程中渐进地提升分辨率——低分辨率 scale 用完整注意力确保全局一致性，高分辨率 scale 用稀疏注意力减少计算。这与我们在 InfinityStar 中的多 scale 架构天然契合。

### 关键发现
- 不同 transformer 层对稀疏化的敏感度不同：浅层和深层更敏感（U 型曲线）
- 注意力分布在大多数 block pair 上是稀疏的，只有少数 block pair 有显著的注意力值
- 3D SFC 排列比 raster scan 在相同稀疏度下保留更多空间局部性

### 可借鉴点
| 思想 | 如何应用到 InfinityStar |
|------|------------------------|
| **层敏感度 U 型分布** | 在 hybrid pruning 中，不是简单地让前 N 个 chunk 全量、后面全部剪枝，而是根据 U 型规律让首尾 chunk 全量/低剪枝率，中间 chunk 高剪枝率 |
| **渐进分辨率策略** | 低 scale（全局结构）不剪枝，高 scale（局部细节）逐步增大剪枝比例——目前我们已经在低 scale 跳过剪枝，但比例可以更精细 |
| **3D SFC 排列** | 在计算 token importance 时，将 token 按空间填充曲线排列后取局部窗口的统计量，可能比当前的逐帧独立 PCA+高频计算更好 |


## 2. Understanding Attention in Transformers: A Guide to Attention Entropy (2504.12027)

### 核心方法
这篇论文提供了 **attention entropy** 的系统分析框架：

1. **Attention Entropy 定义**：对每个 head 的注意力分布计算信息熵 H = -sum p*log(p)。高熵 = 注意力分散（关注全局信息），低熵 = 注意力集中（关注局部/特定位置）。

2. **层间 Entropy 分布规律**：
   - 浅层：熵较高，建立全局上下文表示
   - 中间层：熵下降，执行特征提取和细化
   - 深层：熵回升，整合全局信息用于最终输出
   - 这形成了 "高-低-高" 的 U 型曲线

3. **Entropy 与模型质量的关系**：
   - 训练良好的模型在各层有更清晰的 entropy pattern
   - 量化/剪枝后的模型 entropy pattern 被打乱时质量下降明显
   - Entropy 可以作为 "free" 的质量指标（不需要额外计算）

### 关键发现
- **Pruning 安全区**：中间层的 entropy 最低（最聚焦），移除这些层对模型影响相对较小
- **Quality proxy**：可以用 attention entropy 的变化量来预测剪枝/加速对质量的影响
- **Head importance**：不同 head 的 entropy 差异很大，某些 head 始终高熵（全局 head），某些始终低熵（局部 head）

### 可借鉴点
| 思想 | 如何应用到 InfinityStar |
|------|------------------------|
| **Entropy 作为层级剪枝指导** | 在 hybrid pruning 中，可以预先 profiling 每层的 attention entropy，自动确定哪些层适合全量通过、哪些适合激进剪枝 |
| **Entropy 作为质量监控** | 在推理过程中实时监控 attention entropy，当 entropy pattern 偏离 baseline 过多时动态降低剪枝率 |
| **Head 级别差异化** | 不同的 attention head 可以有不同的 token pruning mask——全局 head（高熵）保留更多 token，局部 head（低熵）可以更激进地剪枝 |
| **自适应 CFG bypass** | 用 attention entropy 差异来判断 cond 和 uncond 是否足够接近，作为 CFG bypass 的触发条件 |


## 3. FVAR: Frequency-Driven Vision Autoregressive Model (2511.18838)

### 核心方法
FVAR 从频率域角度重新设计了视觉自回归模型：

1. **Next-Focus Prediction**：不是直接预测下一个 scale 的全部内容，而是预测一个 "focus map"（通过 PSF kernel 参数化），指示模型应该关注哪些空间区域。这本质上是一种 learned attention/importance map。

2. **Point Spread Function (PSF) Kernels**：用 PSF 参数（位置、方差、幅度）来建模每个 scale 的空间注意力分布。PSF 参数少（每个 scale 几十个参数），但能生成连续的重要性权重 map。

3. **High-Frequency Residual Learning**：将预测分为低频基础 + 高频残差。低频部分直接从上一个 scale 上采样，模型只需要学习高频细节的增量。这显著降低了学习难度。

4. **频率感知 Loss**：在不同频段上计算 loss，低频用 L2，高频用 perceptual loss / adversarial loss。

### 关键发现
- 视觉生成中，大部分低频信息在早期 scale 已经确定，后续 scale 主要添加高频细节
- 空间上重要性不均匀：边缘、纹理丰富区域需要更多计算资源
- PSF 参数化的 importance map 比 per-token score 更平滑、更稳定

### 可借鉴点
| 思想 | 如何应用到 InfinityStar |
|------|------------------------|
| **PSF 参数化 importance map** | 用低维 PSF 参数来参数化 token importance，而非对每个 token 独立计算分数。这能产生更平滑的 importance map，避免边缘 token 被频繁误判为不重要 |
| **高频残差思想** | 在 token recovery 时，不是直接插值完整特征，而是插值 "相对于上一次缓存的残差"。即 result = cache + interpolate(pruned_output - cache[selected])，可能保留更多细节 |
| **频率域分析** | 在 token importance scoring 中加入频率域信息：对 feature map 做 2D FFT，高频能量集中的区域（边缘、纹理）给予更高重要性，这与我们现有的 high-pass texture scoring 互补 |
| **Next-focus 思想** | 当前 scale 的 pruning mask 可以参考上一个 scale 的 attention pattern/importance distribution 来 "预测" 下一个 scale 的重要区域，实现跨 scale 的 importance 传递 |


## 4. 综合分析与建议

### 优先级排序

**高优先级（直接可用）：**
1. **U 型层敏感度感知的 hybrid pruning** (Jenga + Entropy)
   - 当前 hybrid pruning 是简单的 "前 N chunk 全量 + 后面剪枝"
   - 改进：首尾 chunk 全量或低剪枝率，中间 chunk 激进剪枝
   - 实现难度低，只需修改 prune_full_chunks_per_scale 的逻辑
   
2. **高频残差插值恢复** (FVAR)
   - 当前 IDW 插值是对完整特征做插值
   - 改进：插值残差而非完整值，即 result = cache + interpolate(output - cache[selected])
   - 实现难度低，只需修改 partial_update_recover_interp

3. **Attention Entropy 自适应 CFG bypass** (Entropy)
   - 计算 cond 和 uncond 的 attention entropy 差异
   - 差异小时 bypass CFG（两者注意力模式接近 -> 引导不需要）
   - 比当前硬编码 "最后两个 scale" 更智能

**中优先级（需要实验验证）：**
4. **文本-视觉联合 attention 重要性** 
   - 用文本 token 的 cross-attention score 作为辅助信号
   - 文本关注度高的视觉区域应该得到更高的 importance
   - 需要在 attention 层中提取 cross-attention score

5. **跨 scale 的 importance 传递** (FVAR next-focus)
   - 当前每个 scale 独立计算 importance
   - 改进：上一个 scale 的 importance map 作为先验，当前 scale 的 scoring 在此基础上调整
   - 可以减少 importance scoring 的噪声和不一致

**低优先级（研究性探索）：**
6. **PSF 参数化 importance map** (FVAR)
   - 用连续参数化的高斯混合来描述重要性分布
   - 更平滑但灵活度有限，适合作为正则化手段

7. **Head 级别差异化剪枝** (Entropy)
   - 不同 head 使用不同的 pruning mask
   - 实现复杂度较高，需要修改 attention kernel

### 与现有系统的集成路径

当前系统:
  scoring (v1/v2) -> pruning -> transformer layers -> PU/interp recovery

改进后:
  entropy profiling -> layer-aware pruning schedule
  text attention -> auxiliary importance signal  
  scoring (v1/v2 + freq) -> pruning -> hybrid layers -> residual-interp recovery
  entropy diff -> adaptive CFG bypass

### 实验建议
1. 先跑完当前的 interp recovery 和 hybrid pruning 实验，建立 baseline
2. 在 interp recovery 基础上尝试残差插值（改动很小）
3. 实现 entropy-based 自适应 CFG bypass，与硬编码方案对比
4. 实现文本 attention 辅助 importance scoring，看对边缘区域的改善
