---
title: "Visuo-Tactile World Models"
aliases: []
tags:
  - literature-note
  - reading-note
  - world-model
  - tactile-sensing
  - robot-manipulation
  - multimodal-learning
  - contact-rich-tasks
created: "2026-05-30"
source: 
author: Carolina Higuera, Sergio Arnaud, Byron Boots, Mustafa Mukadam, Francois Robert Hogan, Franziska Meier
paper_type: research
theme: "提出首个多任务视觉-触觉世界模型 VT-WM，通过融合视觉与触觉感知，使机器人世界模型在接触丰富的操作中保持物体永恒性并遵循物理规律"
study_area: "机器人操作中的世界模型，聚焦接触丰富任务（堆叠、推、擦拭、放置）的想象与规划"
data_source: "多任务操作数据集（place fruits, push fruits, wipe cloth, stack cubes, scribble with marker），在真实 Franka Panda + Allegro Hand + Digit 360 触觉传感器上评估"
methodology: "Cosmos 视觉编码器 + Sparsh-X 触觉编码器 → 12层 Transformer 预测器（空间-时间自注意力 + 动作交叉注意力）→ CEM 规划器，训练采用 Teacher Forcing + Sampling Loss"
core_variable: "视觉-触觉融合对世界模型想象质量（物体永恒性 Frechet 距离、因果符合度）和零样本规划成功率的影响"
key_finding: "VT-WM 相比纯视觉 V-WM：物体永恒性提升 33%，物理规律符合度提升 29%，零样本真实机器人规划成功率最高提升 35%，仅需 20 个演示即可适应新任务"
relevance: "对构建物理一致性的机器人世界模型有重要启发，触觉-视觉融合范式可直接借鉴到任何接触丰富的机器人操作场景"
---

## 研究问题
- **核心问题**：如何让机器人世界模型在接触丰富的操作任务中保持物理一致性？纯视觉世界模型在遮挡、接触状态模糊时容易产生幻觉（物体消失、瞬移、违反物理规律）。
- **研究动机**：现有视觉世界模型（V-WM）在自由空间运动中表现良好，但在接触丰富任务中存在三大痛点：
  1. **物体永恒性缺失**：遮挡时物体从想象中消失
  2. **物理规律违反**：物体在无外力作用下自发移动或变形
  3. **接触状态歧义**：视觉上无法区分"接触"与"未接触"，导致规划失败
- **与现有工作的差距**：
  - 现有世界模型（V-JEPA、Cosmos、GAIA 等）主要展示生成能力，缺乏真实机器人控制验证
  - 触觉动力学模型（Sutanto et al., Tian et al.）多为任务特定，缺乏通用多任务世界模型
  - 极少有工作将视觉与触觉融合训练通用的 action-conditioned world model

## 方法拆解

### 1. 核心思想

VT-WM 要解决的核心问题是：**纯视觉世界模型无法感知接触，导致在遮挡和接触状态模糊时产生物理不一致的幻觉**。解决方案是引入触觉感知作为补充模态，利用视觉式触觉传感器（Digit 360）提供高分辨率的接触反馈，将世界模型的想象锚定在真实的接触物理中。

如图 1 所示，当机器人在执行 cube stacking 任务时，V-WM 在物体被手遮挡时容易丢失物体表示，而 VT-WM 通过触觉信号保持对蓝色方块的持续感知，在放置和释放后正确恢复其位置。

![VT-WM 框架：视觉+触觉融合](D:/share/docs/papervault/PaperMarkdown/WorldModel/images/83862a092faec0114508c513b5e20cf641a9a84234d4e83962baf7638fd3f6ac.jpg)
*Figure 1: VT-WM 通过触觉补充视觉，提供接触接地，减少 V-WM 中的幻觉，实现更可靠的零样本规划*

### 2. 模型架构（三组件）

![VT-WM 架构细节](D:/share/docs/papervault/PaperMarkdown/WorldModel/images/ea7ea929f1664ad4ee4dbcabb7a9ad101956aaae0df0884c6fbf9baa9b749d04.jpg)
*Figure 3: VT-WM 架构 — Cosmos 编码视觉，Sparsh-X 编码触觉，Transformer 预测器通过因子化空间-时间注意力进行多模态融合*

| 组件 | 输入 | 核心操作 | 输出 | 设计动机 |
|------|------|----------|------|----------|
| **视觉编码器** | 外centric 视频（9帧@6fps, 320×192） | Cosmos Tokenizer（预训练） | 视觉隐状态 $s_k \in \mathbb{R}^d$ | Cosmos 提供高质量的视频表征，比 CLIP 更适合时序建模 |
| **触觉编码器** | Digit 360 触觉图像（4传感器, 2帧@30fps） | Sparsh-X Tokenizer（预训练） | 触觉隐状态 $t_k \in \mathbb{R}^d$ | Sparsh-X 专为 Digit 360 训练，能捕捉接触动力学（力场、滑移、姿态变化） |
| **预测器** | $s_k, t_k, a_k$（拼接后） | 12层 Transformer | $\hat{s}_{k+1}, \hat{t}_{k+1}$ | 统一多模态预测，利用接触信号消除视觉歧义 |

#### 2.1 视觉编码器

- **预训练模型**：Cosmos Tokenizer（Agarwal et al., 2025）
- **输入**：1.5 秒视频片段，9 帧，分辨率 320×192，帧率 6 fps
- **为什么选 Cosmos**：Cosmos 专为物理 AI 设计，在机器人操作视频上预训练，比通用视觉编码器（如 CLIP、DINO）更擅长捕捉空间-时序动态
- **输出**：每帧压缩为视觉 token，形成视觉隐状态序列 $s_k$

#### 2.2 触觉编码器

- **预训练模型**：Sparsh-X（Higuera et al., 2025）
- **输入**：4 个 Digit 360 指尖传感器，每个传感器 2 帧（覆盖最近 0.16 秒）
- **为什么选 Sparsh-X**：Digit 360 是视觉式触觉传感器（通过弹性体表形变成像获取触觉信息），Sparsh-X 通过自监督学习提取低维触觉表征，无需显式标签即可捕捉力场、滑移状态、纹理和材料属性
- **输出**：触觉隐状态 $t_k$
- **传感器物理**：Digit 360 以 30-60 FPS 流式传输图像数据，提供丰富的接触区域信息（力、形状、纹理特征）

#### 2.3 预测器（核心创新）

**输入表示**：
- 视觉 token $s_k$ 和触觉 token $t_k$ 沿**空间维度拼接**，形成统一序列 $\mathbb{R}^{(b, t, s, d)}$
- 动作 $a_k$：7 维控制输入（3D 平移 + 3D 旋转 + 二进制手开合状态）
- 动作序列从 30Hz 分块为每组 5 个 delta 状态，提供完整的驱动历史

**网络结构**：
- 12 层 Transformer
- 位置编码：Rotary Position Embeddings (RoPE)
- 最大上下文长度：9 帧（对视觉和触觉模态）

**注意力机制（关键设计）**：

| 注意力类型 | 作用 | 计算复杂度 |
|-----------|------|-----------|
| **空间自注意力** | 同一 timestep 内所有 token 交互（视觉+触觉+动作） | $O(S^2)$ |
| **时间自注意力** | 每个 token 跨 timestep 演化追踪 | $O(T^2)$ |
| **交叉注意力** | 视觉-触觉 token 融入动作控制输入 | $O(S \cdot A)$ |

- **因子化设计动机**：完整时空注意力的复杂度为 $O((THW)^2)$，对于高分辨率视频不可行。因子化后空间和时间解耦，复杂度大幅降低，同时保留捕捉局部动力学和全局上下文的能力
- **交替模式**：自注意力块 → 交叉注意力块 → 自注意力块 → ... 这种迭代精炼允许模型基于感官观测和动作输入逐步优化隐状态

**输出投影**：
- Transformer 后通过模态特定的输出头，分别预测 $\hat{s}_{k+1}$ 和 $\hat{t}_{k+1}$

### 3. 训练策略

#### 3.1 双目标损失函数

$$\mathcal{L} = \mathcal{L}_{teacher} + \mathcal{L}_{sampling}$$

**Teacher Forcing Loss**（主要训练信号）：

$$\mathcal{L}_{teacher} = \sum_{k=1}^{T-1} ||\hat{s}_{k+1} - s_{k+1}||_1 + ||\hat{t}_{k+1} - t_{k+1}||_1$$

- $\hat{s}_{k+1}, \hat{t}_{k+1}$：基于时间 $k$ 之前的**真实状态**预测下一步
- $s_{k+1}, t_{k+1}$：时间 $k+1$ 的真实编码隐状态
- 使用 L1 损失（而非 L2），对异常值更鲁棒
- 提供密集监督和稳定梯度，但可能导致自回归 rollout 时的分布偏移

**Sampling Loss**（提升长程一致性）：

$$\mathcal{L}_{sampling} = \sum_{k=1}^{H} ||\hat{s}_{k+1}^{sampled} - s_{k+1}||_1 + ||\hat{t}_{k+1}^{sampled} - t_{k+1}||_1$$

- 采样步数：$H = 3 \sim 5$
- 采样状态**不使用梯度**（stop-gradient），防止训练不稳定
- 自回归生成未来状态后，基于这些采样状态继续预测，强制模型在自身预测上也能保持稳定
- **设计动机**：单独使用 Teacher Forcing 时，模型只在真实数据上训练，rollout 时会因为误差累积而漂移。Sampling Loss 迫使模型学习在自身预测分布上的稳定动态

#### 3.2 训练配置

| 超参数 | 取值 |
|--------|------|
| 视觉输入 | 9帧 @ 6fps, 320×192 |
| 触觉输入 | 4传感器 × 2帧 @ 30fps |
| 动作输入 | 7维（3D平移 + 3D旋转 + 手开合） |
| 动作分块 | 5个 delta 状态一组 |
| 最大上下文 | 9 帧 |
| Transformer 层数 | 12 |
| 注意力机制 | 因子化空间-时间 + 动作交叉注意力 |
| 位置编码 | RoPE |
| 损失权重 | Teacher : Sampling = 1 : 1 |
| 视觉编码器 | Cosmos Tokenizer（预训练，冻结/微调） |
| 触觉编码器 | Sparsh-X（预训练，冻结/微调） |

- 训练数据集：多任务操作数据（place fruits, push fruits, wipe cloth, stack cubes, scribble with marker），详细超参数见 Appendix B

### 4. 规划（推理时）

**目标**：给定当前状态和目标图像，规划最优动作序列。

**算法**：Cross-Entropy Method (CEM)

**流程**：
1. **初始化**：从当前视觉和触觉嵌入开始，作为预测器的初始上下文
2. **采样**：每步采样 $N$ 个动作序列 $\{a_{k:k+H}^{(i)}\}_{i=1}^N$，覆盖 horizon $H$
3. **想象**：预测器为每个序列自回归生成未来隐状态 $(s_{k+1:k+H}, t_{k+1:k+H})$
4. **评估**：代价函数 = 最终预测视觉状态 $s_{k+H}$ 与目标图像隐状态 $s_{goal}$ 的 $\ell_2$ 距离
5. **选择**：保留 top-k 最优序列
6. **更新**：向最优序列更新采样分布（均值 + 方差）
7. **迭代**：重复 3-6 直到收敛
8. **执行**：选择最优序列在真实机器人上**开环执行**

**搜索空间**：$\mathbb{R}^7$（3D 腕部位移 + 3D 旋转 + 手开合状态）

**关键设计**：
- **触觉不作为目标信号**：规划目标仅基于视觉（目标图像），触觉的作用是提升世界模型可靠性，从而间接改善规划
- **触觉上下文的价值**：初始状态的触觉嵌入帮助消除视觉歧义（如区分"已接触"和"未接触"），产生更物理一致的想象未来和更准确的代价评估

### 5. 消融实验设计

作者在实验中隐式对比了以下变体：
- **V-WM（基线）**：纯视觉，无触觉
- **VT-WM（完整）**：视觉 + 触觉

通过控制相同的动作序列和初始条件，直接比较 rollout 质量。关键发现：
- 触觉在**接触丰富任务**（push, wipe, stack）中收益最大
- 在**自由空间任务**（reach）中收益很小，验证了"触觉主要解决接触问题"的假设
- 触觉对**因果符合度**的提升（29%）略低于对**物体永恒性**的提升（33%），说明触觉对静态物体的约束更强于对动态交互的建模

## 实验验证
- **数据集 / 指标**：

| 任务                   | 类型    | 评估指标                                            |
| -------------------- | ----- | ----------------------------------------------- |
| Place fruits         | 单目标放置 | Object Permanence (Frechet ↓)、Causal Compliance |
| Push fruits          | 单目标推动 | Object Permanence (Frechet ↓)、Causal Compliance |
| Wipe cloth           | 单目标擦拭 | Object Permanence (Frechet ↓)、Causal Compliance |
| Stack cubes          | 多目标堆叠 | Object Permanence (Frechet ↓)、Causal Compliance |
| Scribble with marker | 精细操作  | Object Permanence (Frechet ↓)、Causal Compliance |
| Reach button         | 自由空间  | 零样本规划成功率                                        |
| Reach & push         | 多子目标  | 零样本规划成功率                                        |
| Place plate (新任务)    | 下游适应  | 20演示微调后成功率                                      |

- **核心结果**：
  - **物体永恒性（图4）**：VT-WM 相比 V-WM 平均降低归一化 Frechet 距离 33%，在 place fruits (p<0.001)、push fruits (p<10^-6)、stack cubes (p<0.05) 上统计显著

    ![物体永恒性对比|564](D:/share/docs/papervault/PaperMarkdown/WorldModel/images/11b6e18f5f5b2c0de4eabb96d8a5e05fc802b686dd18150b56741c1237810b23.jpg)
    *Figure 4: VT-WM 在物体运动中的归一化 Frechet 距离平均降低约 33%（95% CI）*

  - **因果符合度（图6）**：VT-WM 平均降低幻觉运动 29%，在 place fruits (p<0.001)、push fruits (p<0.05)、wipe cloth (p<0.01) 上显著

    ![因果符合度对比](D:/share/docs/papervault/PaperMarkdown/WorldModel/images/ffe4835b52ed6771fb485141dd4767b0c3edbe34e2c3c696e6cba4fe87d1c22b.jpg)
    *Figure 6: VT-WM 在静止物体的 Frechet 距离上优于 V-WM，整体提升约 29%，反映更强的物理规律遵循能力*

  - **零样本规划（图7左）**：VT-WM 在接触丰富任务中显著优于 V-WM：
    - Push fruits: +10% (83% → 92%)
    - Reach & push: +35% (69% → 93%)
    - Wipe cloth: +31% (70% → 92%)
    - Stack cubes: +11% (75% → 83%)
    - Reach button: 持平 (100%)

    ![零样本规划成功率](D:/share/docs/papervault/PaperMarkdown/WorldModel/images/59b92a236877e110b8feaccf89b37d013a86cadb0b794d1c3099440a0b3ea653.jpg)
    *Figure 7 (左): VT-WM 在真实机器人上的零样本规划成功率，蓝色标签表示 VT-WM 的提升幅度；右: 新任务仅需 20 个演示即可达到 77% 成功率*

  - **下游适应（图7右）**：新任务"place plate in dish rack"仅需 20 个演示微调，VT-WM 达到 77% 成功率

- **消融 / 对比实验**：
  - V-WM 与 VT-WM 在相同动作序列条件下的 rollout 对比（图5）：V-WM 在手悬停于布料上方时产生布料位移幻觉，VT-WM 保持布料静止

    ![Rollout 对比](D:/share/docs/papervault/PaperMarkdown/WorldModel/images/2732f04f9b62260e9a6733350dec7c7884d009d5c29b66515f5e2eb19b939907.jpg)
    *Figure 5: VT-WM 防止无外力物体的虚假运动，而 V-WM 常产生意外位移的幻觉*

  - 触觉预测可视化（附录）：VT-WM 能保持视觉和触觉表征的一致性

## 亮点与局限
- **亮点**：
  - 🎯 **首创性**：首个多任务视觉-触觉世界模型，填补了通用触觉世界模型的空白
  - 🔬 **物理接地**：通过触觉信号将想象锚定在接触物理中，显著提升物体永恒性和物理一致性
  - 🤖 **真实机器人验证**：不仅在模拟中有效，更在真实 Franka + Allegro Hand 上验证零样本规划
  - 📊 **量化清晰**：明确量化了触觉对想象质量（33%/29%）和规划成功率（最高+35%）的提升
  - 🔄 **数据高效**：仅需 20 个演示即可适应新任务，体现多任务预训练的价值
  - 🏗️ **架构优雅**：因子化时空注意力 + 交叉注意力动作条件化的设计兼顾效率与表达能力

- **局限 / 隐含假设**：

  - 实验仅在 5 个操作任务上验证，任务多样性有限（水果、方块、布料、马克笔）
  - 使用特定的 Digit 360 触觉传感器，对其他触觉传感器（如 GelSight）的泛化性未验证
  - 规划为开环执行（open-loop），未考虑闭环反馈修正
  - 触觉预测仅在附录中展示，未作为评估指标
  - Scribble with marker 任务上因果符合度未显著提升（甚至可能退化），说明对某些精细操作触觉帮助有限
  - CEM 规划计算成本较高，未与模型预测控制（MPC）等更高效方法对比

## 与我工作的关联
- **可直接借鉴**：
  - 视觉-触觉融合的多模态编码范式可推广到任何接触丰富的机器人任务
  - Teacher Forcing + Sampling 的双目标训练策略可提升长程 rollout 一致性
  - CEM + 世界模型的规划框架可作为接触丰富操作的基线方案
  - 触觉表征预训练（Sparsh-X）+ 世界模型微调的两阶段训练流程

- **可作为 baseline / 对比**：
  - 任何涉及世界模型、机器人操作、触觉感知的工作都可引用本文作为 SOTA baseline
  - 视觉-触觉融合 vs 纯视觉的对比实验设计可作为后续研究的参考

- **可改进方向**：
  - 引入闭环 MPC 替代开环 CEM，提升规划鲁棒性
  - 探索更多触觉传感器类型（GelSight、BioTac 等）的泛化性
  - 结合力/力矩传感，扩展触觉信号维度
  - 研究触觉表征的可解释性，理解模型关注哪些接触特征
  - 扩展到动态操作（投掷、滑动、弹跳等更复杂的接触模式）
  - 与强化学习结合，实现世界模型驱动的策略学习

## 双向链接
- 相关概念：
  - [[World Model]] — 世界模型，机器人想象与规划的核心
  - [[Tactile Sensing]] — 触觉感知，Digit 360 等视觉式触觉传感器
  - [[Multimodal Learning]] — 多模态学习，视觉+触觉融合
  - [[Object Permanence]] — 物体永恒性，世界模型的核心物理理解能力
  - [[Cross-Entropy Method]] — 交叉熵方法，用于动作序列优化
  - [[Contact-Rich Manipulation]] — 接触丰富操作
- 相关论文：待通过双向链接补全功能自动匹配
