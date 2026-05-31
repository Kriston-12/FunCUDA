# CUDA Reduction 优化路线总报告（v0 -> v4，并可扩展到后续版本）

## 目标

这份报告总结 `my_reduce_v0_global_memory.cu` 到 `my_reduce_v4_add_during_load.cu` 的优化路径，并把每一步优化与 Nsight Compute 报告中的关键指标变化对应起来。

报告重点回答三个问题：

1. 每一版代码到底改了什么。
2. NCU 指标为什么这样变化。
3. 从 roofline 和 GPU 架构角度看，性能变化的根本原因是什么。

---

## 1. 平台与架构背景

从 Nsight Compute 报告可以确定当前设备为：

- Compute Capability: `7.5`
- `#SMs = 40`
- warp size = `32`
- 每个 SM 有多路 warp scheduler
- shared memory 按 bank 组织，连续地址映射到不同 bank，跨线程的跨步访问容易产生 bank conflict

对于 reduction 这类 kernel，算法特征是：

- 每处理一个输入元素，做的浮点运算很少，主要是 `load + add`
- 算术强度很低（Arithmetic Intensity 低）
- 因此通常不会接近 FP32 计算峰值，而更容易受以下因素限制：
  - global memory 访问模式
  - shared memory bank conflict
  - warp 内谓词化/分支造成的有效线程数下降
  - 每轮 reduction 的同步与调度开销

这也和所有报告中的 roofline 现象一致：

- `v0` 到 `v3` 都显示 achieved FP32 performance 接近 `0%`
- `v4` 也只有接近 `1%` FP32 peak

这说明整个优化过程并不是在把 kernel 推向“计算峰值”，而是在不断减少：

- memory system 的低效访问
- shared memory replay / bank conflict
- 不必要的线程和指令工作量

换句话说，这条优化路线的核心不是“让 ALU 更忙”，而是“让每个输出结果所需的代价更小”。

---

## 2. 版本对应关系

| 版本 | 代码 | 报告 |
|---|---|---|
| v0 | `my_reduce_v0_global_memory.cu` | `naive-non-shared-mem.txt` |
| v1 | `my_reduce_v1_shared_memory.cu` | `naive-shared-mem.txt` |
| v2 | `my_reduce_v2_no_divergence_bank_conflict.cu` | `naive-shared-no-div-bank-conf.txt` |
| v3 | `my_reduce_v3_no_bank_conflict.cu` | `v3_no_bank_conflict.txt` |
| v4 | `my_reduce_v4_add_during_load.cu` | `v4_add_during_load.txt` |

---

## 3. 总览表

| 版本 | Duration (ms) | SM Active Cycles | Executed Instructions | Memory Throughput (GB/s) | Max Bandwidth (%) | 主要问题 / 特征 |
|---|---:|---:|---:|---:|---:|---|
| v0 | 5.36 | 3130623.23 | 283901952 | 61.82 | 31.59 | global memory 原地规约，访问低效，分支效率差 |
| v1 | 4.86 | 2840013.15 | 254410752 | 37.40 | 50.27 | shared memory 规约，global traffic 降低 |
| v2 | 5.21 | 3047010.50 | 270139392 | 34.87 | 46.87 | 去发散但引入严重 shared bank conflict |
| v3 | 3.18 | 1860583.90 | 104464384 | 57.93 | 76.71 | sequential addressing，去掉 bank conflict |
| v4 | 1.65 | 963812.12 | 56492032 | 111.60 | 76.73 | add-during-load，单 block 处理 2 倍输入 |

从最终结果看：

- `v0 -> v4`：Duration 从 `5.36 ms` 降到 `1.65 ms`，约下降 `69.2%`
- `SM Active Cycles` 从约 `313 万` 降到约 `96 万`，约下降 `69.2%`
- `Executed Instructions` 从约 `2.84 亿` 降到约 `5650 万`，约下降 `80.1%`

这说明这条优化路线本质上是在持续减少“完成同样一层 reduction pass 所需的总工作量”。

---

## 4. 分阶段分析

## 4.1 v0: global memory 原地规约

### 代码特征

v0 的关键实现：

```cpp
float *blockstart = input + blockIdx.x * blockDim.x;
for (int i = 1; i < blockDim.x; i *= 2) {
    if (threadIdx.x % (2 * i) == 0) {
        blockstart[threadIdx.x] += blockstart[threadIdx.x + i];
    }
    __syncthreads();
}
```

特点：

- reduction 直接在 global memory 上进行
- 每一轮都对 global memory 做读改写
- 活跃线程由 `%` 条件控制
- 访问间隔随着 `i` 增大而变化，容易造成低效访问和 warp 内效率下降

### 指标现象

- `Duration = 5.36 ms`
- `Memory Throughput = 61.82 GB/s`
- `Max Bandwidth = 31.59%`
- `Branch Efficiency = 65.44%`
- `Avg. Divergent Branches = 38502.40`
- 报告明确指出：
  - global load 不理想
  - global store 不理想
  - excessive sectors 很高

### 根本原因

这是最典型的“低算术强度 + 低效 global memory 访问”版本。

从 roofline 角度看：

- reduction 本来就是低 arithmetic intensity
- v0 还把中间规约过程放在 global memory 上做
- 于是大量性能浪费在高延迟 global memory 往返上，而不是浮点运算本身

从 GPU 架构角度看：

- global memory 延迟远高于 shared memory
- strided / irregular global access 破坏 coalescing
- `%` 风格条件导致 warp 内有效线程比例下降
- 结果是：既没有吃满带宽，也没有吃满计算单元

因此 v0 的问题不是单一瓶颈，而是：

- memory access inefficiency
- branch/divergence inefficiency
- 同步开销相对昂贵

---

## 4.2 v1: 引入 shared memory

### 代码变化

v1 的核心变化是先把 block 的数据搬到 shared memory，再在 shared memory 中做 reduction：

```cpp
__shared__ float sdata[THREADS_PER_BLOCK];
sdata[threadIdx.x] = blockstart[threadIdx.x];
```

后续 reduction 仍然采用和 v0 类似的 `%` 条件逻辑。

### 指标变化

相对 v0：

- `Duration: 5.36 -> 4.86 ms`，约提升 `9.3%`
- `SM Active Cycles: 3130623 -> 2840013`，下降约 `9.3%`
- `Executed Instructions: 283.9M -> 254.4M`，下降约 `10.4%`
- `Branch Efficiency: 65.44% -> 100%`
- `Avg. Divergent Branches: 38502.40 -> 0`
- `Memory Throughput: 61.82 -> 37.40 GB/s`，反而下降
- `Max Bandwidth: 31.59% -> 50.27%`，提升

### 为什么 throughput 降了但性能反而变好

这是一个很重要的点。

`Memory Throughput (GB/s)` 下降并不意味着更慢。这里更合理的解释是：

- v1 显著减少了 global memory traffic
- 中间 reduction 不再反复访问 DRAM/L2
- 因而“总传输字节数”下降
- 即使运行更快，最终测得的 GB/s 也可能变小

也就是说，v1 是“少搬了很多没必要搬的数据”，不是“搬得更慢”。

### 根本原因

从 GPU 架构角度看，v1 之所以提升，是因为它把 block 内频繁访问的数据留在 shared memory：

- shared memory 延迟远低于 global memory
- block 内重复使用的数据不再反复经过 L2 / DRAM
- roofline 上看，算术强度虽然没有本质跃迁，但每个输出结果对应的 memory traffic 明显下降

所以 v1 的提升本质是：

**先把数据放到更靠近 SM 的片上存储，再做 reduction，减少昂贵的 global memory 往返。**

---

## 4.3 v2: 试图消除 divergence，但引入 bank conflict

### 代码变化

v2 改成：

```cpp
for (int i = 1; i < blockDim.x; i <<= 1) {
    if (threadIdx.x < blockDim.x / (i << 1)) {
        int idx = threadIdx.x * (i << 1);
        sdata[idx] += sdata[idx + i];
    }
    __syncthreads();
}
```

目标是：

- 不再通过 `%` 判断线程是否活跃
- 改成“前半部分线程工作”的形式，试图改善 divergence

### 指标变化

相对 v1：

- `Duration: 4.86 -> 5.21 ms`，变慢约 `7.2%`
- `SM Active Cycles: 2840013 -> 3047010`，增加约 `7.3%`
- `Executed Instructions: 254.4M -> 270.1M`，增加约 `6.2%`
- `Branch Efficiency` 仍为 `100%`
- 但新增严重警告：
  - shared load 平均 `3.8-way` bank conflict
  - shared store 平均 `2.8-way` bank conflict
  - excessive shared wavefronts 很高

### 根本原因

这一版说明了一个关键事实：

**shared memory 不是天然快，访问模式同样决定性能。**

在 CC 7.5 架构上：

- shared memory 被划分为 bank
- 一个 warp 中多个线程如果落到同一 bank，会发生 bank conflict
- bank conflict 会导致访问被拆分/replay，延长访问完成时间

v2 的 `idx = threadIdx.x * (2 * i)` 正好制造了跨步访问：

- 当 `i` 增大时，同一 warp 的线程访问地址变得更稀疏
- 多线程容易映射到同一 bank
- 原本想减少 divergence，结果把瓶颈转移到了 shared memory bank conflict

从 roofline 角度看：

- 这不是 arithmetic intensity 的问题
- 也不是 FP32 算力不足
- 而是 memory hierarchy 的更底层层次出了问题：shared memory replay 增多，导致 SM 需要花更多周期等待同样的数据交换完成

这就是为什么 v2 虽然在控制流上更“整齐”，性能却比 v1 更差。

---

## 4.4 v3: sequential addressing，真正消除 bank conflict

### 代码变化

v3 改成经典 sequential addressing：

```cpp
for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
        sdata[threadIdx.x] += sdata[threadIdx.x + i];
    }
    __syncthreads();
}
```

特点：

- 活跃线程始终是前半段连续线程
- shared memory 访问地址是连续的
- 避开 v2 的 stride-based bank conflict

### 指标变化

相对 v2：

- `Duration: 5.21 -> 3.18 ms`，提升约 `39.0%`
- `SM Active Cycles: 3047010 -> 1860584`，下降约 `38.9%`
- `Executed Instructions: 270.1M -> 104.5M`，下降约 `61.3%`
- v2 中的 shared bank conflict 警告消失
- `Max Bandwidth: 46.87% -> 76.71%`
- `Memory Throughput: 34.87 -> 57.93 GB/s`

### 为什么这是一个大跳跃

这是整个优化路径中的第一个“结构性拐点”。

v3 的收益来自两个层面：

1. **shared memory 访问模式正确了**
   - 线程访问连续地址
   - 避免 bank conflict
   - 减少 shared memory replay

2. **索引逻辑更简单了**
   - 不再每轮计算跨步 `idx`
   - 控制逻辑与地址生成都更直接
   - 让每一轮 reduction 的真实开销更接近理想值

从 GPU 架构角度看，v3 终于让这个 kernel 更接近“shared-memory-friendly”的实现方式：

- shared memory 真正承担低延迟片上缓存角色
- warp 中线程访问更整齐
- 访存系统的无谓串行化被消除

从 roofline 角度看：

- kernel 仍然是低 arithmetic intensity
- 仍远未靠近 FP32 roof
- 但 memory side 的实际有效利用率明显提升
- 因此整体时间显著下降

需要注意的是，v3 的 `Issue Slots Busy` 和 `Executed IPC` 并没有比 v2 更高，甚至更低。这并不矛盾。

真正的解释是：

- v3 不是让 GPU 每个周期做了更多工作
- 而是让同样的 reduction pass 需要的总工作更少、无效 replay 更少
- 所以总耗时下降，即使“瞬时繁忙度”指标没有同步暴涨

---

## 4.5 v4: add during load，把第一层归约前移

### 代码变化

v4 的关键变化是：

```cpp
int block_num = (N + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK * 2);
float *blockstart = input + blockIdx.x * blockDim.x * 2;
sdata[threadIdx.x] = blockstart[threadIdx.x] + blockstart[threadIdx.x + blockDim.x];
```

含义：

- 每个 `thread` 不再只处理 1 个输入元素
- 每个 `thread` 读取 2 个元素，先加起来再写入 shared memory
- 每个 `block` 处理 `512` 个输入而不是 `256` 个
- `grid size` 直接减半

### 指标变化

相对 v3：

- `Grid Size: 131072 -> 65536`，减半
- `Threads: 33554432 -> 16777216`，减半
- `Duration: 3.18 -> 1.65 ms`，提升约 `48.1%`
- `SM Active Cycles: 1860584 -> 963812`，下降约 `48.2%`
- `Executed Instructions: 104.5M -> 56.5M`，下降约 `45.9%`
- `Branch Instructions: 10616832 -> 5308416`，减半
- `Memory Throughput: 57.93 -> 111.60 GB/s`，上升约 `92.6%`
- `Max Bandwidth` 基本不变：`76.71% -> 76.73%`
- `Occupancy` 基本不变：`90.30% -> 90.41%`

### 根本原因

v4 快的根本原因不是 occupancy 变化，也不是更高的时钟或更高的每周期发射率。

真正原因是：

**它把 reduction tree 的第一层融合到了 load 阶段，直接减少了后续需要处理的元素数、block 数、thread 数和动态指令数。**

这是一种“减少总工作量”的优化，而不是“提高单位工作利用率”的优化。

### 为什么 memory throughput 反而几乎翻倍

因为：

- 整体输入规模没有变
- 但处理这些输入所用时间大幅缩短
- 所以以 `GB/s` 计的吞吐自然会上升

换句话说，v4 不是靠“少读数据”变快，而是靠“同样数据更快处理完”变快。

### roofline 视角

v4 仍然不接近 FP32 roof：

- roofline 仍显示接近 `1%` 的 FP32 peak
- 说明算术强度依然不高

但 v4 的改进很符合 roofline 优化逻辑中的另一类策略：

- 不是把点往“更高的算力利用”推很多
- 而是减少完成任务所需的总 bytes 和总 instructions 的冗余成本
- 从而把 execution time 直接砍掉近一半

因此 v4 的优化本质更偏向：

- algorithm decomposition optimization
- kernel granularity optimization

而不只是局部访存微调。

---

## 4.6 v4 A vs v4 B: 256 threads/block 与 128 threads/block 的差异

这一组对比现在基于同一口径的 release 报告，可以直接把差异主要归因到 `block size` 和对应的 reduction 粒度变化上。

### 代码层面的实际差异

`v4 A` 与 `v4 B` 的算法完全相同：

- 都是 add-during-load
- 都是每个 `thread` 先处理 2 个元素
- 都使用 sequential addressing 的 shared-memory reduction

两者真正的实现差异只有：

1. `THREADS_PER_BLOCK`
  - `A = 256`
  - `B = 128`

2. 因为每个 thread 仍然处理 2 个元素，所以每个 block 处理的元素数不同
  - `A`: 每个 block 处理 `512` 个元素
  - `B`: 每个 block 处理 `256` 个元素

3. 因此 grid 大小不同
  - `A`: `65536`
  - `B`: `131072`

### 指标对比

| 指标 | v4 A | v4 B | 观察 |
|---|---:|---:|---|
| Block Size | 256 | 128 | B 每个 block 的线程数减半 |
| Grid Size | 65536 | 131072 | B 的 block 数翻倍 |
| Threads | 16777216 | 16777216 | 两者总线程数相同 |
| Duration | 1.65 ms | 1.50 ms | B 略快，约 `-9.1%` |
| SM Active Cycles | 963812 | 873493 | B 略低，约 `-9.4%` |
| Executed Instructions | 56.49M | 51.64M | B 略少，约 `-8.6%` |
| Memory Throughput | 111.60 GB/s | 120.66 GB/s | B 略高 |
| Max Bandwidth | 76.73% | 76.39% | 基本不变 |
| Issue Slots Busy | 36.59% | 36.89% | 基本持平，B 略高 |
| Eligible Warps / Scheduler | 0.60 | 0.54 | B 略低 |
| Achieved Occupancy | 90.41% | 89.28% | B 略低 |
| Branch Efficiency | 100% | 100% | 相同 |
| Avg. Divergent Branches | 0 | 0 | 相同 |

### 第一主因：B 的 reduction tree 更浅，单个 CTA 的指令工作量更少

这次 release 对比里，最稳定的变化是：`B` 的单个 block 线程数从 `256` 降到 `128`，因此 shared-memory reduction 的层数少一层。

- `A` 的规约层次大致是：`256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1`
- `B` 的规约层次大致是：`128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1`

也就是说，`B` 每个 CTA 少做一轮：

- 条件判断
- shared-memory 加法
- `__syncthreads()`

这一点和报告中的结果是对齐的：

- `Executed Instructions` 从 `56.49M` 降到 `51.64M`
- `Branch Instructions` 从 `5.31M` 降到 `4.85M`
- `SM Active Cycles` 从 `963812` 降到 `873493`

根本原因不是内存层次发生了结构性变化，而是**每个 CTA 完成自身 reduction 所需的控制流、同步和 shared-memory 操作更少了**。

### 第二主因：虽然 block 数翻倍，但总线程数不变，硬件饱和程度几乎没变

因为两者总线程数相同，但每个 block 的 thread 数不同：

- `A`: `65536` blocks，每个 block `256` threads
- `B`: `131072` blocks，每个 block `128` threads

这意味着：

1. `B` 的 block 数翻倍，但 `Threads` 总数仍然相同
2. `Waves Per SM` 两者都还是 `409.60`
3. `Max Bandwidth` 两者都在 `76%` 左右
4. `Issue Slots Busy` 也几乎一样

这说明：

- GPU 的总体并行规模没有变
- memory hierarchy 的总体利用形态也没有本质变化
- `B` 的收益主要来自“每个 CTA 更省指令”，而不是“整个芯片被喂得更多”

这也是为什么：

- `Memory Throughput` 从 `111.60` 升到 `120.66 GB/s`
- 但 `Max Bandwidth` 基本不变

也就是说，`B` 不是把 memory roof 再往上推了一截，而是在接近相同带宽利用率的前提下，用更少时间完成了同样规模的 pass。

### 为什么 B 的 memory throughput 更高而 max bandwidth 基本不变

在新的 release 对比里，真正的现象是：

- `Memory Throughput: 111.60 -> 120.66 GB/s`
- `Max Bandwidth: 76.73% -> 76.39%`

也就是说：

- 实际吞吐更高了
- 但相对峰值带宽利用率几乎没变

两者的 global access pattern 本身仍然是规则的：

- load 仍然是连续的
- add-during-load 仍然成立
- shared memory 访问模式也没有退化回 bank conflict 模式

真正的问题是：

- `B` 的执行时间更短
- 总处理数据规模不变
- 因而 `GB/s = bytes / time` 自然会略升

换句话说，`B` 并没有改变 memory subsystem 的上限位置，而是**在差不多相同的 bandwidth utilization 下，把同一轮 reduction pass 做得更快一些**。

这点可以从这些指标看出来：

- `Issue Slots Busy` 基本持平
- `Memory Throughput` 上升
- `Max Bandwidth` 基本持平
- `Executed Instructions` 下降

因此 roofline 角度下，`A` 和 `B` 仍然都属于低 arithmetic intensity、远离 FP32 roof 的同一类 kernel；`B` 只是沿着“减少总工作量”的方向又向前走了一小步。

### 关于 occupancy 和 eligible warps 为什么没有同步变好

`B` 并不是所有指标都比 `A` 更漂亮：

- `Achieved Occupancy` 略低：`90.41% -> 89.28%`
- `Eligible Warps / Scheduler` 略低：`0.60 -> 0.54`

这说明 `128-thread` block 并没有从调度层面带来明显红利。根本原因是：

- 这个 kernel 仍然有同步和依赖链
- 只是单 CTA 的工作量略减
- 但 warp scheduler 的整体画像并没有被根本改写

所以 `B` 的收益是“总指令数和总周期更少”，不是“occupancy 或调度效率显著更优”。

### 当前最可靠的结论

基于现有报告，`v4 A` 和 `v4 B` 的根本差异可以分成两层：

1. **主导因素：B 的 CTA 更小，shared-memory reduction 少一层**
  - 动态指令数下降
  - branch 指令下降
  - CTA 内同步和控制开销下降

2. **平衡因素：B 的 block 数翻倍，调度画像并没有显著改善**
  - occupancy 没有更高
  - eligible warps 没有更高
  - bandwidth utilization 也没有明显更高

所以，当前观测到的 `B` 比 `A` 略快，**最核心的原因不是 GPU 被更充分利用了，而是每个 CTA 的 reduction 树更浅，单 CTA 工作量略少。**

更严格的结论应该是：

**在当前 release 对比里，`128 threads/block` 的 `v4 B` 相比 `256 threads/block` 的 `v4 A` 有小幅优势；根本原因是更浅的 reduction tree 减少了 CTA 内的指令与同步开销，而不是 roofline 或 occupancy 层面发生了质变。**

### 如果要做严格的 A/B block-size 对比

建议后续补一个严格版本：

1. `A` 和 `B` 都保持相同的 release 编译选项
2. 都保留相同的错误检查和相同的数据规模
3. 只改变 `THREADS_PER_BLOCK`
4. 再比较：
  - `Duration`
  - `Executed Instructions`
  - `Eligible Warps / Scheduler`
  - `Warp Stall` 原因
  - `Memory Throughput`
  - `Max Bandwidth`

只有在这个前提下，才能把差异主要归因于 `128 vs 256 threads/block`。

---

## 5. 从 v0 到 v4 的总体规律

可以把整个优化过程概括为四类问题的依次清除：

### 第一阶段：去掉最昂贵的 global memory 中间态
- `v0 -> v1`
- 核心收益：把 block 内重复规约从 global memory 移到 shared memory

### 第二阶段：认识到 shared memory 也会出问题
- `v1 -> v2`
- 核心教训：控制流更整齐，不代表一定更快；bank conflict 可能更致命

### 第三阶段：让 shared memory 访问模式顺应硬件 bank 组织
- `v2 -> v3`
- 核心收益：sequential addressing 消除 bank conflict，带来最大一次结构性收益

### 第四阶段：从“访问模式优化”升级到“总工作量优化”
- `v3 -> v4`
- 核心收益：把第一层 reduction 融合到 load 阶段，直接让 grid 与动态工作量减半

---

## 6. 结合 roofline 的统一结论

所有版本共同的 roofline 结论是：

- 这类 reduction kernel 的 arithmetic intensity 很低
- achieved FP32 性能长期接近 0% 到 1%
- 因此优化主线不应理解成“逼近计算峰值”

更准确的理解是：

1. **先减少最昂贵的 off-chip memory traffic**
2. **再修正片上 shared memory 的访问模式**
3. **最后通过算法分解减少总工作量**

这三类优化分别对应：

- `v1`：数据放到更近的地方做
- `v3`：数据在 shared memory 里也要按 bank 友好的方式访问
- `v4`：让每个 block 完成更多有效归约，减少全局启动与中间结果规模

---

## 7. 对后续版本（v5+）的建议与报告模板

如果后续继续优化，建议优先沿这几个方向做，并沿用同样的分析框架：

### 可继续尝试的方向

1. warp-level reduction
   - 在最后 32 个线程以内改用 warp shuffle 或 unrolled warp reduction
   - 目标：减少 `__syncthreads()` 开销和 shared memory 往返

2. loop unrolling
   - 对固定 block size 的 reduction 手工展开
   - 目标：减少循环与索引控制开销

3. vectorized load / wider load
   - 如 `float2` / `float4`
   - 目标：提升 global load efficiency，但前提是地址对齐和访问模式正确

4. 多阶段完整 reduction 总时间评估
   - 目前分析的是单次 pass
   - 更完整的评价应该比较完成整个数组归约的总耗时

### 后续版本报告模板

| 版本 | 核心代码变化 | 期望改善的瓶颈 | 关键观察指标 | 若失败最可能的根因 |
|---|---|---|---|---|
| v5 | warp-level tail reduction | sync / shared traffic | Duration, SM cycles, Executed Inst, Warp Stall | warp 内实现不当，寄存器压力增加 |
| v6 | unroll last steps | control overhead | Issued Inst, Branch Inst, Issue Slots Busy | 指令缓存/寄存器压力，收益不明显 |
| v7 | vectorized load | global load efficiency | DRAM Throughput, sectors/request, Duration | 对齐不佳、访存模式破坏 coalescing |

后续每一版都建议固定看这几组指标：

- `Duration`
- `SM Active Cycles`
- `Executed / Issued Instructions`
- `Memory Throughput (GB/s)`
- `Max Bandwidth (%)`
- `Achieved Occupancy`
- `Branch Efficiency / Divergent Branches`
- shared memory / global memory 的 access warnings
- roofline 的 FP32 peak 占比变化

---

## 8. 最终结论

如果只用一句话概括这条优化路线：

**这不是一条“不断提高 FP32 算力利用率”的路线，而是一条“不断减少 reduction 在 GPU memory hierarchy 和 execution hierarchy 中的浪费”的路线。**

具体来说：

- `v1` 解决了 global memory 中间规约过于昂贵的问题
- `v2` 证明了 shared memory 访问模式错误会抵消控制流优化收益
- `v3` 通过 bank-friendly sequential addressing 获得了显著提升
- `v4` 则进一步从算法层减少总 block 数、总线程数和总指令数，带来接近翻倍的单 pass 提速

到 `v4` 为止，这个 reduction kernel 已经从“低效的 naive 实现”走到了“访存模式和工作量都更合理的优化版本”。

如果继续往下做，最值得追求的不是更高的 occupancy，而是：

- 更少的 synchronization
- 更少的 shared memory 往返
- 更少的动态指令
- 更少的中间 partial sums

这些才是这类低 arithmetic intensity reduction kernel 的真正性能杠杆。