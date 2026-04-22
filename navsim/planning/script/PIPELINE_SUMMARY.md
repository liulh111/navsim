# navsim → ScenarioMax → GPUDrive 三阶段 Waymo 观测抽取 Pipeline 总结

> 最后更新: 2026-04-21
> 目标: 为 navhard_two_stage 的每一个 token(含 stage1 真实帧与 stage2 合成帧), 得到与 GPUDrive 训练/评测完全对齐的 Waymo 2984 维观测 = `ego(6) + partner(63×6=378) + road(200×13=2600)`。三部分全部由 ScenarioMax + GPUDrive 产出, 不再掺入 navsim 自身的 ego / annotation 直读逻辑, 以彻底消除 navsim 端与 GPUDrive 训练端之间的潜在 drift。

---

## 目录
1. [背景与设计动机](#1-背景与设计动机)
2. [具体运行命令](#2-具体运行命令)
3. [代码关键函数逐一介绍](#3-代码关键函数逐一介绍)
4. [与「过滤 nuPlan → 整库转换」方案的详细对比](#4-与过滤-nuplan--整库转换方案的详细对比)
5. [输出文件与张量语义](#5-输出文件与张量语义)
6. [后续可扩展](#6-后续可扩展)

---

## 1. 背景与设计动机

navsim 的 `extract_waymo_feature_stage_one.py` 和 `extract_waymo_feature_stage_two.py` 需要对每一个 scene 产出 2984 维 feature:

```
ego (6) + partner (63×6=378) + road (200×13=2600) = 2984
```

**三块都必须和 GPUDrive 训练时看到的观测完全一致**, 否则训练/评测的 domain 会对不上。旧版 `extract_waymo_feature_stage_*.py` 里, 只有最后 `road` 的抽取方式与 GPUDrive 对齐较为精细, 而 `ego/partner` 是直接读 `scene.frames[idx].ego_status` 和 `annotations.boxes`, 再做一次 ego-local 归一化。一旦 navsim 侧的读取逻辑、单位或排序与 GPUDrive 发生任何漂移(例如 velocity frame, heading 约定, BoundingBoxIndex 调整等), 就会导致训练/评测 drift, 而且很难定位。

因此本 pipeline 的目标是: **把 ego / partner / road 三块统一交给 GPUDrive 去产出**, navsim 只负责准备一个符合 GPUDrive JSON 规格的场景文件, 然后调用 `GPUDriveTorchEnv` 读取其内置的 `self_observation_tensor()` / `partner_observations_tensor()` / `agent_roadmap_tensor()`。

要把场景塞给 GPUDrive, 必须先得到一个 GPUDrive JSON。把 nuPlan 资产(地图 + agent)转换为 GPUDrive JSON 的代码只存在于 ScenarioMax, 而 ScenarioMax 标准流程是从 `.db` 文件出发的, 这就带来了两个关键矛盾:

| 约束 | 含义 |
|---|---|
| **必须支持 stage2 合成帧** | 合成场景没有原始 `.db`, 所以不能走"过滤 .db + 整库转换"的标准流程 |
| **必须与 GPUDrive 逐像素对齐** | 不能自己重写地图抽取/简化逻辑, 也不能自己重算 ego/partner, 必须复用 ScenarioMax + GPUDrive 的官方函数 |

这两个约束迫使我们走出一条新路: **绕开 `.db` 文件, 直接用 navsim 已经加载好的 map_api + ego_pose + annotations 作为输入, 复用 ScenarioMax 的地图抽取/转换函数, 同时把自车和标注帧里的其它 agent 一起拼装成 GPUDrive JSON, 然后让 GPUDrive 去算三部分观测**。三个阶段分别跑在三个 conda 环境下, 用文件做中间态传递。

```
┌──────────────┐   scene_metadata.pkl        ┌──────────────────┐
│  Step 1      │   (ego_pose + ego_vel +     │  Step 2          │
│  navsim-llh  │───goal_local + partners)──> │  ScenarioMax     │
│  (scene API) │                             │  .venv           │
└──────────────┘                             └──────────────────┘
                                                      │
                                  tfrecord-<TOKEN>.json
                                  (ego + partners + roads in
                                   centered-global frame)
                                                      ▼
                                             ┌──────────────────┐
                                             │  Step 3          │
                                             │  gpudrive env    │
                                             │  (C++ simulator) │
                                             └──────────────────┘
                                                      │
        ego_obs.npy (6,)                              │
        partner_obs.npy (63, 6)  ◄────────────────────┘
        road_obs_gpudrive.npy (200, 13)
        waymo_obs_gpudrive.npy (2984,)   ← ego ⊕ partners.flatten ⊕ road.flatten
        waymo_obs_gpudrive.png           ← BEV 可视化(ego + 目标 + 它车 + 道路)
```

Step 1 默认只为 stage1 和 stage2 各导出**第一个可用 token** (快速自检); 传 `all` 参数可一次遍历两阶段所有 token。Step 2/3 的 shell 脚本都带 `--auto`, 会自动发现 `exp/pipeline_output` 下所有 `scene_metadata.pkl` / `tfrecord-*.json`, 按 token 一一处理。

---

## 2. 具体运行命令

三个阶段都有配套的 shell wrapper, 默认处理 stage1/stage2 各一个 token(快速自检):

### Step 1 — 导出 scene 元数据 (navsim-llh 环境)

```bash
conda activate navsim-llh
cd $NAVSIM_DEVKIT_ROOT

# 默认: stage1 第一个 token + stage2 第一个 token
bash scripts/evaluation/run_step1_export_metadata.sh

# 完整: stage1 所有 token + stage2 所有 token (跑完整评测时用)
bash scripts/evaluation/run_step1_export_metadata.sh all
```

"all" 参数会向 Hydra 追加 `+export_all=true`, 由 Python 侧的 `cfg.get("export_all", False)` 读取。

每个 token 输出: `/data/llh/navsim_workspace/exp/pipeline_output/<TOKEN>/scene_metadata.pkl`, 内容(所有矢量坐标均为 navsim 当前帧 ego-local):

```python
{"token":              "00016f8b45c25a1d",
 "stage":              "stage_one",                       # 或 "stage_two"
 "map_name":           "us-ma-boston",
 "ego_pose":           [331068.128, 4690724.234, 1.4974], # 全局 [x, y, heading(rad)]
 "ego_velocity_local": [6.12, 0.03],                      # ego-local [vx, vy]
 "ego_size":           [4.049, 1.852, 1.5],               # Pacifica length/width/height
 "goal_local":         [gx, gy],                          # stage2 fallback=(0,0)
 "partners_local": [
     {"rel_x": 8.4, "rel_y": -1.3, "heading": -0.02,
      "vel_x": 5.8, "vel_y": 0.0,
      "length": 4.8, "width": 2.1, "height": 1.7,
      "name": "vehicle"},
     ...                                                  # 按 annotation 顺序, 不排序
 ]}
```

### Step 2 — 生成 GPUDrive JSON (ScenarioMax .venv)

```bash
bash scripts/evaluation/run_step2_generate_gpudrive_json.sh
```

内部调用:
```bash
/data/llh/navsim_workspace/ScenarioMax/.venv/bin/python \
    /data/llh/navsim_workspace/navsim/navsim/planning/script/step2_generate_gpudrive_json.py --auto
```

`--auto` 会自动发现 `exp/pipeline_output/*/scene_metadata.pkl` 并全部处理。输出每个 token 一个 `gpudrive_json/tfrecord-<TOKEN>.json`。

典型日志:
```
INFO: Token:      00016f8b45c25a1d  (stage=stage_one)
INFO: Map:        us-ma-boston
INFO: Center:     [331068.13, 4690724.23]
INFO: Heading:    1.4974 rad
INFO: #partners:  18
INFO: Loading map from /data/llh/navsim_workspace/dataset/maps ...
INFO: Extracting static map elements via ScenarioMax ...
INFO:   Extracted 347 map elements
INFO: Converting to GPUDrive road format ...
INFO:   Road features: 347
INFO:   Simplification: 36148 pts -> 2751 pts (2404 segments)
INFO: Saved GPUDrive JSON: .../tfrecord-00016f8b45c25a1d.json
INFO:   Objects: 19 (ego + 18 partners)
```

### Step 3 — 抽取 2984 维 Waymo 观测 (gpudrive 环境)

```bash
conda activate gpudrive
bash scripts/evaluation/run_step3_extract_waymo_obs.sh
```

内部调用:
```bash
python /data/llh/navsim_workspace/navsim/navsim/planning/script/step3_extract_road_obs.py --auto
```

每个 token 输出:
- `ego_obs.npy`            — shape `(6,)`,    `float32`
- `partner_obs.npy`        — shape `(63, 6)`, `float32`
- `road_obs_gpudrive.npy`  — shape `(200, 13)`, `float32`
- `waymo_obs_gpudrive.npy` — shape `(2984,)`,  `float32` (ego ⊕ partners.flatten ⊕ road.flatten)
- `waymo_obs_gpudrive.png` — BEV 可视化(ego 框 + 目标点 + partner 框 + 道路点位), 由增强版 `visualize_road_obs` 绘制
- `road_obs_gpudrive.png`  — 同函数不传 ego/partners 时的"仅道路"版本, 便于单独核对地图

### 下游: WaymoMLPAgent 评测

```bash
bash scripts/evaluation/run_waymo_mlp_agent_pdm_score_evaluation.sh
```

该脚本调用 `run_pdm_score.py agent=waymo_mlp_agent`, 对应 `navsim/agents/waymo_mlp_agent.py`。它以 `scene.scene_metadata.scene_token` 为 key, 从 `exp/pipeline_output/<TOKEN>/waymo_obs_gpudrive.npy` 加载 2984 维观测, 过一个**随机初始化的 3 层 MLP** 产出 `(num_poses, 3)` 的 trajectory。其唯一作用是验证 pipeline 端到端联通(给 MLP 训练/对齐验证打底), 而不是一个实际可用的 planner。

---

## 3. 代码关键函数逐一介绍

### 3.1 `step1_export_scene_metadata.py`

核心是跨环境"拆解": navsim 环境里有 torch + navhard 数据, ScenarioMax 环境里又有 nuPlan devkit, 两套完全不能共存。所以这一阶段只负责把 Step 2/3 需要的所有字段**序列化进一个 pickle**, 让后续阶段在各自的环境里重新组装。

Sensor 是用不到的(feature extraction 不读 camera/lidar), 所以用 `SensorConfig.build_no_sensors()`, 加载 scene 时省掉所有 sensor I/O。

```python
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg):
    export_all = bool(cfg.get("export_all", False))        # ← 新增开关
    scene_loader = SceneLoader(..., sensor_config=SensorConfig.build_no_sensors())

    stage_plans = [
        ("stage_one", sorted(set(scene_loader.tokens_stage_one) & metric_tokens)),
        ("stage_two", sorted(set(scene_loader.reactive_tokens_stage_two) & metric_tokens)),
    ]
    for stage, tokens in stage_plans:
        selected = tokens if export_all else tokens[:1]    # 默认各取第一个
        for token in selected:
            scene = scene_loader.get_scene_from_token(token)
            _export_one_token(scene, token, stage)
```

`_export_one_token` 从"当前帧" `scene.frames[num_history_frames - 1]` 里抽取 ego / partners / goal:

```python
frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
ego_pose           = frame.ego_status.ego_pose          # 全局 [x, y, heading]
ego_velocity_local = frame.ego_status.ego_velocity      # ego-local [vx, vy]
ego_size           = [pacifica.length, pacifica.width, 1.5]

# goal: stage1 用未来轨迹终点, stage2 没有未来帧, 兜底 (0, 0)
goal_local = [trajectory.poses[-1, 0], trajectory.poses[-1, 1]]  # stage_one
             if stage == "stage_one" else [0.0, 0.0]

# partners: annotation 顺序, 不按距离排序 (对齐标准 nuPlan→GPUDrive 流)
for i in range(annotations.boxes.shape[0]):
    partners_local.append({
        "rel_x":   boxes[i, BoundingBoxIndex.X],
        "rel_y":   boxes[i, BoundingBoxIndex.Y],
        "heading": wrap_to_pi(boxes[i, BoundingBoxIndex.HEADING]),
        "vel_x":   velocity_3d[i, 0],                   # 注 ① ego-local
        "vel_y":   velocity_3d[i, 1],
        "length":  boxes[i, BoundingBoxIndex.LENGTH],
        "width":   boxes[i, BoundingBoxIndex.WIDTH],
        "height":  boxes[i, BoundingBoxIndex.HEIGHT],
        "name":    annotations.names[i],
    })
```

关键点:

- `ego_pose` 是**全局 nuPlan 坐标系下的三元组** `[x, y, heading]`, Boston map 大约在 `(331068, 4690724)` 一带
- `num_history_frames - 1` 取的是"现在这一帧", 与 `extract_waymo_feature.py` 里 `ego_state` 的取法一致
- 只要 token 能被 SceneLoader 加载(stage1 或 stage2 均可), 这一步就能产出 metadata。stage2 合成帧也有 `scene.map_api`、`ego_pose` 和 `annotations`, **不依赖 `.db` 文件**
- ① **`velocity_3d` 是 ego-local 帧**(见 `navsim_scenario_utils.py:81` 里把它再次 `rotate_vector(v, ego_heading)` 还原到全局以构造 nuPlan `Agent`), 所以和 `boxes` 的 `rel_x/rel_y/heading` 保持同一 ego-local 参考系
- partners **不排序**; 顺序就是 `annotations.boxes` 的自然顺序, 对齐 `scenariomax/unified_to_gpudrive/convert_to_json.py:60` 里"dict 迭代顺序进 `objects` 列表"的标准约定
- stage2 的 `scene.get_future_trajectory()` 会在没有未来帧时 raise, 这里显式 `try/except` 回落到 `(0, 0)`, 与 `extract_waymo_feature_stage_two.py:113-122` 的既有兜底保持一致

### 3.2 `step2_generate_gpudrive_json.py`

核心三步: 重建 map_api → ScenarioMax 抽取 → 构造 minimal JSON。

#### 3.2.1 `get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_name)`

为什么要**重新**加载 map_api? 因为 `scene.map_api` 是一个 `NuPlanMap` 实例, 里面持有 GPKG 文件句柄、Shapely 几何对象等, 完全无法 pickle 跨环境传递。而 `get_maps_api` 在 ScenarioMax 环境里用的是**同一份 GPKG 文件** (`/data/llh/navsim_workspace/dataset/maps/nuplan-maps-v1.0/*.gpkg`), 所以重新加载得到的 map_api **和 navsim 里的那一个在数据层面是完全一致的**。

#### 3.2.2 `extract_static_map_elements(map_api, center)` — ScenarioMax 官方函数

位于 `scenariomax/raw_to_unified/datasets/nuplan/extractor.py`。行为:

1. 从 nuPlan map_api 抓 7 个 semantic layers(以 `center` 为圆心、默认 **250 m** 半径):
   `LANE`, `LANE_CONNECTOR`, `ROADBLOCK`, `ROADBLOCK_CONNECTOR`, `STOP_LINE`, `CROSSWALK`, `INTERSECTION`
2. 对每个 lane 提取 centerline、polygon、左右邻居、speed_limit、entry/exit lanes
3. 对每条道路边界用 GeoPandas `unary_union()` 合并所有 roadblock + intersection polygon, 再解构出 boundary line(这步在 Boston 上特别慢, 是整个 pipeline 的主要耗时)
4. 所有几何坐标**已经相对 `center` 做过平移**(即 ego-centric 坐标)

返回一个 dict, 形如:
```python
{lane_id: {"type": "LANE_SURFACE_STREET", "polyline": [...], "polygon": [...],
           "speed_limit_mph": ..., "entry_lanes": [...], ...},
 boundary_id: {"type": "ROAD_EDGE_BOUNDARY", "polyline": [...]},
 ...}
```

本 token 抽出了 347 个 map elements。

#### 3.2.3 `convert_map_features(static_map_elements)` — ScenarioMax 官方函数

位于 `scenariomax/unified_to_gpudrive/converter/roadgraph.py`。行为:

1. 按 `TYPE_MAPPING` 把 scenario_net 的字符串类型映射成 Waymax 整数 ID:
   ```python
   LANE_FREEWAY          → 1
   LANE_SURFACE_STREET   → 2
   LANE_BIKE_LANE        → 3
   ROAD_LINE_*           → 6..13
   ROAD_EDGE_BOUNDARY    → 15
   CROSSWALK             → 18
   SPEED_BUMP            → 19
   STOP_SIGN             → 17
   ```
2. 把 `polyline` / `polygon` 转成 `[{"x", "y", "z"}, ...]` 列表(补 `z=0.0`)
3. **3D 结构检测**: 如果在 xy 平面上 0.2 m 以内存在 z 值不同的点, 认为该场景有立体交叉, 直接返回 `(None, None)` 把整个场景丢弃(nuPlan + Boston 经常命中)
4. 过滤掉 `ROAD_EDGE_SIDEWALK` 和 `DRIVEWAY`
5. 额外构造 `edge_segments`: 把每条 road edge 拆成 `[[x1,y1,z1],[x2,y2,z2]]` 二元组, 供 GPUDrive 碰撞检测用

返回:
```python
road_features = [
    {"geometry":[{"x","y","z"},...], "type":"lane", "map_element_id":2, "id":123},
    {"geometry":[...], "type":"road_edge", "map_element_id":15, "id":124},
    ...
]
edge_segments = [[[x1,y1,z1],[x2,y2,z2]], ...]
```

#### 3.2.4 `simplify_geometry(points, threshold)` — Python 移植的 C++ 算法

这是整个 pipeline 里最需要解释清楚的一步。

**为什么需要它?** GPUDrive 的 C++ 头文件 `consts.hpp` 定义了:
```cpp
constexpr int32_t kMaxRoadEntityCount = 10000;  // 整个场景的"路段"总数上限
constexpr int32_t kMaxAgentMapObservationsCount = 200;  // 每辆车看到的道路点数
```
这里的"路段"(segment)指的是 `相邻两个几何点连成的一小段`。Boston token 原始数据是 **36,148 个几何点 → 35,801 个路段**, 远超 10,000 上限, 直接塞进 GPUDrive 会 SIGABRT。

**GPUDrive C++ 怎么简化的?** 看 `src/json_serialization.hpp` line 142-204, 摘录关键逻辑:

```cpp
if (num_segments >= 10 &&
    (road.type == RoadLane || RoadEdge || RoadLine))
{
    std::vector<bool> skip(N, false);
    while (skipChanged) {
        skipChanged = false;
        int64_t k = 0;
        while (k < N - 1) {
            // 找下一个未被 skip 的点 k_1
            int64_t k_1 = k + 1;
            while (k_1 < N - 1 && skip[k_1]) k_1++;
            // 再找下下个未被 skip 的点 k_2
            int64_t k_2 = k_1 + 1;
            while (k_2 < N && skip[k_2]) k_2++;

            // 三角形 (p[k], p[k_1], p[k_2]) 的面积
            float area = 0.5 * std::abs(
                (p1.x - p3.x) * (p2.y - p1.y)
              - (p1.x - p2.x) * (p3.y - p1.y));

            if (area < polylineReductionThreshold) {
                skip[k_1] = true;     // 中点可以丢
                k = k_2;              // 跳过中点, 从 k_2 继续
                skipChanged = true;
            } else {
                k = k_1;              // 保留中点, 继续往前走
            }
        }
    }
    // 强制保留首尾
    skip[0] = false; skip[N-1] = false;
    // 重建几何
    new_geometry = [p for p in points if not skip[idx(p)]];
}
```

**几何直觉**:
- 三个连续点 `p1, p2, p3` 如果近似共线, 那么它们构成的三角形面积就很小 → `p2` 基本在 `p1→p3` 这条直线上 → 丢掉 `p2` 对道路形状几乎没有影响
- 面积阈值 `polyline_reduction_threshold` 越大, 简化越激进。GPUDrive 的默认训练/评测配置 (`ppo_base_puffer.yaml`、`eval_config.yaml` 等) **统一用 `0.1`**, 所以我们也用 `0.1`。
- "迭代" (outer `while(skipChanged)`) 的意义: 一轮扫描后, 原本隔着 `p2` 的 `p1,p3` 现在挨在一起, 它们和外面的 `p4` 又可能组成新的三角形——可能面积更小, 于是第二轮又能丢掉更多点。迭代到完全稳定为止。
- **首尾点永远保留** (C++ 第 186-187 行, 也就是 `skip[0] = false; skip[N-1] = false`)。
- 只对 `RoadLane / RoadEdge / RoadLine` 生效, 人行横道、减速带、停车标志不简化(因为它们本来就是小的几何体, 不会有太多点)。
- 只在 `num_segments >= 10` 才触发——短道路不动。

**Python 端移植版** (`step2_generate_gpudrive_json.py`):
```python
def simplify_geometry(points, threshold):
    if len(points) < 10:
        return points
    skip = [False] * len(points)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(points):
            if skip[i]: i += 1; continue
            j = i + 1
            while j < len(points) and skip[j]: j += 1
            if j >= len(points): break
            k = j + 1
            while k < len(points) and skip[k]: k += 1
            if k >= len(points): break
            p1, p2, p3 = points[i], points[j], points[k]
            area = 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"])
                           - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))
            if area < threshold:
                skip[j] = True
                changed = True
            i = j
    return [p for p, s in zip(points, skip) if not s]
```

这是对 C++ 版本的**逐行复刻**, 包括:
- 三角形面积用同一个公式(而不是用 Shapely)
- 同样的迭代终止条件(`skipChanged`)
- 同样的"找下一个未被 skip 的点"推进方式
- 同样的阈值 `0.1`

本 token 实测结果: `36148 点 → 2751 点 (2404 段)`, 一次性过了 `kMaxRoadEntityCount=10000` 的闸。

**为什么要在 Python 端做 (而不是交给 GPUDrive C++ 做)?** 原则上 C++ 端也会简化一次, 但:
1. C++ 里如果整个场景已经超 `kMaxRoadEntityCount` **在读取阶段就会 abort**, 根本到不了简化那一步
2. 提前在 Python 里简化, 既能控制整体段数, 也能和 C++ 结果完全对齐(同一个算法、同一个阈值)

> 兜底逻辑: 如果简化后还是超 10000, `step2` 会按 `x²+y²` 到 ego 的平均距离把最远的道路整条剪掉, 直到段数达标。Boston token 上没触发这个兜底。

#### 3.2.5 `build_gpudrive_json_full(metadata, road_features)` — 真实 ego + 真实 partners

当前版本把 ego 和所有 partner **一起**写进 JSON, 以便 Step 3 直接从 GPUDrive 的 `self_observation_tensor()` / `partner_observations_tensor()` 读出 6 维 ego 和 63×6 partner 观测, 而不再走 navsim 的直读逻辑。

所有坐标都会被转写到**以 `center = ego_pose[:2]` 为原点的全局平移系**:

- **`extract_static_map_elements(map_api, center)` 内部就是以 `center` 为圆心做 250 m 裁剪 + 坐标平移**, 返回的 `static_map_elements` 已经是"center-平移系"的坐标
- 因此 ego 恰好在 `(0, 0)`; partner 由"ego-local"经 `R(ego_heading)` 旋转到"center-平移系"(保持全局朝向)

```python
def _rotate(vec_xy, heading):
    c, s = cos(heading), sin(heading)
    return c * vec_xy[0] - s * vec_xy[1], s * vec_xy[0] + c * vec_xy[1]

ego = {
    "position": [{"x":0,"y":0,"z":0}] * 91,               # ego 在 center, 就是 (0,0)
    "heading":  [ego_heading] * 91,
    "velocity": [{"x": vx_c, "y": vy_c}] * 91,            # R(h) @ ego_velocity_local
    "valid":    [True] * 91,
    "goalPosition": {"x": gx_c, "y": gy_c, "z": 0.0},     # R(h) @ goal_local
    "width": 1.852, "length": 4.049, "height": 1.5,
    "type": "vehicle", "id": 0, "mark_as_expert": False,
}

for i, p in enumerate(partners_local[:63]):               # 不排序, 仅截 63
    px_c, py_c = _rotate((p["rel_x"], p["rel_y"]), ego_heading)
    vx_c, vy_c = _rotate((p["vel_x"], p["vel_y"]), ego_heading)
    head_c = p["heading"] + ego_heading
    objects.append({
        "position": [{"x":px_c,"y":py_c,"z":0.0}] * 91,
        "heading":  [head_c] * 91,
        "velocity": [{"x":vx_c,"y":vy_c}] * 91,
        "valid":    [True] * 91,
        "goalPosition": {"x":px_c,"y":py_c,"z":0.0},      # 无 goal, 设在自身位置
        "width": p["width"], "length": p["length"], "height": p["height"] or 1.5,
        "type": "vehicle", "id": i+1, "mark_as_expert": False,
    })

json = {
    "name": f"tfrecord-{token}.json",
    "scenario_id": token,
    "objects": [ego] + partners,
    "roads":    road_features,
    "tl_states": {},
    "metadata": {
        "sdc_track_index":     0,        # ego 放在 objects[0]
        "objects_of_interest": [],
        "tracks_to_predict":   [],       # ← 对齐 nuPlan 标准流, 不指定 ego 为 track
    },
}
```

为什么要把 91 帧都填上相同的值? 因为 GPUDrive C++ 加载端在 `level_gen.cpp:62` 直接按 `agentInit.velocity[curStepIdx]` 读取**每一步**的轨迹, `episodeLen=91` 决定了数组长度; 只要 valid 标记是 `True` 并且所有帧的值一致, agent 在 step 0 的观测就与 navsim "当前帧"一致。

踩过的坑:

| 坑 | 症状 | 原因 |
|---|---|---|
| 用 1 帧 position | SIGABRT in C++ | `episodeLen=91`, C++ 会按下标 0..90 读 |
| `difficulty: "EASY"` | SIGABRT | C++ 侧类型是整数, 字符串会 crash |
| 多写 `is_sdc`, `total_distance_traveled` | SIGABRT | `from_json` 会断言字段集, 多余字段也会触发 |
| 按距离给 partner 排序 | 与标准 nuPlan 流漂移 | 标准流程是 dict 迭代顺序, 改排序会改 partner_obs 的 row 含义 |
| 把 ego 放 `objects[N]` 却把 `sdc_track_index=0` | ego 观测乱 | C++ 用 `sdc_track_index` 来选 SDC, 要与 ego 在 `objects` 里的下标严格一致 |
| partner 超过 63 个 | C++ abort (`kMaxAgentCount=64`) | 代码里硬截断 63 个, 超出的按 annotation 顺序丢弃 |

### 3.3 `step3_extract_road_obs.py`

现在一次输出 4 个 `.npy` + 1 张 BEV 图, 覆盖整个 2984 维观测:

| 文件 | shape | 含义 |
|---|---|---|
| `ego_obs.npy` | `(6,)` | `[speed, length, width, rel_goal_x, rel_goal_y, is_collided]` |
| `partner_obs.npy` | `(63, 6)` | 每行 `[speed, rel_x, rel_y, orientation, length, width]` |
| `road_obs_gpudrive.npy` | `(200, 13)` | `[x, y, seg_len, seg_w, seg_h, orient, type_one_hot(7)]` |
| `waymo_obs_gpudrive.npy` | `(2984,)` | `ego ⊕ partner.flatten() ⊕ road.flatten()` |
| `waymo_obs_gpudrive.png` | — | 增强版 `visualize_road_obs` 同时画 ego/goal/partner/road |

#### 3.3.1 jaxlib 补丁

gpudrive conda 环境里没装 jaxlib, 但 GPUDrive 的某些 import 链会 `isinstance(x, jaxlib.xla_extension.ArrayImpl)`。直接 `unittest.mock.MagicMock()` 会让 `ArrayImpl` 变成 Mock 对象, 在 `isinstance()` 里报 `TypeError: isinstance() arg 2 must be a type`。必须创造**真的模块 + 真的空类**:

```python
_fake_mod = type(sys)("jaxlib.xla_extension")  # ModuleType
class _FakeArrayImpl: pass                       # 真类
_fake_mod.ArrayImpl = _FakeArrayImpl
sys.modules["jaxlib.xla_extension"] = _fake_mod
```

#### 3.3.2 `GPUDriveTorchEnv` 初始化

```python
env_config = EnvConfig(
    ego_state=True,
    road_map_obs=True,
    partner_obs=True,
    norm_obs=False,          # 反归一化, 方便与 navsim 端直接对齐
    init_mode="all_valid",   # ← partners 也算 valid agent, 被 partner_observations_tensor 看见
)
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=SceneDataLoader(root=args.json_dir, batch_size=1, dataset_size=1,
                                sample_with_replacement=False, file_prefix="tfrecord"),
    max_cont_agents=64,       # ego + 63 partners
    device="cpu",
    render_config=RenderConfig(),
)
```

关键 config 差异(相对旧版"仅道路"pipeline):

- `max_cont_agents=1` → `64`: 允许 C++ 把 63 个 partner 作为 "agent" 加载进来, 它们的 pose/velocity 会被 partner tensor 暴露
- `init_mode="all_non_trivial"` (默认) → `"all_valid"`: 只要 JSON 里 `valid[i]=True`, 对应 agent 就会被初始化, 不再要求它"非平凡"(即使速度为 0 的静态 agent 也会被看见)

这一步 GPUDrive 的 C++ 做了几件事:
1. 读 JSON → 反序列化 `MapRoad` + 每个 `Object` 的 `Trajectory` (位置/朝向/速度的 91 帧数组)
2. **polyline 再简化一次**(同一个算法同一个阈值, 但我们 Python 端已经简化过, 基本没活干)
3. 计算所有几何点的 `mean`, 把坐标去中心化 → 存在 `road.mean` 里(查询时再加回去)
4. 每辆 agent 通过 KNN 空间索引取离自己最近的 200 个路段点 → `agent_roadmap_tensor`
5. 每辆 agent 的 ego-local `self_observation` (goal/速度/碰撞位) 以及 63 个 partner 的 ego-local 观测 → `self_observation_tensor` / `partner_observations_tensor`
6. 所有 ego-local 量都是基于每个 agent 自己的 pose 做 `减 ego 位置、旋转 -heading`, 因此 SDC 观测到的 partner 位置就是 navsim `annotations` 里的 `rel_x/rel_y`(在纯几何意义上一致)

#### 3.3.3 三个抽取函数

三个函数分别对应 `env_torch._get_{ego_state, partner_obs, road_map_obs}(mask=None)` 的代码路径, 区别只是**不归一化、不 flatten、只取 world 0 的 ego agent**。

**(a) `extract_ego_obs(env) → (6,)`** — 和 `LocalEgoState` 的 6 列一一对应:
```python
ego_state = LocalEgoState.from_tensor(
    self_obs_tensor=env.sim.self_observation_tensor(),
    backend="torch", device="cpu",
)
# 和 env_torch.py:776-786 一致
stacked = torch.stack([ego_state.speed, ego_state.vehicle_length, ego_state.vehicle_width,
                       ego_state.rel_goal_x, ego_state.rel_goal_y, ego_state.is_collided],
                      dim=-1)                 # (num_worlds, max_agents, 6)
ego_obs = stacked[0, 0].cpu().numpy()         # (6,)
```

**(b) `extract_partner_obs(env) → (63, 6)`** — `PartnerObs` 每行 6 列, 排列顺序与 `env_torch.py:834-844` 完全一致:
```python
partner_obs = PartnerObs.from_tensor(env.sim.partner_observations_tensor(), ...)
stacked = torch.cat([partner_obs.speed, partner_obs.rel_pos_x, partner_obs.rel_pos_y,
                     partner_obs.orientation, partner_obs.vehicle_length, partner_obs.vehicle_width],
                    dim=-1)                   # (num_worlds, max_agents, 63, 6)
partner_obs_np = stacked[0, 0].cpu().numpy()  # (63, 6)
```
**不按距离排序** — partner 的行顺序完全由 JSON 里 `objects` 列表的顺序决定, 即 navsim `annotations` 的自然顺序。这和标准 nuPlan → ScenarioMax → GPUDrive 流"dict 迭代顺序"保持一致。如果下游训练/评测希望按距离排序, 那是下游的事, 不应在这里动手脚。

**(c) `extract_road_obs(env) → (200, 13)`** — 与旧版完全一致:
```python
roadgraph = LocalRoadGraphPoints.from_tensor(
    local_roadgraph_tensor=env.sim.agent_roadmap_tensor(),
    backend="torch", device="cpu",
)
roadgraph.one_hot_encode_road_point_types()   # type: int → 7-dim one-hot
road_obs = torch.cat([roadgraph.x.unsqueeze(-1), roadgraph.y.unsqueeze(-1),
                      roadgraph.segment_length.unsqueeze(-1),
                      roadgraph.segment_width.unsqueeze(-1),
                      roadgraph.segment_height.unsqueeze(-1),
                      roadgraph.orientation.unsqueeze(-1),
                      roadgraph.type.float()], dim=-1)  # (world, agent, 200, 13)
ego_road = road_obs[0, 0].numpy()             # (200, 13)
```

最后拼接:
```python
waymo_obs = np.concatenate([ego_obs, partner_obs_np.flatten(), ego_road.flatten()]).astype(np.float32)
assert waymo_obs.shape == (2984,)
```

13 列对应的语义:

| 列索引 | 含义 |
|---|---|
| 0 | `x` (ego-local, m) |
| 1 | `y` (ego-local, m) |
| 2 | `segment_length` (m) |
| 3 | `segment_width` (m) |
| 4 | `segment_height` (m) |
| 5 | `orientation` (rad, ego-local) |
| 6..12 | type one-hot, 7 类 |

**7 类对应**(GPUDrive `EntityType` 枚举, 与 C++ 源码顺序一致):
```
0: None        1: RoadEdge    2: RoadLine    3: RoadLane
4: CrossWalk   5: SpeedBump   6: StopSign
```

> ⚠️ 这个顺序和 waymo.md 里 "RoadLine=1, RoadEdge=2" 的顺序**不一样**, 是 GPUDrive 自己的 `EntityType` enum 顺序。如果用户的下游特征必须严格照 waymo.md 排, 那在 one-hot 之前需要做一次索引置换。

#### 3.3.4 `visualize_road_obs(roads, save_path, ego=None, partners=None)` — 一函数画全

该函数同时服务于"完整 2984 维观测可视化"和"仅道路调试可视化", 差别只是是否传入 `ego` / `partners`:

- `ego` 传入时: 在 `(0, 0)` 画一个 ego 大小的方框(`is_collided` 为真时染红)+ 黄色星号 + 虚线表示 goal
- `partners` 传入时: 对每个非零行(按距离意义自动过滤全零 pad), 画一个 cyan 方框 + 方向线段
- 道路部分对 one-hot 取 `argmax` 得到类名, 每类一个颜色, 画散点(点位)+ 短线(按 `orientation` 画 `segment_length` 长度的小段)
- 显示范围 `[-80, 80]×[-80, 80]` m

本 token(stage1 Boston)实测:
- 道路: 200 行中约 200 行非零(KNN 取满); 分布 `RoadLane=166, RoadEdge=31, CrossWalk=3`
- partners: 18 行非零(= annotations 个数, 其余 45 行为 pad 零)
- 形状与 BEV 直观对得上: 一个典型的十字路口 + 两条主干道 + ego 前方若干车辆

---

## 4. 与「过滤 nuPlan → 整库转换」方案的详细对比

用户提到另一种思路是"先根据 navtest tokens 过滤 nuPlan `.db`, 再对筛出来的 nuPlan 场景跑标准 ScenarioMax 转换"。下面做逐层对比。

### 4.1 数据来源

| 维度 | 本 pipeline (三阶段) | 标准 ScenarioMax 流 |
|---|---|---|
| 入口数据 | `navsim Scene` (pickle 化的 navtest metadata) | nuPlan `.db` 文件 (scenario filter 过滤出来) |
| 触达方式 | navsim `SceneLoader.get_scene_from_token(token)` | `NuPlanScenarioBuilder` + `ScenarioFilter` |
| 地图文件 | `/data/llh/navsim_workspace/dataset/maps/*.gpkg` | 同一份 GPKG |
| Ego pose | `scene.frames[num_history_frames-1].ego_status.ego_pose` | nuPlan scenario 的 `initial_ego_state.rear_axle` |
| stage1 真实帧 | ✅ 支持 | ✅ 支持 |
| **stage2 合成帧** | ✅ 支持 | ❌ 合成帧没有对应 `.db`, 流程断在第一步 |

### 4.2 地图抽取函数调用

**两者调用的是完全相同的两个 ScenarioMax 函数**:

```python
static_map_elements  = extract_static_map_elements(map_api, center)
road_features, edges = convert_map_features(static_map_elements)
```

具体到参数:

| 参数 | 本 pipeline | 标准 ScenarioMax |
|---|---|---|
| `map_api` | `get_maps_api(NUPLAN_MAPS_ROOT, v1.0, map_name)` | `scenario.map_api` (本质也是 `NuPlanMap` 实例) |
| `center` | `ego_pose[:2]` (navsim 当前帧) | `scenario.initial_ego_state.rear_axle.point` (nuPlan 场景起始帧) |
| 半径 | 默认 250 m (函数内部固定, 无参数可调) | 默认 250 m |
| map layers | LANE, LANE_CONNECTOR, ROADBLOCK, ROADBLOCK_CONNECTOR, STOP_LINE, CROSSWALK, INTERSECTION | 完全一样 |

两者的 **`map_api` 底层 GPKG 数据完全相同**, `center` 是两者唯一可能有 cm 级偏差的地方(因为"navsim 当前帧" vs "nuPlan scenario 起始帧"是否为同一帧, 取决于 scene 构造逻辑), 但既然都在 250 m 半径内抽取, 这点偏差几乎不改变被抓到的 map element 集合。

### 4.3 类型映射与几何转换

`convert_map_features` 的几个关键规则(**两条 pipeline 完全共用**):

1. 字符串类型 → 整数 `map_element_id` (Waymax ID):
   ```
   LANE_SURFACE_STREET      →  2
   ROAD_LINE_BROKEN_SINGLE_WHITE → 6
   ROAD_LINE_SOLID_SINGLE_WHITE  → 7
   ROAD_EDGE_BOUNDARY       → 15
   CROSSWALK                → 18
   SPEED_BUMP               → 19
   STOP_SIGN                → 17
   ```
2. 2D → 3D: xy 坐标后补 `z=0.0`
3. **3D 结构检测**: xy 平面 0.2m 内存在不同 z, 整场景丢弃(返回 `(None, None)`)
4. 过滤 `ROAD_EDGE_SIDEWALK` 和 `DRIVEWAY`

这一层**输出的 `road_features` 在两条 pipeline 里逐字段相同**。

### 4.4 Polyline 简化

标准 ScenarioMax 流**不在 Python 端做简化**, 直接把完整的 `road_features` 写进 JSON, 让 GPUDrive C++ 读 JSON 时跑 `from_json(MapRoad, polylineReductionThreshold=0.1)` 做简化。

本 pipeline 在 Python 端预先跑一次**完全相同的算法**(同一个面积公式、同一个阈值 `0.1`、同一个迭代逻辑、同一个 `num_segments >= 10` 的触发条件), 是因为 C++ 端如果场景在读取阶段已经超过 `kMaxRoadEntityCount=10000` 就会直接 abort, 根本到不了简化那一步。

**结论**: 两者最终送进 GPUDrive 的几何点**在算法意义上是一致的**, 但顺序上:
- 标准流程: C++ 读 → C++ 简化 → 建 KNN
- 本 pipeline: Python 简化 → C++ 读(基本无活) → C++ 建 KNN

所以到 KNN 以及之后的所有步骤, 两者产出**逐 bit 等价**。

### 4.5 送入 GPUDrive 的 JSON 差异

这是两条 pipeline 唯一实质性不同的地方——**非地图部分**(本 pipeline 当前版本已经把 ego+partner 全部写进 JSON, 与标准流非常接近了):

| JSON 字段 | 本 pipeline | 标准 ScenarioMax |
|---|---|---|
| `objects[0].position` | `[(0,0,0)] × 91` (ego 在 center) | `[(x_i - center_x, y_i - center_y, 0)]` 真实 91 帧 |
| `objects[0].heading`  | `[ego_heading] × 91` (当前帧 heading) | 真实 heading 轨迹 |
| `objects[0].velocity` | `[R(h) @ ego_velocity_local] × 91` (当前帧速度重复 91 次) | 真实速度 91 帧 |
| `objects[0].goalPosition` | `R(h) @ goal_local` (stage1 未来轨迹终点 / stage2 → `(0,0)`) | 通常是真实 scenario 的 goal |
| `objects[1..N]` | N = `annotations` 个数(最多 63), 用自然顺序; 每个 agent 91 帧位置/速度/朝向都复制当前帧 | 所有轨迹对象, 每帧都是真实值 |
| `tl_states` | `{}` | 真实红绿灯 91 帧 |
| `roads` | **完全相同** | **完全相同** |
| `metadata.sdc_track_index` | `0` (ego 放 objects[0]) | `0` 或 ego 所在的 index |
| `metadata.tracks_to_predict` | `[]` ← 对齐 nuPlan 标准流 | `[]` (nuPlan 流里确实是空) |

换句话说, 本 pipeline 的 JSON 是**"把 navsim 当前帧静态化"**(同一帧复制 91 次), 而标准 pipeline 的 JSON 是"真实 91 帧序列"。这对 **step 0 的 ego/partner 观测**没有影响(因为观测只看 `curStepIdx=0`), 只会在运行 episode simulation 的时候体现——本 pipeline 跑到 step 1 时所有 agent 会"原地站着"。但我们只需要 step 0 的观测, 所以这是可接受的。

### 4.6 最终 2984 维观测的一致性

观测由三块组成: `ego(6) ⊕ partner(63×6) ⊕ road(200×13)`。逐块分析:

**road(200×13)**: 列语义 `[x, y, seg_len, seg_w, seg_h, orient, type(7)]`:
- `seg_len, seg_w, seg_h, type` 只和道路几何有关 → 两者完全相同
- `x, y, orient` 是 ego-local 坐标, 取决于 ego 的 `(x, y, heading)`。只要"navsim 当前帧 ego pose" == "nuplan scenario 起始帧 ego pose"(对 navtest 过滤出来的 stage1 场景成立), 两条 pipeline 的 road 块**逐 bit 等价**

**ego(6)**: `[speed, length, width, rel_goal_x, rel_goal_y, is_collided]`:
- `speed`: 都来自 `velocity.linear.length()`; 本 pipeline 用 `R(h)@ego_velocity_local` 写进去, 标准流直接写真实全局速度。`speed = |v|` 对旋转不敏感 → 两者相同
- `length, width`: Pacifica `(4.049, 1.852)` vs nuPlan scenario 里 ego 真实尺寸, 二者一致
- `rel_goal_x, rel_goal_y`: 都是"goal 的 ego-local 坐标"。本 pipeline 把 `goal_local` 旋到 center-平移系, C++ 再旋回 ego-local, 净效应 = navsim `trajectory.poses[-1, :2]` 原值。标准流是 scenario goal 的 ego-local, 对同一个 token **如果 scene 的 "当前帧 = scenario 起始帧", 且二者对 goal 定义一致, 就相同**
- `is_collided`: step 0 都是 `False = 0.0`

**partner(63×6)**: 每行 `[speed, rel_x, rel_y, orient, length, width]`:
- `speed`: 同理与 rotation 无关, 两者相同
- `rel_x, rel_y`: 都是 partner 的"ego-local"坐标。navsim 的 `annotations.boxes[:, X/Y]` 已经是 ego-local, 本 pipeline 把它旋到 center-平移系再交给 C++, C++ 再算回 ego-local, 净效应 = 原值。标准流直接提供 global → C++ 算 ego-local, 也得到原值 → **两者相同**
- `orient`: navsim `annotations.boxes[:, HEADING]` 也是 ego-local, 净效应同上 → **相同**
- `length, width`: 直接来自 `annotations.boxes` vs nuPlan agent 尺寸, 通常由同一份数据标注 → 相同

**partner 的行顺序**: 本 pipeline 用 `annotations` 的顺序, 标准流用 unified scenario 的 dict 迭代顺序。**两套顺序不一定相同**, 但本 pipeline 特意**不排序**, 以对齐 nuPlan 标准流的 "dict 迭代 → objects 列表顺序" 约定。如果下游模型关心每一行的具体归属, 需要用每行的 `id` (本 pipeline 设为 `i+1`, annotation 索引 + 1)来 cross-reference。

所以结论是:

> **只要 `scene.frames[num_history_frames-1]` 的 ego 位姿 = nuPlan scenario 的起始帧 ego 位姿**(对 navtest stage1 场景成立), **两条 pipeline 输出的 2984 维观测, 在 ego(6) + road(2600) 这两块上逐 bit 等价**。partner(378) 在数值上相同, 但每一行对应的具体 agent id 可能不同(取决于两边的迭代顺序是否恰好一致)。

对 stage2 合成帧, 标准 pipeline **根本跑不通**, 本 pipeline 成为唯一可行方案。

### 4.7 其他工程差异

| 维度 | 本 pipeline | 标准 ScenarioMax |
|---|---|---|
| 性能 | 单 token ≈ 7~10 分钟(Boston GPKG + `unary_union` 是瓶颈) | 整库一次跑, 单 token 摊销更低 |
| 可并行化 | Step 2 各 token 独立, 易于 GNU Parallel | ScenarioMax 自带多进程 worker |
| 依赖环境 | 3 个 conda env | 1 个 (.venv) |
| 中间产物 | 按 token 目录存, 便于 debug | 整库 JSON(几十 GB) |
| 合成帧 | ✅ | ❌ |
| 支持 navtest 过滤 | ✅ 自然 | ✅ |

### 4.8 何时选哪条?

| 需求 | 推荐方案 |
|---|---|
| 只跑 stage1, 一次性大规模 | 标准 ScenarioMax(更快, 一次性产物) |
| 必须覆盖 stage2 合成帧 | **本 pipeline(唯一可行)** |
| 和 navsim 已经加载的 scene 紧耦合(需要 sensor data/lidar 元数据对齐) | **本 pipeline** |
| 和 GPUDrive 原生训练/评测 pipeline 对齐(想用已有的 training JSON) | 标准 ScenarioMax |

---

## 5. 输出文件与张量语义

单个 token 的最终目录结构:

```
exp/pipeline_output/00016f8b45c25a1d/
├── scene_metadata.pkl                   # Step 1: token + map + ego + goal + partners
├── gpudrive_json/
│   └── tfrecord-00016f8b45c25a1d.json   # Step 2: 完整 GPUDrive scenario
├── ego_obs.npy                          # Step 3: (6,)     float32
├── partner_obs.npy                      # Step 3: (63, 6)  float32
├── road_obs_gpudrive.npy                # Step 3: (200, 13) float32
├── waymo_obs_gpudrive.npy               # Step 3: (2984,)  float32 (ego ⊕ partner ⊕ road)
├── waymo_obs_gpudrive.png               # Step 3: BEV(ego + goal + partners + roads)
└── road_obs_gpudrive.png                # Step 3: BEV(仅道路, 调试用)
```

### 5.1 `ego_obs.npy` 的 `(6,)` 语义

| 列 | 含义 | 单位 |
|---|---|---|
| 0 | speed(`|velocity|`) | m/s |
| 1 | vehicle_length | m |
| 2 | vehicle_width | m |
| 3 | rel_goal_x (ego-local) | m |
| 4 | rel_goal_y (ego-local) | m |
| 5 | is_collided (0 = 未碰撞, 1 = 已碰撞) | — |

与 `env_torch.py:776-786` 的 `LocalEgoState` 6 列排列完全一致。

### 5.2 `partner_obs.npy` 的 `(63, 6)` 语义

每行代表一个 partner(第 `i` 个 partner 对应 JSON 里 `objects[i+1]`, id = `i+1`):

| 列 | 含义 | 单位 |
|---|---|---|
| 0 | speed | m/s |
| 1 | rel_pos_x (ego-local) | m |
| 2 | rel_pos_y (ego-local) | m |
| 3 | orientation (ego-local) | rad |
| 4 | vehicle_length | m |
| 5 | vehicle_width | m |

与 `env_torch.py:834-844` 的 `PartnerObs` 6 列排列完全一致。**行顺序就是 JSON `objects` 的顺序**(= `annotations` 的自然顺序, 不按距离排序), 与标准 ScenarioMax 流保持一致。不足 63 行的部分全零填充。

### 5.3 `road_obs_gpudrive.npy` 的 `(200, 13)` 语义

| 列 | 含义 | 单位 |
|---|---|---|
| 0 | x (ego-local) | m |
| 1 | y (ego-local) | m |
| 2 | segment_length | m |
| 3 | segment_width | m |
| 4 | segment_height | m |
| 5 | orientation (ego-local) | rad |
| 6 | one-hot: None | — |
| 7 | one-hot: RoadEdge | — |
| 8 | one-hot: RoadLine | — |
| 9 | one-hot: RoadLane | — |
| 10 | one-hot: CrossWalk | — |
| 11 | one-hot: SpeedBump | — |
| 12 | one-hot: StopSign | — |

行顺序: 按 GPUDrive C++ 的 KNN 输出(距 ego 由近到远)。不足 200 行时 C++ 会补零。

### 5.4 `waymo_obs_gpudrive.npy` 的 `(2984,)` 语义

`waymo_obs = np.concatenate([ego_obs, partner_obs.flatten(), road_obs.flatten()])` — 三段的切片公式:

| 切片 | 内容 | 元素数 |
|---|---|---|
| `[0 : 6]` | ego_obs | 6 |
| `[6 : 6 + 63*6 = 384]` | partner_obs.flatten() | 378 |
| `[384 : 384 + 200*13 = 2984]` | road_obs.flatten() | 2600 |

下游 `WaymoMLPAgent.compute_trajectory` 直接把这一整段 `.npy` load 进去, `reshape(-1)` 送 MLP, 不做任何再切分。

---

## 6. 后续可扩展

1. **批量化**: ✅ 已完成。Step 1 支持 `bash run_step1_export_metadata.sh all`(默认只 stage1/stage2 各一个 token, 可快速自检); Step 2/3 的 shell 脚本都带 `--auto`, 会自动发现 `exp/pipeline_output/*/scene_metadata.pkl` 和 `exp/pipeline_output/*/gpudrive_json/tfrecord-*.json` 全部处理。跑完整 navhard_two_stage 只需依次执行三个 shell(建议分别在三个 conda env 的 tmux 里)。
2. **Stage2 合成帧验证**: stage2 的 `scene.map_api` 与 stage1 一致(都是 navsim 重建的 NuPlanMap), 但 `scene_metadata.map_name` 必须存在于 stage2 的 synthetic pickle 里。当前 pipeline 已在 stage_two 第一个 token 上端到端跑通, 扩展到全集只需 `run_step1_export_metadata.sh all`。
3. **直接集成进 `extract_waymo_feature.py`**: 目前三阶段靠文件解耦; 如果接受"每 token 都要启动 3 个进程"的开销, 可以把 Step 2/3 包成 `subprocess.run(...)` 调用, 在 navsim-llh 主进程里串起来。
4. **对齐验证**:
   - road 块: 拿一个 navtest stage1 token, 两条 pipeline 各跑一遍, `np.allclose(road_obs_ours, road_obs_standard)`, 预期逐元素等价(浮点舍入除外)
   - ego 块: 同 token 下, `np.allclose(ego_obs_ours, ego_obs_standard)`, 预期除 `rel_goal_x/y` 之外完全一致
   - partner 块: 因两边 `objects` 顺序可能不同, 不能直接 `allclose`; 应按 `(rel_x, rel_y)` 或 `id` 做集合匹配后逐 agent 对比
5. **性能优化**: Step 2 里 `unary_union` 耗时约 5 分钟, 是整个 pipeline 的瓶颈。可以考虑缓存 `static_map_elements` 结果(同一张 map 只算一次 `unary_union`)。
6. **WaymoMLPAgent 训练**: 当前 `navsim/agents/waymo_mlp_agent.py` 是**随机初始化**的 MLP, 用来验证 pipeline 联通。要做实际 planner 可以:
   - 加 `get_feature_builders()` / `get_target_builders()`, 分别读 `.npy` 和 `scene.get_future_trajectory()`
   - 用 navsim 的 `run_training.py` 管线训练(`compute_loss` 已经预留为 L1 pattern, 可直接抄 `EgoStatusMLPAgent`)
   - 训练集只需用 `run_step1_export_metadata.sh all` + step2/3 把 navhard_two_stage 全集的 `.npy` 预生成一次, 训练时就是纯内存读, 无 GPUDrive 开销
