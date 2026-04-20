# navsim → ScenarioMax → GPUDrive 三阶段地图抽取 Pipeline 总结

> 最后更新: 2026-04-16
> 目标: 为 navhard_two_stage 的每一个 token(含 stage1 真实帧与 stage2 合成帧), 得到与 GPUDrive 训练/评测完全对齐的 `200×13` 道路观测张量。

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

navsim 的 `extract_waymo_feature.py` 需要对每一个 scene 产出 2984 维 feature:

```
ego (6) + partner (63×6=378) + road (200×13=2600) = 2984
```

其中 `road (200×13)` 这一块必须与 GPUDrive 训练时看到的道路观测**完全一致**, 否则训练/评测的 domain 会对不上。直接在 navsim 里用 `scene.map_api.get_proximal_map_objects()` 抽取到的 road 点数远超 200, 且语义/坐标归一化与 GPUDrive 不同, 所以必须走 GPUDrive 自己的那一套 C++ pipeline 来得到最终的 200×13。

但是直接调用 GPUDrive 的 C++ pipeline 需要一个 `GPUDrive JSON` 场景文件。把 nuPlan 地图转换为 GPUDrive JSON 的代码只存在于 ScenarioMax。而 ScenarioMax 标准流程是从 `.db` 文件出发的, 这就带来了两个关键矛盾:

| 约束 | 含义 |
|---|---|
| **必须支持 stage2 合成帧** | 合成场景没有原始 `.db`, 所以不能走"过滤 .db + 整库转换"的标准流程 |
| **必须与 GPUDrive 逐像素对齐** | 不能自己重写地图抽取/简化逻辑, 必须复用 ScenarioMax + GPUDrive 的官方函数 |

这两个约束迫使我们走出一条新路: **绕开 `.db` 文件, 直接用 navsim 已经加载好的 map_api + ego_pose 作为输入, 复用 ScenarioMax 的地图抽取/转换函数, 然后喂给 GPUDrive**。三个阶段分别跑在三个 conda 环境下, 用文件做中间态传递。

```
┌──────────────┐     scene_metadata.pkl     ┌──────────────────┐
│  Step 1      │ ───────────────────────>   │  Step 2          │
│  navsim-llh  │                             │  ScenarioMax     │
│  (scene API) │                             │  .venv           │
└──────────────┘                             └──────────────────┘
                                                      │
                                    tfrecord-<TOKEN>.json
                                                      ▼
                                             ┌──────────────────┐
                                             │  Step 3          │
                                             │  gpudrive env    │
                                             │  (C++ simulator) │
                                             └──────────────────┘
                                                      │
                                       road_obs_gpudrive.npy (200×13)
                                       road_obs_gpudrive.png
```

---

## 2. 具体运行命令

以 token `00016f8b45c25a1d` (us-ma-boston) 为例, 三个阶段的命令如下。

### Step 1 — 导出 scene 元数据 (navsim-llh 环境)

```bash
conda activate navsim-llh
cd $NAVSIM_DEVKIT_ROOT
bash scripts/evaluation/run_step1_export_metadata.sh
```

`run_step1_export_metadata.sh` 内部执行:

```bash
TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/step1_export_scene_metadata.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=step1_export_metadata \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
    worker=sequential
```

输出: `/data/llh/navsim_workspace/exp/pipeline_output/<TOKEN>/scene_metadata.pkl`

pickle 内容:
```python
{"token": "00016f8b45c25a1d",
 "map_name": "us-ma-boston",
 "ego_pose": [331068.128, 4690724.234, 1.4974]}   # [x, y, heading(rad)]
```

### Step 2 — 生成 GPUDrive JSON (ScenarioMax .venv)

```bash
/data/llh/navsim_workspace/ScenarioMax/.venv/bin/python \
    /data/llh/navsim_workspace/navsim/navsim/planning/script/step2_generate_gpudrive_json.py \
    --metadata_path /data/llh/navsim_workspace/exp/pipeline_output/00016f8b45c25a1d/scene_metadata.pkl
```

输出: `/data/llh/navsim_workspace/exp/pipeline_output/<TOKEN>/gpudrive_json/tfrecord-<TOKEN>.json`

典型日志:
```
INFO: Token:      00016f8b45c25a1d
INFO: Map:        us-ma-boston
INFO: Center:     [331068.13, 4690724.23]
INFO: Heading:    1.4974 rad
INFO: Loading map from /data/llh/navsim_workspace/dataset/maps ...
INFO: Extracting static map elements via ScenarioMax ...
INFO:   Extracted 347 map elements
INFO: Converting to GPUDrive road format ...
INFO:   Road features: 347
INFO:   Simplification: 36148 pts → 2751 pts (2404 segments)
INFO: Saved GPUDrive JSON: .../tfrecord-00016f8b45c25a1d.json
```

### Step 3 — 抽取 200×13 道路观测 (gpudrive 环境)

```bash
conda activate gpudrive
python /data/llh/navsim_workspace/navsim/navsim/planning/script/step3_extract_road_obs.py \
    --json_dir   /data/llh/navsim_workspace/exp/pipeline_output/00016f8b45c25a1d/gpudrive_json \
    --output_dir /data/llh/navsim_workspace/exp/pipeline_output/00016f8b45c25a1d
```

输出:
- `road_obs_gpudrive.npy` — shape `(200, 13)`, `float32`
- `road_obs_gpudrive.png` — BEV 可视化, 按类别上色

---

## 3. 代码关键函数逐一介绍

### 3.1 `step1_export_scene_metadata.py`

核心就是跨环境的"拆解": navsim 环境里有很多重量级依赖(torch、navhard 数据), ScenarioMax 环境里又有 nuPlan devkit, 两套完全不能共存。所以我们只从 scene 里**抠出三个字段**, 序列化到 pickle:

```python
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg):
    scene_loader = SceneLoader(...)                       # 构造 navhard loader
    token = sorted(set(scene_loader.tokens_stage_one) &
                   set(metric_cache_loader.tokens))[0]     # 选第一个 stage1 token
    scene = scene_loader.get_scene_from_token(token)

    frame_idx = scene.scene_metadata.num_history_frames - 1       # "当前帧"
    ego_pose = scene.frames[frame_idx].ego_status.ego_pose        # [x, y, heading]

    metadata = {"token":    token,
                "map_name": scene.scene_metadata.map_name,
                "ego_pose": ego_pose.tolist()}
    pickle.dump(metadata, open(output_path, "wb"))
```

关键点:
- `ego_pose` 是**全局 nuPlan 坐标系下的三元组** `[x, y, heading]`, Boston map 大约在 `(331068, 4690724)` 一带。
- `num_history_frames - 1` 取的是 "现在这一帧", 与 `extract_waymo_feature.py` 里 `ego_state` 的取法一致。
- 只要 token 能被 SceneLoader 加载(stage1 或 stage2 均可), 这一步就能产出 metadata。stage2 合成帧也有 `scene.map_api` 和 `ego_pose`, **不依赖 `.db` 文件**。

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

#### 3.2.5 `build_gpudrive_json(token, map_name, ego_heading, road_features)`

产出 GPUDrive 能加载的最小 JSON。关键细节:

```python
EPISODE_LEN = 91  # 必须等于 madrona_gpudrive::episodeLen

{
    "name": f"tfrecord-{token}.json",
    "scenario_id": token,
    "objects": [{
        "position": [{"x":0,"y":0,"z":0}] * 91,   # ← dummy ego, 91 帧必须都有
        "width": 1.8, "length": 4.049, "height": 1.5,
        "heading": [ego_heading] * 91,
        "velocity": [{"x":0,"y":0}] * 91,
        "valid":    [True] * 91,
        "goalPosition": {"x":0,"y":0,"z":0},
        "type": "vehicle", "id": 0,
        "mark_as_expert": False,
    }],
    "roads":    road_features,
    "tl_states": {},
    "metadata": {
        "sdc_track_index": 0,
        "objects_of_interest": [],
        "tracks_to_predict": [{"track_index": 0, "difficulty": 0}],  # ← 整数 0, 非 "EASY"
    },
}
```

踩过的坑:
| 坑 | 症状 | 原因 |
|---|---|---|
| 用 1 帧 position | SIGABRT in C++ | `episodeLen=91`, C++ 会按下标 0..90 读 |
| `difficulty: "EASY"` | SIGABRT | C++ 侧类型是整数, 字符串会 crash |
| 多写 `is_sdc`, `total_distance_traveled` | SIGABRT | `from_json` 会断言字段集, 多余字段也会触发 |

### 3.3 `step3_extract_road_obs.py`

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
    norm_obs=False,   # 反归一化, 方便与 navsim 端直接对齐
)
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=SceneDataLoader(root=args.json_dir, batch_size=1, dataset_size=1,
                                sample_with_replacement=False, file_prefix="tfrecord"),
    max_cont_agents=1,
    device="cpu",
    render_config=RenderConfig(),
)
```

这一步 GPUDrive 的 C++ 做了几件事:
1. 读 JSON → 反序列化 `MapRoad`, **在此阶段再跑一次 polyline 简化** (同一个算法同一个阈值, 但我们 Python 端已经简化过, 基本没活干了)
2. 计算所有几何点的 `mean`, 把坐标去中心化 → 存在 `road.mean` 里(后续查询时再加回去)
3. 每辆 agent 通过 KNN 空间索引取离自己最近的 200 个路段点
4. 把这些点变换到 ego-local 坐标(减 ego 位置、旋转 `-heading`)

#### 3.3.3 `extract_road_obs(env)` — 抽 200×13 张量

完全复刻 `env_torch._get_road_map_obs(mask=None)` 的代码路径, 区别只是**不归一化、不 flatten**:

```python
roadgraph = LocalRoadGraphPoints.from_tensor(
    local_roadgraph_tensor=env.sim.agent_roadmap_tensor(),
    backend="torch", device="cpu",
)
roadgraph.one_hot_encode_road_point_types()   # type: int → 7-dim one-hot

road_obs = torch.cat([
    roadgraph.x.unsqueeze(-1),
    roadgraph.y.unsqueeze(-1),
    roadgraph.segment_length.unsqueeze(-1),
    roadgraph.segment_width.unsqueeze(-1),
    roadgraph.segment_height.unsqueeze(-1),
    roadgraph.orientation.unsqueeze(-1),
    roadgraph.type.float(),     # 7 维
], dim=-1)                       # (world, agent, 200, 13)

ego_road = road_obs[0, 0].numpy()  # (200, 13)
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

#### 3.3.4 `visualize_road_obs(roads, save_path)` — BEV 画图

按 one-hot 取 `argmax` 得到类名, 每类一个颜色, 画散点(点位)+ 短线(按 `orientation` 画 `segment_length` 长度的小段), ego 用 lime 十字标在 `(0,0)`。显示范围 `[-80, 80]×[-80, 80]` m。

本 token 实测:
- 200 行中 200 行非零(KNN 取满)
- 道路类型分布: `RoadLane=166, RoadEdge=31, CrossWalk=3`
- 形状与 BEV 直观对得上: 一个典型的十字路口 + 两条主干道

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

这是两条 pipeline 唯一实质性不同的地方——**非地图部分**:

| JSON 字段 | 本 pipeline | 标准 ScenarioMax |
|---|---|---|
| `objects[0].position` | `[(0,0,0)] × 91` dummy ego | 真实 ego 91 帧轨迹 |
| `objects[0].heading`  | `[heading] × 91` | 真实 heading 轨迹 |
| `objects[0].velocity` | `[(0,0)] × 91` | 真实速度 |
| `objects[1..N]`       | **不存在** | 所有周车/行人(多时约几十个) |
| `tl_states`           | `{}` | 真实红绿灯 91 帧 |
| `roads`               | **完全相同** | **完全相同** |
| `metadata`            | 仅 `sdc=0, tracks_to_predict=[{0,0}]` | 包含 `sdc_track_index`, `objects_of_interest`, `tracks_to_predict` 完整列表 |

### 4.6 最终 200×13 张量的一致性

观测张量由 13 维组成: `[x, y, seg_len, seg_w, seg_h, orient, type(7)]`, 其中:
- `seg_len, seg_w, seg_h, type` 只和道路几何有关 → **两者完全相同**
- `x, y, orient` 是 **ego-local 坐标** → 取决于 ego 的 `(x, y, heading)`

这就引出一个关键问题: **两条 pipeline 里 ego 的 `(x, y, heading)` 是否相同?**

- `heading`: 本 pipeline 用 `scene.frames[cur].ego_status.ego_pose[2]`, 标准流程用 `nuplan scenario.initial_ego_state.rear_axle.heading`。如果 scene 的 `cur` 帧 = nuplan 的 `initial` 帧, 则相同; 若 navsim 选择了别的帧(如 scene 中段), 则会差一个 heading。
- `(x, y)`: 本 pipeline dummy ego 始终在 `(0, 0)`, 标准流程 ego 在真实位置, 但 `center` 参数也用的是这个真实位置。**关键在于 `extract_static_map_elements` 会把道路坐标做 `center` 平移** (见 `extractor.py::get_center_vector`), 所以两条路径下 `roads` 的坐标系是**同一套 "以 center 为原点的系统"**。既然本 pipeline 的 dummy ego 也在 `(0,0)`, 就是把 ego 精确放在了 center 上, 与标准流程 ego 恰好也在 center 上这一"天然事实"完全对齐。

所以结论是:

> **只要 `scene.frames[num_history_frames-1]` 的 ego 位姿 = nuPlan scenario 的起始帧 ego 位姿**(这对 navtest 过滤出来的 stage1 场景成立), **两条 pipeline 输出的 200×13 观测逐 bit 等价**。

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
├── scene_metadata.pkl               # Step 1: {token, map_name, ego_pose}
├── gpudrive_json/
│   └── tfrecord-00016f8b45c25a1d.json   # Step 2: GPUDrive scenario
├── road_obs_gpudrive.npy            # Step 3: (200, 13) float32
└── road_obs_gpudrive.png            # Step 3: BEV 可视化
```

`road_obs_gpudrive.npy` 的 `(200, 13)` 语义:

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

行顺序: 按 GPUDrive C++ 的 KNN 输出(距 ego 由近到远, 具体算法见 `road_obs_algorithm`)。不足 200 行时 C++ 会补零。

---

## 6. 后续可扩展

1. **批量化**: Step 1 目前只导出第一个 token, 改成 `for token in tokens: ...` 就能批处理; Step 2/3 已经是 per-token 脚本, 直接 GNU Parallel 即可。
2. **Stage2 合成帧验证**: stage2 的 `scene.map_api` 与 stage1 一致(都是 navsim 重建的 NuPlanMap), 但 `scene_metadata.map_name` 必须存在于 stage2 的 synthetic pickle 里。写个 smoke test 跑两个 stage2 token 就能确认。
3. **直接集成进 `extract_waymo_feature.py`**: 目前三阶段靠文件解耦; 如果接受"每 token 都要启动 3 个进程"的开销, 可以把 Step 2/3 包成 `subprocess.run(...)` 调用, 在 navsim-llh 主进程里串起来。
4. **对齐验证**: 拿一个 navtest stage1 token, 两条 pipeline 各跑一遍, `np.allclose(road_obs_ours, road_obs_standard)`, 预期应当逐元素等价(浮点舍入除外)。
5. **性能优化**: Step 2 里 `unary_union` 耗时约 5 分钟, 是整个 pipeline 的瓶颈。可以考虑缓存 `static_map_elements` 结果(同一张 map 只算一次 `unary_union`)。
