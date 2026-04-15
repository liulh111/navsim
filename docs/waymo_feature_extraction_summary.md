# Waymo 格式特征提取：navsim 直接提取 vs ScenarioMax+GPUDrive 流水线

## 1. nuPlan Map API 可提取的信息

navsim 和 ScenarioMax 底层都使用 nuPlan 的 **`AbstractMap`** 接口（具体实现为 `NuPlanMap`），通过 `get_proximal_map_objects(point, radius, layers)` 查询指定范围内的地图元素。

### 1.1 可查询的 SemanticMapLayer（共 9 种支持查询）

| SemanticMapLayer | 几何类型 | 主要属性 | 可用于查询 |
|---|---|---|---|
| **LANE** | Polygon | `baseline_path`(中心线), `left/right_boundary`(边界线), `adjacent_edges`, `speed_limit` | ✓ |
| **LANE_CONNECTOR** | Polygon | `baseline_path`(中心线), `left/right_boundary`, `turn_type` | ✓ |
| **CROSSWALK** | Polygon | `polygon` 外边界 | ✓ |
| **STOP_LINE** | Polygon | `polygon`, `stop_line_type`(STOP_SIGN/TRAFFIC_LIGHT/YIELD/TURN_STOP/PED_CROSSING) | ✓ |
| **INTERSECTION** | Polygon | `polygon`, `interior_edges`, `is_signaled` | ✓ |
| **ROADBLOCK** | Polygon | 包含若干 LANE，`incoming/outgoing_edges` | ✓ |
| **ROADBLOCK_CONNECTOR** | Polygon | 包含若干 LANE_CONNECTOR | ✓ |
| **WALKWAYS** | Polygon | `polygon` | ✓ |
| **CARPARK_AREA** | Polygon | `polygon` | ✓ |

> 另有 BASELINE_PATHS、BOUNDARIES 可通过 `_get_vector_map_layer` 访问（非 object 级别查询）。  
> SPEED_BUMP、PUDO、EXTENDED_PUDO 在枚举中存在但未实现。

### 1.2 nuPlan → Waymo 道路类型映射

Waymo/GPUDrive 定义了 7 种道路类型（type index 0-6）。从 nuPlan Map API 的映射关系如下：

| Waymo Type | Index | nuPlan 数据来源 | 提取方式 |
|---|---|---|---|
| **none** | 0 | — | 填充用零值 |
| **RoadLine** | 1 | `LANE.left/right_boundary`（有相邻车道的边界） | `boundary.discrete_path` → 逐段中点 polyline |
| **RoadEdge** | 2 | `LANE.left/right_boundary`（无相邻车道的边界） | 同上 |
| **RoadLane** | 3 | `LANE.baseline_path` + `LANE_CONNECTOR.baseline_path` | `baseline_path.discrete_path` → 逐段中点 polyline |
| **CrossWalk** | 4 | `CROSSWALK.polygon` | `polygon.exterior.coords` → 逐段中点 |
| **SpeedBump** | 5 | nuPlan 未实现 | 不可提取 |
| **StopSign** | 6 | `STOP_LINE`（`stop_line_type == STOP_SIGN`） | `polygon.exterior.coords` → 逐段中点 |

> **区分 RoadLine vs RoadEdge**：通过 `LANE.adjacent_edges` 属性判断。`adjacent_edges` 返回 `(left_adj, right_adj)` 元组，某侧为 `None` 表示该侧无相邻车道，对应 RoadEdge；否则为 RoadLine。

### 1.3 每个道路点的 13 维特征

```
[x, y, segment_length, segment_width, segment_height, orientation, type_onehot[7]]
```

| 维度 | 含义 | 说明 |
|---|---|---|
| 0-1 | x, y | 线段中点在 ego-local 坐标系下的位置 |
| 2 | segment_length | 线段半长 (half_length) |
| 3 | segment_width | 线段宽度 (RoadEdge=0.15, RoadLine=0.1, 其他=0) |
| 4 | segment_height | 线段高度 (RoadEdge=0.15, 其他=0) |
| 5 | orientation | 线段方向角（ego-local, [-π, π]） |
| 6-12 | type one-hot | 7 维 one-hot：[none, RoadLine, RoadEdge, RoadLane, CrossWalk, SpeedBump, StopSign] |

---

## 2. ScenarioMax + GPUDrive 流水线

### 2.1 整体流程

```
nuPlan 原始数据
    │
    ▼  [ScenarioMax: extractor.py]
UnifiedScenario (Python dict)
    │  - static_map_elements: 道路几何 + 类型
    │  - dynamic_agents: 轨迹 + 尺寸
    │
    ▼  [ScenarioMax: convert_to_json.py]
GPUDrive JSON 文件
    │  - objects[]: 动态实体 (position/heading/velocity × T)
    │  - roads[]: 道路几何 (type + geometry[])
    │  - metadata: sdc_index, tracks_to_predict
    │
    ▼  [GPUDrive C++: json_serialization.hpp]
内部数据结构 (MapRoad, MapObject)
    │  - 多边形简化 (Ramer-Douglas-Peucker)
    │
    ▼  [GPUDrive C++: level_gen.cpp]
物理实体 (Entity)
    │  - makeRoadEdge(): 逐段创建，位置=中点，scale=(half_len, w, h)
    │  - makeCube(): CrossWalk/SpeedBump 用 4 角点
    │  - makeStopSign(): 点几何
    │
    ▼  [GPUDrive C++: sim.cpp → collectMapObservationsSystem]
MapObservation[200] (9 维/条, KNN 最近 200 条)
    │  - [position(2), scale(3), heading(1), type(1), id(1), mapType(1)]
    │  - 坐标已转换到 agent-local 坐标系
    │
    ▼  [GPUDrive Python: env_torch.py + roadgraph.py]
最终 2984 维观测向量
    - type(1 float) → one_hot(7 dims)：9 维 → 13 维
    - 去掉 id 和 mapType
    - 拼接 ego(6) + partner(63×6) + road(200×13) = 2984
```

### 2.2 ScenarioMax 关键方法

**`extractor.py` → `extract_static_map_elements(map_api, center)`**
- 使用 `map_api.get_proximal_map_objects()` 查询 250m 范围
- 查询层：LANE, LANE_CONNECTOR, ROADBLOCK, ROADBLOCK_CONNECTOR, STOP_LINE, CROSSWALK, INTERSECTION, BOUNDARIES
- 提取的道路类型更细：区分 ROAD_LINE_SOLID_SINGLE_WHITE / ROAD_LINE_BROKEN_DOUBLE_YELLOW 等子类型
- 坐标以 ego 初始位置为原点进行平移

**`convert_to_json.py` → `roadgraph.py`**
- 将 UnifiedScenario 转为 GPUDrive 期望的 JSON 格式
- 道路类型映射到 `map_element_id` 枚举（1=LANE_FREEWAY, 6=ROAD_LINE_BROKEN_SINGLE_WHITE, 15=ROAD_EDGE_BOUNDARY, 18=CROSSWALK 等）

### 2.3 GPUDrive 关键方法

**`json_serialization.hpp` → `from_json(j, MapRoad)`**
- 解析 JSON 中的 `type` (string) → `EntityType` 枚举
- 多边形简化：几何点数 ≥10 时使用 Ramer-Douglas-Peucker 算法去除共线点

**`level_gen.cpp` → `makeRoadEdge(road, segment_idx)`**
```cpp
// 核心：从连续两点创建一个线段实体
Vector3 start = {p1.x - mean.x, p1.y - mean.y, z};
Vector3 end   = {p2.x - mean.x, p2.y - mean.y, z};
position = midpoint(start, end);           // 线段中点
rotation = atan2(dy, dx);                  // 线段方向
scale    = {distance(start,end)/2, 0.1, 0.1};  // 半长, 宽, 高
```

**`sim.cpp` → `collectMapObservationsSystem()`**
- 对每个 agent，遍历所有道路实体
- 通过 `ReferenceFrame.observationOf()` 转换到 agent-local 坐标系
- KNN 选取最近的 200 条道路（`kMaxAgentMapObservationsCount = 200`）

**`env_torch.py` → Python 端后处理**
- `roadgraph.one_hot_encode_road_point_types()`：type 单值 → 7 维 one-hot
- 去掉 id 和 mapType 字段
- 拼接为 2984 维向量

---

## 3. navsim 直接提取 vs ScenarioMax+GPUDrive：显著差异

### 3.1 Map API 对比

| | navsim | ScenarioMax |
|---|---|---|
| **API 类型** | `AbstractMap`（实际为 `NuPlanMap`） | `NuPlanMap`（从 `NuPlanScenario.map_api` 获取） |
| **底层实现** | 同一个 `NuPlanMap` 类 | 同一个 `NuPlanMap` 类 |
| **地图数据源** | nuPlan GPKG 文件 (`NUPLAN_MAPS_ROOT`) | 同一套 nuPlan GPKG 文件 |
| **加载方式** | `get_maps_api(root, version, map_name)` | `NuPlanScenario` 自动加载 |
| **结论** | **完全相同的底层 Map API** ||

### 3.2 流水线差异

| 方面 | navsim 直接提取 | ScenarioMax + GPUDrive |
|---|---|---|
| **步骤数** | 1 步（Python 直接提取） | 3 步（Python提取 → JSON中间件 → C++仿真） |
| **中间格式** | 无 | UnifiedScenario dict → GPUDrive JSON |
| **坐标变换** | Python 中完成（global → ego-local） | C++ 仿真中完成（global → world-mean-centered → ego-local） |
| **多边形简化** | 无（保留全部原始点） | GPUDrive C++ 端做 Ramer-Douglas-Peucker 简化 |
| **道路点数** | 每种类型 ≤500，总数可变 | 固定 200 条（KNN 最近） |
| **one-hot 编码** | Python 中直接构建 | C++ 存 type index → Python 端 `F.one_hot()` |
| **道路子类型** | 仅 7 大类（RoadLine/RoadEdge/...） | 保留细分类型（solid_white/broken_yellow 等）到 `mapType` 字段 |

### 3.3 关键差异分析

**1. 多边形简化**
- ScenarioMax+GPUDrive 在 C++ 端对长几何（≥10 段）执行 Ramer-Douglas-Peucker 简化，减少冗余共线点
- navsim 直接保留 nuPlan `discrete_path` 的全部原始点（~0.5m 间隔），通过 `MAX_POINTS_PER_TYPE` 限制总量

**2. 道路选择策略**
- GPUDrive：全局 KNN 选最近 200 条，不区分类型
- navsim：按类型分别取最近 500 条，总数可变，保证每种类型都有代表

**3. 坐标中心化**
- GPUDrive：先减去全局均值（`WorldMeans`），再在仿真中转 agent-local
- navsim：直接从 global 坐标转 ego-local（平移 + 旋转）

**4. 道路类型粒度**
- GPUDrive 的 `MapType` 枚举保留了 Waymo 原始的细分类型（19 种），存储在额外的 `mapType` 字段中，但最终观测中被去掉，只保留 7 大类的 one-hot
- navsim 从一开始就只区分 7 大类

**5. 边界线的宽度/高度**
- GPUDrive `makeRoadEdge` 对所有线段统一使用 `width=0.1, height=0.1`
- navsim 根据类型区分：RoadEdge `(0.15, 0.15)`，RoadLine `(0.1, 0.0)`，其他 `(0.0, 0.0)`
