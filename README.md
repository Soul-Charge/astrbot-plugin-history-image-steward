# history_image_steward（历史图片管家）

> 让模型只看见最近 K 轮的图片，更早的历史图换成文字描述 —— 默认不写库、停用即恢复。

## 它解决什么问题

群友发的图会被 AstrBot 编码成 `data:image/...;base64,...` **写进会话历史**，此后每一轮请求
都原样重发。图片在 token 计数里只算 765/张，所以 AstrBot 的 token 压缩永远不会被触发；
但请求体是实打实的几十 MB，网关直接 `Failed to buffer the request body: length limit exceeded`。

实测（2026-09-20 分析报告）：`bot:GroupMessage:200000000` 历史 50.77 MB、95 轮，
9/19 当天连续 15 次请求全部失败；单靠 `max_context_length` 轮数压缩救不了
「图片集中在最近几轮」的会话（如 `300000000`，最近 10 轮仍有 10.46 MB）。

## 它怎么工作

```
第 1 层  @filter.on_llm_request(priority=500)   —— 只读，不改请求
  (a) dry_run 时按 req.contexts 统计「本次将剥离几张」并打日志；
  (b) 从 req.image_urls 兜底预热 caption（**补充来源**，见下面的 v1.0.1 说明）。
      —— 这里绝不动 req.contexts（原因见下）。

第 2 层  补丁 ProviderOpenAIOfficial._prepare_chat_payload  —— 真正剥离 + 预热
  该方法对 contexts 做的是 copy.deepcopy，产出的 messages 只喂给本次 HTTP 请求，
  边界按 K+1 条 user 消息计（+1 是已经合并进来的当前轮），因此：
      · 最近 K 轮的图 -> 原样保留（模型还能「指上一张图」）
      · 更早的图 -> 命中 caption 缓存则换 [图片：<描述>]，否则换 [图片] 占位
      · 当前轮的图 -> 绝不被剥（它就是那个 +1）
  同一层顺手把「本轮被保留在窗口内」的图丢给后台任务生成 caption（v1.0.1 起的**主路径**）。
```

> **v1.0.1 修正（2026-09-20 线上实测）**
> v1.0.0 只在钩子里从 `req.image_urls` 预热 caption，结果 17 次请求剥掉 322 张图，
> caption 任务**一次都没起来** —— 因为 AstrBot 4.28.1 的 `internal.py` 里
> `await prepare_request_images(...)` 就紧跟在 `call_event_hook(OnLLMRequestEvent, ...)`
> 之后一行，钩子触发时 `req.image_urls` 还是空的（引用图更走 `extra_user_content_parts`）。
> 现在主路径改成在请求体那层预热「保留下来的那批图」：它们将来滑出窗口时，是从会话历史里
> 读出来的**同一份字节**，指纹必然对得上。钩子侧保留为兜底，且无论有没有图都打一行 DEBUG；
> caption 调用自身也过补丁，用 ContextVar 标记避免自环。

当前轮的图与 `req.contexts`（纯历史）在钩子上天然分离，所以剥离在结构上不可能误伤当前轮
（边界按 K+1 计）。反过来说，也正因为钩子时刻拿不到当前轮的图，caption 预热才必须挪到第 2 层。

## ⚠️ 为什么不直接在钩子里改 req.contexts（重要，已实测）

看起来最自然的做法是「在 `on_llm_request` 里把 `req.contexts` 的历史图原地换成文字」，
但那会**永久改写数据库**：

```
tool_loop_agent_runner.py:451
    self.run_context.messages = await self.context_manager.process(...)   # 就地写回
        ↑ run_context.messages 由 req.contexts 构造

internal.py:397  _save_to_history(..., agent_runner.run_context.messages, ...)
conversation_mgr.py:280  db.update_conversation(content=history)   # 整体覆盖 conversations.content
```

也就是说：**任何在 `on_llm_request` 里对 `req.contexts` 的就地修改，都会在本轮结束时
被写回 `conversations` 表**（AstrBot 内置的按模型能力清理历史、本机的
`astrbot_plugin_ignore_bad_images` 都走这条路）。老图会被永久替换、无法恢复，
「停用插件即恢复」也就不成立了。

本插件因此把剥离放在 **provider 的请求体组装层**（`_prepare_chat_payload` 里的 deepcopy 上），
`req.contexts` 一个字节都不动。

## 两种模式

| 模式 | `persist_strip` | 行为 | DB |
|---|---|---|---|
| **默认（非破坏）** | `false` | 只改本次请求体 | 一个字节都不动，停用即完全恢复 |
| 瘦身模式 | `true` | 钩子额外原地改写 `req.contexts`，剥离结果随本轮写库 | 老图被永久换成文字，**不可逆** |

瘦身模式的收益是 `data_v4.db` 与 WAL 会显著变小（那个群 50 MB 的历史会降到 KB 级），
代价是不可逆。要开的话请先备份 `data_v4.db` + `-wal` + `-shm`。

## 配置（WebUI → 插件 → 历史图片管家）

| 键 | 默认 | 说明 |
|---|---|---|
| `enable` | `true` | 总开关 |
| `dry_run` | **`true`** | 只记日志，不改请求体、也不生成描述 |
| `persist_strip` | **`false`** | 是否把剥离结果写回库（见上表，不可逆） |
| `image_keep_rounds` | **`2`** | 图片记忆深度 K（按 user 消息计数）。0 = 历史图全剥；1 = 上一张图还能指 |
| `enable_caption` | `true` | 剥离图是否替换为文字描述 |
| `caption_provider_id` | `""` | 空 = 自动选第一个声明了 `image` 模态的 provider |
| `caption_prompt` | `请用一两句话客观描述这张图片内容` | caption 提示词 |
| `caption_timeout_sec` | `20` | 后台 caption 超时，超时该图后续落 `[图片]` |
| `caption_max_chars` | `150` | 描述截断长度，0 = 不截断 |
| `cache_max_entries` | `1000` | caption 缓存 LRU 上限 |

配置保存后 AstrBot 会自动重载插件（`/config/plugin/update` → `plugin_manager.reload`），
无需重启容器。

**老图不补描述**：插件安装前就已经躺在历史里的图不会触发同步生成描述（那会让当次请求多等几秒），
直接落 `[图片]` 占位。只有上线后新收到的图才会预热描述（v1.0.1 起在它**还在 K 窗口内**时就后台生成）。这是确认过的取舍。

## 与其它配置 / 插件的关系

| 项 | 关系 |
|---|---|
| `max_context_length = 20` | **保留** —— 管文本（RAG 块、tool 结果） |
| `dequeue_context_length = 10` | 保留 |
| `image_compress_options` | **可调回 1280/95** —— 当前轮恢复全画质 |
| `/reset` 清存量 | 仍建议做一次（给 DB 瘦身 + 干净起点），但不再是救火必需 |
| `astrbot_plugin_ignore_bad_images` | 钩子优先级 1000（先跑），负责丢坏图、把引用规范化成真 data URL，本插件（500）拿到的是干净的 data URL |
| `livingmemory` | 本插件只替换历史里的 `image_url` 内容块，不删消息、不改结构 |

## 缓存文件

`data/plugin_data/history_image_steward/caption_cache.json`

```json
{"version": 1, "entries": {"<sha1(base64 载荷)>": "一只橘猫趴在键盘上"}}
```

指纹只看 base64 **载荷**、不看 `data:<mime>;base64` 前缀：当前轮（生成描述时）和若干轮后
（落进历史时）的 MIME 前缀并不保证一致，这样两边才能对上。落盘走「同目录临时文件 +
`os.replace`」的原子替换。

## 验证计划

### 第一步：dry_run（默认状态）

重载插件后正常聊天，看容器日志：

```bash
docker logs astrbot --since 10m 2>&1 | grep history_image_steward
```

期望输出：

```
[history_image_steward] 配置已加载: K=2 dry_run=True persist_strip=False caption=True provider=auto(第一个含 image 模态) timeout=20.0s max_chars=150 cache_max=1000
[history_image_steward] caption 缓存: /AstrBot/data/plugin_data/history_image_steward/caption_cache.json (已载入 0 条)
[history_image_steward] 请求体剥离补丁: {ProviderOpenAIOfficial: patched}
[history_image_steward] DRY-RUN 本次将剥离 12 张历史图（其中 12 张落 [图片] 占位），保留最近窗口 3 张，边界下标 45/49 umo=bot:GroupMessage:200000000
```

与数据库里的图片分布对账（此时请求体与数据库都**完全没有被改动**）：

```bash
docker exec astrbot python3 -c "import sqlite3;c=sqlite3.connect('file:/AstrBot/data/data_v4.db?mode=ro',uri=True);print(list(c.execute('select user_id, length(content), json_array_length(content) from conversations order by length(content) desc limit 5')))"
```

### 第二步：关闭 dry_run，生效

在 WebUI 把 `dry_run` 改成 `false`，保存（插件自动重载）。然后逐条验收：

| 验收项 | 期望 |
|---|---|
| 发图当轮识图 | 正常（当前轮不受损） |
| K=2 时问「上一张图是什么」 | 模型仍能答（保留窗口内） |
| 再问三轮前的图 | 回答基于 `[图片：…]` 文字描述 |
| 只发图不配文字的消息 | 占位符替换后 content 非空、API 不报错 |
| 数据库 `length(content)` | **不变**（默认模式） |
| `Failed to buffer` 计数 | 停止增长 |

```bash
docker logs astrbot --since 1h 2>&1 | grep -c "Failed to buffer"
docker logs astrbot --since 10m 2>&1 | grep "history_image_steward" | tail -20
```

## 回滚

- 紧急：WebUI 里关掉 `enable`，或直接禁用/删除插件 → 立刻恢复 AstrBot 原生行为
  （禁用时插件的 `terminate()` 会把补丁置为 no-op）；
- 默认模式下数据库全程未被动过，不需要任何数据恢复；
- 缓存文件可以直接删，下次启动会重建（只会让老图重新落 `[图片]`）。

## 文件

| 文件 | 作用 |
|---|---|
| `main.py` | 插件入口：配置、`on_llm_request` 钩子、provider 请求体补丁、后台 caption、缓存落盘 |
| `steward_core.py` | 纯逻辑层（无 AstrBot 依赖）：边界计算、图片指纹、剥离、LRU 缓存 |
| `_conf_schema.json` | WebUI 可视化配置 |
| `tests/` | 单元测试 + 假 astrbot 的集成测试 |

## 跑测试

```bash
cd data/plugins/history_image_steward
python3 tests/test_steward_core.py      # 27 条：纯逻辑
python3 tests/test_main_steward.py      # 24 条：钩子 / 请求体补丁（假 astrbot，不碰容器）
```
