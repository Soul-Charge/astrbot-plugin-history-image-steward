"""history_image_steward —— 历史图片管家。

问题
----
群友发的图会被 AstrBot 编码成 data URL 写进会话历史，此后每一轮请求都原样重发。
图片在 token 计数里只算 765/张，所以 token 压缩永远不会被触发；但请求体是实打实的
几十 MB，网关直接 Failed to buffer the request body: length limit exceeded。

方案（默认非破坏性，DB 全程不动）
------------------------------
1. @filter.on_llm_request：只做两件事 ——
   (a) dry_run 时按 req.contexts 统计「本次将剥离几张」并打日志；
   (b) 从 req.image_urls 兜底预热 caption。这只是**补充来源**：AstrBot 4.28.1 里
       这个字段在钩子时刻常常还是空的（当前轮的图，尤其引用图，由钩子之后的
       prepare_request_images 才收进来）；无论有没有图都会打一行 DEBUG。
   注意：这里**绝不修改** req.contexts（原因见下）。

2. 补丁 ProviderOpenAIOfficial._prepare_chat_payload：真正的剥离发生在这一层。
   该方法对 contexts 做的是 copy.deepcopy，产出的 messages 只用于本次 HTTP 请求体，
   不会回写会话历史 —— 这是本插件「DB 全程不动」的关键。
   边界按 K+1 条 user 消息计（+1 是已经合并进来的当前轮），因此保留窗口与纯历史场景一致。

3. 在同一层顺手给「本轮被保留在 K 窗口内」的图起后台任务生成 caption（主路径）。
   选这里的理由：这些图将来滑出窗口时，是从会话历史里读出来的**同一份字节**，
   指纹必然对得上；而钩子时刻根本拿不到它们 —— internal.py 里
   `await prepare_request_images(...)` 就紧跟在 `call_event_hook(...)` 之后一行。
   自环保护：我们自己的 caption 调用也会经过 provider → 补丁，用 ContextVar 标记，
   绝不在 caption 请求上再起 caption 任务。

为什么不能直接在钩子里改 req.contexts（重要）
-------------------------------------------
tool_loop_agent_runner.py:451 会 self.run_context.messages = await context_manager.process(...)，
而 run_context.messages 是由 req.contexts 构造的；随后 internal.py 的 _save_to_history()
把 run_context.messages 整体 update_conversation(history=...) 覆盖写回 conversations 表。
所以在钩子里原地改 req.contexts，改动会在这一轮结束时被**持久化进数据库**，
老图会被永久替换成文字、无法恢复。默认路径因此改走 provider 层（deepcopy 上操作）。

如果确实想连数据库一起瘦身，把配置 persist_strip 打开即可：
那时钩子会额外原地改写 req.contexts，剥离结果随本轮写库（不可逆，请先备份 data_v4.db）。

停用/删除插件即完全恢复 AstrBot 原生行为。
"""

from __future__ import annotations

import asyncio
import contextvars
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:  # 常规加载路径：插件属于 data.plugins.<dir> 命名空间包
    from . import steward_core as core
except ImportError:  # pragma: no cover - 兼容非包方式导入（例如直接跑单测）
    import steward_core as core  # type: ignore[no-redef]

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
except Exception:  # pragma: no cover - 拿不到就走插件目录兜底
    get_astrbot_plugin_data_path = None  # type: ignore[assignment]

try:
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
except Exception:  # pragma: no cover - 缺失时插件只保留钩子层
    ProviderOpenAIOfficial = None  # type: ignore[assignment]


PLUGIN_NAME = "history_image_steward"
DESCRIPTION = (
    "历史图片管家 - 只把最近 K 轮的图片原样发给模型，更早的图换成文字描述或 [图片] 占位，"
    "杜绝 base64 历史把请求体撑爆（默认不写库、停用即恢复）"
    "（v1.1 起同图去重折叠）"
)
VERSION = "v1.1.0"
LOG_TAG = "[history_image_steward]"

# caption 调用自己也会走 provider → _prepare_chat_payload 补丁；用它把
# 「这次请求是我们自己 caption 任务发的」标出来，避免自环（caption 生 caption）。
# 用 ContextVar 而不是实例属性：每个 asyncio 任务有独立的上下文副本，
# 既不会串到别的并发请求，也能被同一任务里 await 下去的补丁代码读到。
_IN_CAPTION_CALL = contextvars.ContextVar(
    "history_image_steward_in_caption_call", default=False
)

# 比 astrbot_plugin_ignore_bad_images（1000）低：先让它把坏图清掉、把引用规范化成
# 真正的 data URL，我们再处理剩下的历史图。
HOOK_PRIORITY = 500

PATCH_VERSION = 1
PATCH_TARGET = "_prepare_chat_payload"
PATCH_ATTR = "_his_payload_patch"
PATCH_ORIG_ATTR = "_his_payload_original"

SAVE_DEBOUNCE_SEC = 5.0
DATA_DIR_NAME = "history_image_steward"
CACHE_FILENAME = "caption_cache.json"

DEFAULT_CAPTION_PROMPT = "请用一到两句话客观描述这张图片内容"

__all__ = [
    "HistoryImageStewardPlugin",
    "Settings",
    "Steward",
    "apply_payload_patch",
]

#: 当前生效的 Steward（补丁里按名字取，保证热重载后拿到的是新实例）
_ACTIVE: "Steward | None" = None


# ============================================================ 配置读取


def _conf_get(config: Any, key: str, default: Any) -> Any:
    """AstrBotConfig / dict / None 都能安全取值。"""
    if config is None:
        return default
    try:
        value = config.get(key, default)
    except Exception:
        return default
    return default if value is None else value


def _conf_bool(config: Any, key: str, default: bool) -> bool:
    value = _conf_get(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y")
    return bool(value)


def _conf_int(config: Any, key: str, default: int) -> int:
    try:
        return int(_conf_get(config, key, default))
    except (TypeError, ValueError):
        return default


def _conf_float(config: Any, key: str, default: float) -> float:
    try:
        return float(_conf_get(config, key, default))
    except (TypeError, ValueError):
        return default


def _conf_str(config: Any, key: str, default: str) -> str:
    value = _conf_get(config, key, default)
    if value is None:
        return default
    return str(value)


def _declares_image(provider: Any) -> bool:
    """provider 是否声明了 image 模态（用于自动挑选 caption 提供商）。"""
    config = getattr(provider, "provider_config", None)
    if not isinstance(config, dict):
        return False
    modalities = config.get("modalities")
    return isinstance(modalities, list) and "image" in modalities


@dataclass
class Settings:
    """_conf_schema.json 对应的运行时配置。"""

    enable: bool = True
    dry_run: bool = True
    persist_strip: bool = False
    keep_rounds: int = 2
    enable_caption: bool = True
    caption_provider_id: str = ""
    caption_prompt: str = DEFAULT_CAPTION_PROMPT
    caption_timeout_sec: float = 20.0
    caption_max_chars: int = 150
    cache_max_entries: int = 1000

    @classmethod
    def from_config(cls, config: Any) -> "Settings":
        return cls(
            enable=_conf_bool(config, "enable", True),
            dry_run=_conf_bool(config, "dry_run", True),
            persist_strip=_conf_bool(config, "persist_strip", False),
            keep_rounds=max(0, _conf_int(config, "image_keep_rounds", 2)),
            enable_caption=_conf_bool(config, "enable_caption", True),
            caption_provider_id=_conf_str(config, "caption_provider_id", "").strip(),
            caption_prompt=_conf_str(
                config, "caption_prompt", DEFAULT_CAPTION_PROMPT
            ).strip()
            or DEFAULT_CAPTION_PROMPT,
            caption_timeout_sec=max(
                1.0, _conf_float(config, "caption_timeout_sec", 20.0)
            ),
            caption_max_chars=_conf_int(config, "caption_max_chars", 150),
            cache_max_entries=max(0, _conf_int(config, "cache_max_entries", 1000)),
        )

    def describe(self) -> str:
        return (
            f"K={self.keep_rounds} dry_run={self.dry_run} "
            f"persist_strip={self.persist_strip} caption={self.enable_caption} "
            f"provider={self.caption_provider_id or 'auto(第一个含 image 模态)'} "
            f"timeout={self.caption_timeout_sec}s max_chars={self.caption_max_chars} "
            f"cache_max={self.cache_max_entries}"
        )


# ============================================================ 核心调度


class Steward:
    """配置 + 缓存 + 两层逻辑；与 Star 解耦，便于单测。"""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.settings = Settings()
        self.cache = core.CaptionCache()
        self._inflight: set[str] = set()
        self._caption_tasks: set[asyncio.Task] = set()
        self._save_task: asyncio.Task | None = None
        self._provider_warned = False
        self.totals = {
            "requests": 0,
            "stripped": 0,
            "captioned": 0,
            "uncaptioned": 0,
            "prewarmed": 0,
            "payload_strip_calls": 0,
        }

    # ------------------------------------------------------------- 配置
    def configure(self, config: Any) -> None:
        self.settings = Settings.from_config(config)
        self.cache = core.CaptionCache(
            path=self.cache_path(),
            max_entries=self.settings.cache_max_entries,
        )
        loaded = self.cache.load()
        logger.info(f"{LOG_TAG} 配置已加载: {self.settings.describe()}")
        logger.info(
            f"{LOG_TAG} caption 缓存: {self.cache.path} "
            f"(已载入 {loaded} 条)"
        )

    @staticmethod
    def cache_path() -> str | None:
        if get_astrbot_plugin_data_path is not None:
            try:
                base = get_astrbot_plugin_data_path()
                if base:
                    return os.path.join(base, DATA_DIR_NAME, CACHE_FILENAME)
            except Exception:
                pass
        try:
            return os.path.join(
                os.path.dirname(os.path.abspath(__file__)), CACHE_FILENAME
            )
        except Exception:
            return None

    # -------------------------------------------------- 第 1 层：请求钩子
    async def handle(self, event: Any, req: Any) -> None:
        """在请求发往模型之前运行。默认**不修改** req.contexts。"""
        settings = self.settings
        if not settings.enable:
            return

        umo = getattr(event, "unified_msg_origin", "") or ""

        # (a) 当前轮的图：后台预热 caption 缓存。
        #     dry_run 下完全不动，保证「只记日志」是零副作用的。
        if settings.enable_caption and not settings.dry_run:
            self._prewarm_current_images(req)

        contexts = getattr(req, "contexts", None)
        if not isinstance(contexts, list) or not contexts:
            return

        if settings.dry_run:
            stats = self._strip(contexts, keep_rounds=settings.keep_rounds, dry_run=True)
            if stats.stripped:
                logger.info(f"{LOG_TAG} DRY-RUN 本次将{stats.describe()} umo={umo}")
            else:
                logger.debug(f"{LOG_TAG} 无需剥离 umo={umo} {stats.describe()}")
            return

        if not settings.persist_strip:
            # 默认路径：真正的剥离在 _prepare_chat_payload（deepcopy 出来的请求体）上做，
            # 这里不碰 req.contexts，所以本轮结束时写回数据库的历史仍然是原样的图片。
            logger.debug(f"{LOG_TAG} 历史图剥离交给 provider 请求体层 umo={umo}")
            return

        # 显式开启 persist_strip：原地改写 req.contexts，剥离结果随本轮写库（不可逆）
        stats = self._strip(contexts, keep_rounds=settings.keep_rounds, dry_run=False)
        if stats.stripped:
            self._record(stats)
            logger.info(
                f"{LOG_TAG} 本次已{stats.describe()}"
                "（persist_strip=true，已随本轮写回会话历史）"
                f" umo={umo}"
            )

    # ------------------------------------------- 第 2 层：provider 请求体
    def strip_payload(self, payloads: Any, context_query: Any) -> None:
        """剥离本次 HTTP 请求体里的历史图。

        作用对象是 _prepare_chat_payload 内部 copy.deepcopy 出来的 messages，
        与会话历史、run_context.messages 没有任何共享引用，改完即弃。
        """
        settings = self.settings
        if not settings.enable or settings.dry_run:
            return

        messages = payloads.get("messages") if isinstance(payloads, dict) else None
        if not isinstance(messages, list):
            messages = context_query
        if not isinstance(messages, list) or not messages:
            return

        self.totals["payload_strip_calls"] += 1
        stats = self._strip(
            messages,
            keep_rounds=settings.keep_rounds,
            current_round_included=True,
            dry_run=False,
        )
        if stats.stripped:
            self._record(stats)
            logger.info(f"{LOG_TAG} 本次已{stats.describe()}（仅改请求体，未写库）")

        # 主路径：给「本轮保留在 K 窗口内」的图预热 caption。
        # 这些图就是将来滑出窗口时从历史里读出来的同一份字节，指纹必然对得上；
        # 钩子时刻的 req.image_urls 在 AstrBot 4.28.1 里并不可靠，只当补充来源。
        if settings.enable_caption and not _IN_CAPTION_CALL.get():
            self._prewarm_refs(stats.kept_refs, source="请求体保留窗口")

    def _strip(
        self,
        contexts: Any,
        *,
        keep_rounds: int,
        dry_run: bool,
        current_round_included: bool = False,
    ) -> core.StripStats:
        return core.strip_history_images(
            contexts,
            keep_rounds=keep_rounds,
            lookup=self.cache.get,
            enable_caption=self.settings.enable_caption,
            max_chars=self.settings.caption_max_chars,
            dry_run=dry_run,
            current_round_included=current_round_included,
        )

    def _record(self, stats: core.StripStats) -> None:
        self.totals["requests"] += 1
        self.totals["stripped"] += stats.stripped
        self.totals["captioned"] += stats.captioned
        self.totals["uncaptioned"] += stats.uncaptioned

    # ------------------------------------------------- 当前轮 caption 预热
    def _prewarm_current_images(self, req: Any) -> None:
        """兜底来源：钩子时刻的 req.image_urls。

        线上实测（2026-09-20）：AstrBot 4.28.1 里这个字段在钩子时刻常常还是空的，
        所以它只是补充；主路径在 strip_payload()（请求体保留窗口那一批）。
        无论有没有图都打一行 DEBUG —— 不允许再出现「静默什么都没做」。
        """
        refs = core.collect_image_refs(getattr(req, "image_urls", None))
        if not refs:
            logger.debug(f"{LOG_TAG} caption 预热[钩子 image_urls]：0 张，交给请求体层")
            return
        self._prewarm_refs(refs, source="钩子 image_urls")

    def _prewarm_refs(self, refs: Any, *, source: str) -> int:
        """给一批图片引用预热 caption，返回本次真正起任务的数量。

        每一类跳过原因都进汇总 DEBUG 行，杜绝静默失败。
        """
        candidates = remote = cached = unreadable = inflight = spawned = 0
        for ref in core.collect_image_refs(refs):
            candidates += 1
            if ref.startswith(("http://", "https://")):
                remote += 1
                continue
            key = core.image_ref_key(ref)
            if not key:
                unreadable += 1
                continue
            if key in self.cache:
                cached += 1
                continue
            if key in self._inflight:
                inflight += 1
                continue
            payload = core.materialize_ref(ref)
            if payload is None:
                unreadable += 1
                continue
            self._spawn_caption_task(key, payload)
            spawned += 1
        logger.debug(
            f"{LOG_TAG} caption 预热[{source}]：候选 {candidates} 张"
            f"（起任务 {spawned}，已在缓存 {cached}，在途 {inflight}，"
            f"远端跳过 {remote}，读不到 {unreadable}）"
        )
        return spawned

    def _spawn_caption_task(self, key: str, payload: str) -> None:
        self._inflight.add(key)
        try:
            # 先拿 loop 再建协程：没有运行中的事件循环时，不要留下一个
            # 从未 await 的协程对象（否则运行时会报 "was never awaited"）。
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._inflight.discard(key)
            return
        task = loop.create_task(self._caption_image(key, payload))
        self._caption_tasks.add(task)
        task.add_done_callback(self._caption_tasks.discard)
        self.totals["prewarmed"] += 1

    async def _caption_image(self, key: str, payload: str) -> None:
        try:
            provider = self._resolve_caption_provider()
            if provider is None:
                return
            # 标记：这次 provider 请求是我们自己发的，补丁层别再给它的图起 caption 任务
            token = _IN_CAPTION_CALL.set(True)
            try:
                response = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=self.settings.caption_prompt,
                        image_urls=[payload],
                    ),
                    timeout=self.settings.caption_timeout_sec,
                )
            finally:
                _IN_CAPTION_CALL.reset(token)
            caption = getattr(response, "completion_text", None) or ""
            if not caption.strip():
                logger.debug(f"{LOG_TAG} caption 返回为空，该图后续将落 [图片] 占位")
                return
            if self.cache.put(key, caption):
                self._schedule_save()
        except asyncio.TimeoutError:
            logger.warning(
                f"{LOG_TAG} caption 生成超时（{self.settings.caption_timeout_sec}s），"
                "该图后续将落 [图片] 占位"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 绝不让后台任务影响主流程
            logger.warning(f"{LOG_TAG} caption 生成失败: {exc}")
        finally:
            self._inflight.discard(key)

    def _resolve_caption_provider(self) -> Any:
        configured = (self.settings.caption_provider_id or "").strip()
        if configured:
            provider = None
            try:
                provider = self.context.get_provider_by_id(configured)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{LOG_TAG} 取 caption provider 失败: {exc}")
            if provider is None:
                self._warn_once(f"配置的 caption provider (id={configured}) 不存在")
                return None
            if not _declares_image(provider):
                self._warn_once(
                    f"配置的 caption provider (id={configured}) 未声明 image 模态，"
                    "仍按配置使用"
                )
            return provider

        try:
            providers = list(self.context.get_all_providers() or [])
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"获取提供商列表失败: {exc}")
            return None

        for provider in providers:
            if _declares_image(provider):
                self._provider_warned = False
                return provider

        self._warn_once(
            "没有找到声明 image 模态的提供商，caption 能力暂时关闭"
            "（历史图将落 [图片] 占位；可在配置里显式指定 caption_provider_id）"
        )
        return None

    def _warn_once(self, message: str) -> None:
        if self._provider_warned:
            return
        self._provider_warned = True
        logger.warning(f"{LOG_TAG} {message}")

    # --------------------------------------------------------- 缓存落盘
    def _schedule_save(self) -> None:
        if self._save_task is not None and not self._save_task.done():
            return
        try:
            self._save_task = asyncio.create_task(self._save_later())
        except RuntimeError:
            self._save_task = None

    async def _save_later(self) -> None:
        await asyncio.sleep(SAVE_DEBOUNCE_SEC)
        await self.flush()

    async def flush(self) -> bool:
        if not self.cache.path:
            return False
        try:
            saved = await asyncio.to_thread(self.cache.save)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{LOG_TAG} 写 caption 缓存失败: {exc}")
            return False
        if saved:
            logger.debug(f"{LOG_TAG} caption 缓存已落盘（{len(self.cache)} 条）")
        return saved

    async def shutdown(self) -> None:
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
        for task in list(self._caption_tasks):
            task.cancel()
        if self._save_task is not None and not self._save_task.done():
            self._save_task.cancel()
        await self.flush()


# ============================================================ provider 补丁


def apply_payload_patch(provider_cls: Any) -> str:
    """给 provider 的请求体组装方法打后置剥离补丁。

    每次都重新打（不因版本号相同而跳过）：插件热重载会产生新的模块对象，
    旧闭包读到的是旧模块的 _ACTIVE / 旧配置，必须让新闭包接管。

    Returns:
        'patched' | 'missing'
    """
    if provider_cls is None:
        return "missing"
    target = getattr(provider_cls, PATCH_TARGET, None)
    if target is None:
        return "missing"

    original = getattr(target, PATCH_ORIG_ATTR, None) or target

    @wraps(original)
    async def _prepare_chat_payload(self, *args, **kwargs):
        payloads, context_query = await original(self, *args, **kwargs)
        steward = _ACTIVE
        if steward is not None:
            try:
                steward.strip_payload(payloads, context_query)
            except Exception as exc:  # noqa: BLE001 - 绝不让插件本身弄挂请求
                logger.error(f"{LOG_TAG} 剥离请求体失败，已放行原始载荷: {exc}")
        return payloads, context_query

    setattr(_prepare_chat_payload, PATCH_ATTR, PATCH_VERSION)
    setattr(_prepare_chat_payload, PATCH_ORIG_ATTR, original)
    setattr(provider_cls, PATCH_TARGET, _prepare_chat_payload)
    return "patched"


# ============================================================ 插件入口


@register(PLUGIN_NAME, "NaE", DESCRIPTION, VERSION)
class HistoryImageStewardPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.context = context
        self.config = config
        self.steward = Steward(context)
        self.steward.configure(config)

        global _ACTIVE
        _ACTIVE = self.steward
        self._apply_patches(context)

    def _apply_patches(self, context: Context) -> None:
        results: dict[str, str] = {
            "ProviderOpenAIOfficial": apply_payload_patch(ProviderOpenAIOfficial)
        }

        try:
            providers = list(context.get_all_providers() or [])
        except Exception as exc:  # noqa: BLE001
            providers = []
            logger.warning(f"{LOG_TAG} 获取提供商列表失败: {exc}")

        for provider in providers:
            try:
                cls = type(provider)
                key = f"{cls.__module__}.{cls.__qualname__}"
                if key not in results:
                    results[key] = apply_payload_patch(cls)
            except Exception:
                continue

        logger.info(f"{LOG_TAG} 请求体剥离补丁: {results}")
        if results.get("ProviderOpenAIOfficial") == "missing":
            logger.error(
                f"{LOG_TAG} 没找到 {PATCH_TARGET}，剥离不会生效"
                "（当前主模型不是 OpenAI 系提供商）。"
                "可改用 persist_strip 模式，或把 enable 关掉。"
            )

    @filter.on_llm_request(priority=HOOK_PRIORITY)
    async def steward_llm_request(
        self, event: AstrMessageEvent, req: Any
    ) -> None:
        """请求发出前：预热当前轮图的 caption；dry_run 时统计并打日志。"""
        try:
            await self.steward.handle(event, req)
        except Exception as exc:  # noqa: BLE001 - 绝不让插件本身弄挂请求
            logger.error(
                f"{LOG_TAG} 处理请求失败，已放行原始请求: {exc}",
                exc_info=True,
            )

    async def terminate(self) -> None:
        try:
            await self.steward.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{LOG_TAG} 退出清理失败: {exc}")
