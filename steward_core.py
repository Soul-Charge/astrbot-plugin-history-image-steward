"""history_image_steward 的纯逻辑层。

这里刻意不导入任何 AstrBot 模块：边界计算、图片引用指纹、caption 缓存、
历史图片剥离全部是纯函数 / 纯数据类，可以脱离容器直接跑单元测试。

关键事实（AstrBot 4.28.1 实测）：

- 会话历史存在 conversations.content，是 OpenAI 格式的消息列表；
- 历史里的图片是内容块 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}；
- 当前轮的图片**不能**指望在 OnLLMRequestEvent 钩子里从 req.image_urls 读到：
  internal.py 里 `await prepare_request_images(...)` 就紧跟在钩子调用之后一行，
  引用图那条支路更是走 extra_user_content_parts。2026-09-20 线上实测：17 次请求
  322 张图全部剥离成功，但 caption 预热一次都没触发（因为那时 image_urls 还是空的）。
- 所以要预热 caption，只能从「请求体里被保留在 K 窗口内的那批图」出发 ——
  strip_history_images() 顺手把它们通过 StripStats.kept_refs 回传。
  它们将来滑出窗口时是从历史里读出来的同一份字节，指纹必然对得上。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

IMAGE_PLACEHOLDER = "[图片]"
CAPTION_TEMPLATE = "[图片：{caption}]"
DUP_PLACEHOLDER = "[同前图]"

_CACHE_VERSION = 1

__all__ = [
    "CAPTION_TEMPLATE",
    "DUP_PLACEHOLDER",
    "IMAGE_PLACEHOLDER",
    "CaptionCache",
    "StripStats",
    "collect_image_refs",
    "count_image_blocks",
    "find_keep_boundary",
    "image_ref_key",
    "iter_image_blocks",
    "materialize_ref",
    "resolve_file_url_path",
    "strip_history_images",
    "truncate_caption",
]


# =========================================================== 图片引用 -> 指纹


def resolve_file_url_path(image_ref: str) -> str:
    """把 file:// URL 还原成本地路径（与 AstrBot image_ref_utils 同语义）。"""
    if not image_ref.startswith("file://"):
        return image_ref

    parsed = urlparse(image_ref)
    path = unquote(parsed.path or "")
    netloc = unquote(parsed.netloc or "")

    if netloc and netloc.lower() != "localhost":
        path = f"//{netloc}{path}" if path else netloc
    elif not path and netloc:
        path = netloc

    if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]

    return path or image_ref


def _digest_b64_payload(payload: str) -> str:
    """对 base64 载荷本身做指纹。

    只看载荷、不看 data:<mime>;base64 前缀，因此同一张图无论被贴上
    image/jpeg 还是 image/png 都会落到同一个 key —— 这一步很关键：
    当前轮（caption 生成时）和若干轮后（落进历史时）的 MIME 前缀并不保证一致。

    归一化（去空白、URL-safe 字符还原、去 = 填充）对两侧一致生效，
    所以标准 base64 与变体 base64 也能对上。
    """
    compact = "".join(payload.split())
    compact = compact.replace("-", "+").replace("_", "/").rstrip("=")
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


def _digest_raw(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def image_ref_key(image_ref: Any) -> str | None:
    """把图片引用规范成「与载体形式无关」的内容指纹。

    支持 data: / base64:// / file:// / 本地路径；
    http(s):// 返回 None（本地拿不到内容，不值得为缓存去下载）。
    """
    if not isinstance(image_ref, str):
        return None

    text = image_ref.strip()
    if not text:
        return None

    if text.startswith("data:"):
        head, sep, payload = text.partition(",")
        if not sep:
            return None
        if ";base64" in head:
            return _digest_b64_payload(payload)
        # 非 base64 的 data URL（如内联 svg 明文）只能整体指纹
        return _digest_raw(text)

    if text.startswith("base64://"):
        return _digest_b64_payload(text[len("base64://") :])

    if text.startswith(("http://", "https://")):
        return None

    try:
        with open(resolve_file_url_path(text), "rb") as fh:
            data = fh.read()
    except OSError:
        return None

    if not data:
        return None
    return _digest_b64_payload(base64.b64encode(data).decode("ascii"))


def materialize_ref(image_ref: str) -> str | None:
    """把本地文件引用转成 data URL，避免后台 caption 任务读到已被清理的临时文件。

    data: / base64:// / http(s) 原样返回；本地路径读不到时返回 None。
    """
    if not isinstance(image_ref, str) or not image_ref.strip():
        return None

    text = image_ref.strip()
    if text.startswith(("data:", "base64://", "http://", "https://")):
        return text

    try:
        with open(resolve_file_url_path(text), "rb") as fh:
            data = fh.read()
    except OSError:
        return None

    if not data:
        return None
    # 与 AstrBot 原生 _encode_image_bs64 保持一致的 MIME 兜底写法
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


# =========================================================== 历史边界与遍历


def find_keep_boundary(contexts: Any, keep_rounds: int) -> int:
    """返回切割下标 cut。

    contexts[:cut] 是需要剥图的旧历史，contexts[cut:] 原样保留。
    深度按 user 消息计数（一轮 ≈ 一条 user 消息），tool/assistant 不计数。

    - keep_rounds <= 0 -> 历史图全剥（cut == len）；
    - 历史里的 user 消息不足 K 条 -> 不剥（cut == 0）。
    """
    if not isinstance(contexts, list):
        return 0
    if keep_rounds <= 0:
        return len(contexts)

    seen = 0
    for index in range(len(contexts) - 1, -1, -1):
        message = contexts[index]
        if isinstance(message, dict) and message.get("role") == "user":
            seen += 1
            if seen >= keep_rounds:
                return index
    return 0


def iter_image_blocks(
    contexts: Any,
    start: int = 0,
    end: int | None = None,
) -> Iterator[tuple[int, dict, str]]:
    """遍历区间内所有 image_url 内容块，产出 (消息下标, 内容块, url)。"""
    if not isinstance(contexts, list):
        return

    total = len(contexts)
    begin = max(0, start)
    stop = total if end is None else min(end, total)

    for index in range(begin, stop):
        message = contexts[index]
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            payload = block.get("image_url")
            if isinstance(payload, dict):
                url = payload.get("url")
            elif isinstance(payload, str):  # 兼容早期的字符串写法
                url = payload
            else:
                url = None
            if isinstance(url, str) and url:
                yield index, block, url


def count_image_blocks(contexts: Any, start: int = 0, end: int | None = None) -> int:
    return sum(1 for _ in iter_image_blocks(contexts, start, end))


def collect_image_refs(image_urls: Any) -> list[str]:
    """收集 ProviderRequest.image_urls 里的有效引用（当前轮的图）。"""
    if not isinstance(image_urls, (list, tuple)):
        return []
    return [
        item.strip()
        for item in image_urls
        if isinstance(item, str) and item.strip()
    ]


def replace_block_with_text(block: dict, text: str) -> None:
    """原地把图片块换成文本块：保留消息条数、顺序与所在下标。"""
    block.clear()
    block.update({"type": "text", "text": text})


def truncate_caption(text: str, max_chars: int) -> str:
    """按配置截断描述；max_chars <= 0 表示不截断。"""
    if not isinstance(text, str):
        return ""
    stripped = " ".join(text.split())
    if max_chars and max_chars > 0 and len(stripped) > max_chars:
        return stripped[:max_chars].rstrip() + "…"
    return stripped


# =========================================================== 剥离主流程


@dataclass
class StripStats:
    """一次剥离的账本，dry_run 与实跑共用同一套统计口径。"""

    cut_index: int = 0
    messages_total: int = 0
    old_messages: int = 0
    stripped: int = 0
    captioned: int = 0
    uncaptioned: int = 0
    # 同图去重折叠：旧历史区里指纹与前面某份完全相同的图块数量。
    deduped: int = 0
    kept: int = 0
    effective_rounds: int = 0
    # 保留窗口内的图片引用（原样，未修改），供调用方后台预热 caption。
    # 只在实跑时有意义；dry_run 下也会填，方便离线核对。
    kept_refs: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"剥离 {self.stripped} 张历史图"
            f"（{self.captioned} 张换成文字描述，{self.uncaptioned} 张落 "
            f"{IMAGE_PLACEHOLDER} 占位），保留最近窗口 {self.kept} 张"
            f"，边界下标 {self.cut_index}/{self.messages_total}"
            f"，去重折叠 {self.deduped} 张"
        )


def strip_history_images(
    contexts: Any,
    *,
    keep_rounds: int,
    lookup: Callable[[str], str | None] | None = None,
    enable_caption: bool = True,
    max_chars: int = 150,
    dry_run: bool = False,
    placeholder: str = IMAGE_PLACEHOLDER,
    current_round_included: bool = False,
) -> StripStats:
    """把「最近 K 轮之外」的历史图片替换成文字。

    lookup 接收 image_ref_key 的指纹，返回描述或 None。
    dry_run=True 时只统计、不修改任何内容块。
    旧历史区里指纹相同的图只保留第一份，其余折叠成 DUP_PLACEHOLDER（v1.1）。

    current_round_included=True 用于「列表里已经合并了当前这一轮 user 消息」的场景
    （例如 provider 组装请求体时看到的 messages）：此时按 K+1 条 user 消息计边界，
    这样保留窗口与纯历史场景完全一致，当前轮的图不会被误伤。
    """
    stats = StripStats()
    stats.messages_total = len(contexts) if isinstance(contexts, list) else 0

    effective_rounds = keep_rounds + 1 if current_round_included else keep_rounds
    stats.effective_rounds = effective_rounds

    cut = find_keep_boundary(contexts, effective_rounds)
    stats.cut_index = cut
    stats.old_messages = cut
    stats.kept = count_image_blocks(contexts, cut)
    stats.kept_refs = [url for _index, _block, url in iter_image_blocks(contexts, cut)]

    # 同图去重折叠：旧历史区里指纹相同的图只保留第一份（走 caption/占位），
    # 其余重复块换成 DUP_PLACEHOLDER，避免同一句 caption 在请求体里复读 5~9 次
    # 被模型放大。指纹算不出来的块（远端 URL 等）视为"每份都独特"，不参与去重。
    seen: set[str] = set()

    for _index, block, url in iter_image_blocks(contexts, 0, cut):
        # 去重需要无条件先算指纹（sha1，成本可忽略）；caption 分支复用同一个 key。
        key = image_ref_key(url)

        if key is not None and key in seen:
            stats.stripped += 1
            stats.deduped += 1
            if not dry_run:
                replace_block_with_text(block, DUP_PLACEHOLDER)
            continue
        if key is not None:
            seen.add(key)

        caption: str | None = None
        if enable_caption and lookup is not None and key:
            with suppress(Exception):
                caption = lookup(key)

        if caption and caption.strip():
            text = CAPTION_TEMPLATE.format(
                caption=truncate_caption(caption, max_chars)
            )
            stats.captioned += 1
        else:
            text = placeholder
            stats.uncaptioned += 1

        stats.stripped += 1
        if not dry_run:
            replace_block_with_text(block, text)

    return stats


# =========================================================== caption 缓存


@dataclass
class CaptionCache:
    """sha1(图片内容) -> caption 的 LRU 缓存，JSON 落盘。

    主路径只读内存；写入走 save()（原子替换），由调用方决定何时落盘。
    """

    path: str | None = None
    max_entries: int = 1000
    entries: OrderedDict = field(default_factory=OrderedDict)

    # ------------------------------------------------------------- 读
    def load(self) -> int:
        if not self.path or not os.path.exists(self.path):
            return 0
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:
            return 0

        payload = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(payload, dict):
            return 0

        self.entries = OrderedDict()
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                self.entries[key] = value
        self._shrink()
        return len(self.entries)

    def get(self, key: str | None) -> str | None:
        if not key:
            return None
        value = self.entries.get(key)
        if value is None:
            return None
        self.entries.move_to_end(key)
        return value

    def __contains__(self, key: object) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------- 写
    def put(self, key: str | None, caption: str | None) -> bool:
        if not key or not isinstance(caption, str) or not caption.strip():
            return False
        self.entries[key] = caption.strip()
        self.entries.move_to_end(key)
        self._shrink()
        return True

    def _shrink(self) -> None:
        limit = self.max_entries if self.max_entries and self.max_entries > 0 else 0
        if limit <= 0:
            return
        while len(self.entries) > limit:
            self.entries.popitem(last=False)

    def save(self) -> bool:
        """原子写盘：先写同目录临时文件，再 os.replace。"""
        if not self.path:
            return False

        directory = os.path.dirname(os.path.abspath(self.path)) or None
        try:
            if directory:
                os.makedirs(directory, exist_ok=True)
            payload = {"version": _CACHE_VERSION, "entries": dict(self.entries)}
            fd, tmp_path = tempfile.mkstemp(
                prefix=".caption_cache.", suffix=".tmp", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=0)
                os.replace(tmp_path, self.path)
            except BaseException:
                with suppress(OSError):
                    os.unlink(tmp_path)
                raise
            return True
        except Exception:
            return False
