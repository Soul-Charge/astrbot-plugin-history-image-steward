"""main.py 的集成测试：用假的 astrbot 模块把插件跑起来，全程不碰容器。

重点验证两件事：
1. 钩子层（@filter.on_llm_request）**绝不修改** req.contexts —— 因为
   run_context.messages 由 req.contexts 构造、并在本轮结束时整体写回 conversations 表，
   在钩子里原地改 req.contexts 会把剥离结果持久化进数据库（不可逆）。
2. 真正的剥离发生在 provider._prepare_chat_payload 产出的请求体（deepcopy）上，
   且保留窗口与纯历史场景完全一致（按 K+1 条 user 消息计边界）。
"""

from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
import json
import logging
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(HERE)
sys.path.insert(0, PLUGIN_DIR)


# ====================================================== 伪造 astrbot 模块树


class _FakeFilter:
    def on_llm_request(self, **kwargs):
        def decorator(func):
            func._hook_priority = kwargs.get("priority", 0)
            return func

        return decorator


class _FakeStar:
    def __init__(self, context=None, config=None) -> None:
        self.context = context


class ProviderOpenAIOfficial:
    """最小化模拟真实 provider 的请求体组装（含 deepcopy 语义）。"""

    def __init__(self) -> None:
        self.calls = 0

    async def _prepare_chat_payload(
        self,
        prompt=None,
        image_urls=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        **kwargs,
    ):
        self.calls += 1
        messages = copy.deepcopy(list(contexts or []))
        if prompt:
            messages.append({"role": "user", "content": prompt})
        payloads = {"messages": messages, "model": model or "fake"}
        return payloads, messages


def _install_fake_astrbot(data_path: str) -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot-test")
    api.event = types.ModuleType("astrbot.api.event")
    api.event.AstrMessageEvent = type("AstrMessageEvent", (), {})
    api.event.filter = _FakeFilter()
    api.star = types.ModuleType("astrbot.api.star")
    api.star.Context = type("Context", (), {})
    api.star.Star = _FakeStar
    api.star.register = lambda *args, **kwargs: (lambda cls: cls)

    core = types.ModuleType("astrbot.core")
    utils = types.ModuleType("astrbot.core.utils")
    path_mod = types.ModuleType("astrbot.core.utils.astrbot_path")
    path_mod.get_astrbot_plugin_data_path = lambda: data_path
    provider = types.ModuleType("astrbot.core.provider")
    sources = types.ModuleType("astrbot.core.provider.sources")
    oai = types.ModuleType("astrbot.core.provider.sources.openai_source")
    oai.ProviderOpenAIOfficial = ProviderOpenAIOfficial

    astrbot.api = api
    astrbot.core = core
    core.utils = utils
    core.provider = provider
    provider.sources = sources
    sources.openai_source = oai
    utils.astrbot_path = path_mod

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": api.event,
            "astrbot.api.star": api.star,
            "astrbot.core": core,
            "astrbot.core.utils": utils,
            "astrbot.core.utils.astrbot_path": path_mod,
            "astrbot.core.provider": provider,
            "astrbot.core.provider.sources": sources,
            "astrbot.core.provider.sources.openai_source": oai,
        }
    )


def _load_main(data_path: str):
    _install_fake_astrbot(data_path)
    spec = importlib.util.spec_from_file_location(
        "history_image_steward_main", os.path.join(PLUGIN_DIR, "main.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["history_image_steward_main"] = module
    spec.loader.exec_module(module)
    return module


# ====================================================== 测试替身


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class _FakeProvider:
    def __init__(self, provider_id: str, modalities: list, reply: str) -> None:
        self.provider_config = {"id": provider_id, "modalities": modalities}
        self.reply = reply
        self.calls = []
        self.fail = False

    async def text_chat(self, prompt=None, image_urls=None, **kwargs):
        self.calls.append({"prompt": prompt, "image_urls": list(image_urls or [])})
        if self.fail:
            raise RuntimeError("provider boom")
        return _FakeResponse(self.reply)


class _FakeContext:
    def __init__(self, providers) -> None:
        self.providers = providers

    def get_all_providers(self):
        return list(self.providers)

    def get_provider_by_id(self, provider_id: str):
        for provider in self.providers:
            if provider.provider_config.get("id") == provider_id:
                return provider
        return None


class _FakeEvent:
    unified_msg_origin = "bot:GroupMessage:200000000"


def _b64(text: bytes) -> str:
    return base64.b64encode(text).decode()


def _img(payload: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + payload},
    }


def _history(rounds: int) -> list:
    contexts = []
    for index in range(rounds):
        contexts.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "第%d轮" % index},
                    _img(_b64(("img-%d" % index).encode())),
                ],
            }
        )
        contexts.append({"role": "assistant", "content": "ok"})
    return contexts


async def _drain(steward, timeout: float = 5.0) -> None:
    for _ in range(50):
        pending = [task for task in steward._caption_tasks if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)
        save_task = steward._save_task
        if save_task is not None and not save_task.done():
            await asyncio.wait([save_task], timeout=timeout)
        await asyncio.sleep(0.01)
        if not steward._caption_tasks and steward._inflight == set():
            break


class StewardTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_path = self._tmp.name
        self.main = _load_main(self.data_path)
        self.main.SAVE_DEBOUNCE_SEC = 0.02
        self.main._ACTIVE = None
        self.provider = _FakeProvider(
            "deepseek-flash", ["text", "image"], "一只橘猫趴在键盘上"
        )
        self.other = _FakeProvider("text-only", ["text"], "不该被选中")
        self.context = _FakeContext([self.other, self.provider])

    def tearDown(self) -> None:
        self.main._ACTIVE = None
        self._tmp.cleanup()

    def _make(self, **overrides):
        config = {
            "enable": True,
            "dry_run": False,
            "persist_strip": False,
            "image_keep_rounds": 2,
            "enable_caption": True,
            "caption_provider_id": "",
            "caption_prompt": "描述这张图",
            "caption_timeout_sec": 5,
            "caption_max_chars": 150,
            "cache_max_entries": 100,
        }
        config.update(overrides)
        steward = self.main.Steward(self.context)
        steward.configure(config)
        return steward, config

    async def _flow(self, steward, req) -> None:
        await steward.handle(_FakeEvent(), req)
        await _drain(steward)


class SettingsTest(StewardTestBase):
    def test_from_config_coercions(self) -> None:
        settings = self.main.Settings.from_config(
            {
                "enable": "false",
                "dry_run": "true",
                "persist_strip": "yes",
                "image_keep_rounds": "3",
                "enable_caption": 0,
                "caption_provider_id": "  deepseek-flash  ",
                "caption_timeout_sec": "7",
                "cache_max_entries": -1,
                "caption_prompt": "   ",
            }
        )
        self.assertFalse(settings.enable)
        self.assertTrue(settings.dry_run)
        self.assertTrue(settings.persist_strip)
        self.assertEqual(settings.keep_rounds, 3)
        self.assertFalse(settings.enable_caption)
        self.assertEqual(settings.caption_provider_id, "deepseek-flash")
        self.assertEqual(settings.caption_timeout_sec, 7.0)
        self.assertEqual(settings.cache_max_entries, 0)
        self.assertEqual(settings.caption_prompt, self.main.DEFAULT_CAPTION_PROMPT)

    def test_defaults(self) -> None:
        settings = self.main.Settings.from_config(None)
        self.assertTrue(settings.enable)
        self.assertTrue(settings.dry_run)
        self.assertFalse(settings.persist_strip)
        self.assertEqual(settings.keep_rounds, 2)
        self.assertEqual(settings.caption_max_chars, 150)


class HookTest(StewardTestBase):
    def test_disabled_does_nothing(self) -> None:
        steward, _ = self._make(enable=False)
        contexts = _history(4)
        before = json.dumps(contexts, sort_keys=True)
        asyncio.run(
            steward.handle(_FakeEvent(), types.SimpleNamespace(contexts=contexts))
        )
        self.assertEqual(json.dumps(contexts, sort_keys=True), before)
        self.assertEqual(self.provider.calls, [])

    def test_dry_run_is_completely_side_effect_free(self) -> None:
        steward, _ = self._make(dry_run=True)
        contexts = _history(4)
        before = json.dumps(contexts, sort_keys=True)
        req = types.SimpleNamespace(
            contexts=contexts, image_urls=["data:image/jpeg;base64," + _b64(b"cur")]
        )
        asyncio.run(steward.handle(_FakeEvent(), req))
        self.assertEqual(json.dumps(contexts, sort_keys=True), before)
        self.assertEqual(self.provider.calls, [])

    def test_live_hook_never_mutates_contexts(self) -> None:
        """核心回归测试：默认路径下钩子必须对 req.contexts 零改写。

        这条挂了就意味着剥离会随 _save_to_history 写回 conversations 表，
        DB 里的历史图片会被永久替换成文字。
        """
        steward, _ = self._make(persist_strip=False)
        contexts = _history(6)
        before = json.dumps(contexts, sort_keys=True)
        req = types.SimpleNamespace(
            contexts=contexts, image_urls=["data:image/jpeg;base64," + _b64(b"cur")]
        )
        asyncio.run(self._flow(steward, req))
        self.assertEqual(json.dumps(contexts, sort_keys=True), before)
        self.assertIn("data:image/jpeg;base64", json.dumps(contexts))

    def test_persist_strip_optin_does_mutate_contexts(self) -> None:
        steward, _ = self._make(persist_strip=True, image_keep_rounds=1)
        contexts = _history(3)
        req = types.SimpleNamespace(contexts=contexts, image_urls=[])
        asyncio.run(self._flow(steward, req))
        self.assertEqual(contexts[0]["content"][1]["type"], "text")
        self.assertEqual(contexts[4]["content"][1]["type"], "image_url")
        self.assertEqual(steward.totals["stripped"], 2)

    def test_prewarm_generates_and_persists_caption(self) -> None:
        steward, _ = self._make()
        current = _b64(b"brand-new-image")
        req = types.SimpleNamespace(
            contexts=[], image_urls=["data:image/jpeg;base64," + current]
        )
        asyncio.run(self._flow(steward, req))

        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.provider.calls[0]["prompt"], "描述这张图")
        key = self.main.core.image_ref_key("data:image/jpeg;base64," + current)
        self.assertEqual(steward.cache.get(key), "一只橘猫趴在键盘上")
        cache_file = os.path.join(
            self.data_path, "history_image_steward", "caption_cache.json"
        )
        with open(cache_file, encoding="utf-8") as fh:
            self.assertEqual(
                json.load(fh)["entries"][key], "一只橘猫趴在键盘上"
            )

    def test_caption_provider_failure_is_silent(self) -> None:
        steward, _ = self._make()
        self.provider.fail = True
        req = types.SimpleNamespace(
            contexts=[], image_urls=["data:image/jpeg;base64," + _b64(b"x")]
        )
        asyncio.run(self._flow(steward, req))
        self.assertEqual(len(steward.cache), 0)

    def test_explicit_provider_id_is_honoured(self) -> None:
        steward, _ = self._make(caption_provider_id="text-only")
        req = types.SimpleNamespace(
            contexts=[], image_urls=["data:image/jpeg;base64," + _b64(b"x")]
        )
        asyncio.run(self._flow(steward, req))
        self.assertEqual(len(self.other.calls), 1)
        self.assertEqual(self.provider.calls, [])

    def test_no_image_provider_is_survivable(self) -> None:
        steward = self.main.Steward(_FakeContext([self.other]))
        steward.configure({"enable": True, "dry_run": False})
        req = types.SimpleNamespace(
            contexts=[], image_urls=["data:image/jpeg;base64," + _b64(b"x")]
        )
        asyncio.run(self._flow(steward, req))
        self.assertEqual(len(steward.cache), 0)

    def test_contexts_none_is_survivable(self) -> None:
        steward, _ = self._make()
        asyncio.run(
            self._flow(steward, types.SimpleNamespace(contexts=None, image_urls=None))
        )
        self.assertEqual(steward.totals["stripped"], 0)


class PayloadStripTest(StewardTestBase):
    def test_payload_keeps_current_round_and_drops_older(self) -> None:
        steward, _ = self._make(image_keep_rounds=2)
        messages = copy.deepcopy(_history(4))
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "这轮的图"}, _img(_b64(b"cur"))],
            }
        )
        steward.strip_payload({"messages": messages}, messages)
        self.assertEqual(messages[0]["content"][1]["type"], "text")
        self.assertEqual(messages[2]["content"][1]["type"], "text")
        self.assertEqual(messages[4]["content"][1]["type"], "image_url")
        self.assertEqual(messages[6]["content"][1]["type"], "image_url")
        self.assertEqual(messages[8]["content"][1]["type"], "image_url")
        self.assertEqual(len(messages), 9)

    def test_payload_keep_zero_still_keeps_current_round(self) -> None:
        steward, _ = self._make(image_keep_rounds=0)
        messages = copy.deepcopy(_history(2))
        messages.append({"role": "user", "content": [_img(_b64(b"cur"))]})
        steward.strip_payload({"messages": messages}, messages)
        self.assertEqual(messages[0]["content"][1]["type"], "text")
        self.assertEqual(messages[2]["content"][1]["type"], "text")
        self.assertEqual(messages[4]["content"][0]["type"], "image_url")

    def test_boundary_matches_hook_statistics(self) -> None:
        """钩子口径（纯历史，K）与 provider 口径（含当前轮，K+1）必须一致。"""
        steward, _ = self._make(image_keep_rounds=2)
        history = _history(5)
        hook_stats = steward._strip(history, keep_rounds=2, dry_run=True)

        messages = copy.deepcopy(history)
        messages.append({"role": "user", "content": [_img(_b64(b"cur"))]})
        steward.strip_payload({"messages": messages}, messages)

        stripped_after = 0
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "text" and part.get("text") == "[图片]":
                    stripped_after += 1
        self.assertEqual(hook_stats.stripped, stripped_after)
        self.assertEqual(hook_stats.kept, 2)

    def test_payload_uses_cached_caption(self) -> None:
        steward, _ = self._make(image_keep_rounds=1)
        payload = _b64(b"cached-image")
        steward.cache.put(
            self.main.core.image_ref_key("data:image/jpeg;base64," + payload),
            "一只橘猫",
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "看"}, _img(payload)]},
            {"role": "assistant", "content": "嗯"},
            {"role": "user", "content": [{"type": "text", "text": "中间"}, _img(_b64(b"mid"))]},
            {"role": "assistant", "content": "嗯"},
            {"role": "user", "content": [{"type": "text", "text": "新的"}, _img(_b64(b"new"))]},
        ]
        steward.strip_payload({"messages": messages}, messages)
        self.assertEqual(
            messages[0]["content"][1], {"type": "text", "text": "[图片：一只橘猫]"}
        )
        self.assertEqual(messages[2]["content"][1]["type"], "image_url")
        self.assertEqual(messages[4]["content"][1]["type"], "image_url")

    def test_prewarmed_caption_is_used_when_image_ages_out(self) -> None:
        steward, _ = self._make(image_keep_rounds=1)
        current = _b64(b"aging-image")
        req = types.SimpleNamespace(
            contexts=[], image_urls=["data:image/jpeg;base64," + current]
        )
        asyncio.run(self._flow(steward, req))
        self.assertEqual(steward.totals["prewarmed"], 1)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "看"}, _img(current)]},
            {"role": "assistant", "content": "嗯"},
            {"role": "user", "content": [{"type": "text", "text": "中间"}, _img(_b64(b"mid"))]},
            {"role": "assistant", "content": "嗯"},
            {"role": "user", "content": [{"type": "text", "text": "新的"}, _img(_b64(b"n"))]},
        ]
        steward.strip_payload({"messages": messages}, messages)
        self.assertEqual(
            messages[0]["content"][1],
            {"type": "text", "text": "[图片：一只橘猫趴在键盘上]"},
        )

    def test_payload_not_touched_in_dry_run(self) -> None:
        steward, _ = self._make(dry_run=True)
        messages = _history(3)
        before = json.dumps(messages, sort_keys=True)
        steward.strip_payload({"messages": messages}, messages)
        self.assertEqual(json.dumps(messages, sort_keys=True), before)

    def test_payload_falls_back_to_context_query(self) -> None:
        steward, _ = self._make(image_keep_rounds=0)
        messages = _history(2)
        steward.strip_payload(None, messages)
        self.assertEqual(messages[0]["content"][1]["type"], "text")


    def test_payload_prewarms_the_kept_window(self) -> None:
        """v1.0.1 主路径：预热「请求体里保留下来的那批图」。

        线上 bug（2026-09-20）：旧实现从钩子时刻的 req.image_urls 预热，
        而 AstrBot 4.28.1 在那里往往还是空的 —— 17 次请求 322 张图全剥了，
        caption 任务一次都没起来，全部落成 [图片] 占位。
        """
        steward, _ = self._make(image_keep_rounds=1)
        messages = copy.deepcopy(_history(3))
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "这轮"}, _img(_b64(b"cur"))],
            }
        )

        async def scenario() -> None:
            steward.strip_payload({"messages": messages}, messages)
            await _drain(steward)

        asyncio.run(scenario())

        def _key(raw: bytes) -> str:
            return self.main.core.image_ref_key("data:image/jpeg;base64," + _b64(raw))

        # 窗口内（img-2 与当前轮的 cur）被预热；窗口外的 img-0/img-1 不该被调用
        self.assertEqual(steward.cache.get(_key(b"img-2")), "一只橘猫趴在键盘上")
        self.assertEqual(steward.cache.get(_key(b"cur")), "一只橘猫趴在键盘上")
        self.assertIsNone(steward.cache.get(_key(b"img-0")))
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(steward.totals["prewarmed"], 2)
        # 旧图照样被剥成占位
        self.assertEqual(messages[0]["content"][1]["type"], "text")

    def test_payload_prewarm_pays_off_next_round(self) -> None:
        """端到端：第 N 轮预热的描述，第 N+1 轮那张图滑出窗口时直接可用。"""
        steward, _ = self._make(image_keep_rounds=1)
        ref = "data:image/jpeg;base64," + _b64(b"aging")

        round1 = [
            {"role": "user", "content": [{"type": "text", "text": "看"}, _img(_b64(b"aging"))]},
            {"role": "assistant", "content": "嗯"},
            {"role": "user", "content": [{"type": "text", "text": "后来"}, _img(_b64(b"late"))]},
        ]

        async def scenario() -> None:
            steward.strip_payload({"messages": round1}, round1)
            await _drain(steward)

        asyncio.run(scenario())
        self.assertEqual(
            steward.cache.get(self.main.core.image_ref_key(ref)),
            "一只橘猫趴在键盘上",
        )

        # 下一轮：同一份字节滑出窗口 -> 必须换成预热好的文字描述，而不是 [图片]
        round2 = [
            {"role": "user", "content": [{"type": "text", "text": "看"}, _img(_b64(b"aging"))]},
            {"role": "assistant", "content": "嗯"},
            {"role": "user", "content": [{"type": "text", "text": "中间"}, _img(_b64(b"mid"))]},
            {"role": "assistant", "content": "嗯"},
            {"role": "user", "content": [{"type": "text", "text": "新的"}, _img(_b64(b"new"))]},
        ]
        steward.strip_payload({"messages": round2}, round2)
        self.assertEqual(
            round2[0]["content"][1],
            {"type": "text", "text": "[图片：一只橘猫趴在键盘上]"},
        )
        self.assertEqual(round2[2]["content"][1]["type"], "image_url")

    def test_prewarm_stats_are_logged_not_silent(self) -> None:
        """预热必须留下可核对的汇总行，不允许静默失败。"""
        steward, _ = self._make(image_keep_rounds=1)
        with self.assertLogs("astrbot-test", level="DEBUG") as captured:
            steward._prewarm_refs(["https://cdn.example.com/a.png"], source="单测")
        self.assertTrue(
            any("caption 预热[单测]" in line for line in captured.output),
            captured.output,
        )
        self.assertEqual(steward.totals["prewarmed"], 0)


class PatchTest(StewardTestBase):
    def test_patch_wraps_and_strips_only_the_payload(self) -> None:
        steward, _ = self._make(image_keep_rounds=2)
        self.main._ACTIVE = steward
        self.assertEqual(
            self.main.apply_payload_patch(ProviderOpenAIOfficial), "patched"
        )

        provider = ProviderOpenAIOfficial()
        history = _history(4)
        original = json.dumps(history, sort_keys=True)
        payloads, query = asyncio.run(
            provider._prepare_chat_payload(
                prompt="这轮的图",
                image_urls=["data:image/jpeg;base64," + _b64(b"cur")],
                contexts=history,
            )
        )

        self.assertEqual(json.dumps(history, sort_keys=True), original)
        sent = payloads["messages"]
        self.assertEqual(sent[0]["content"][1]["type"], "text")
        self.assertEqual(sent[2]["content"][1]["type"], "text")
        self.assertEqual(sent[4]["content"][1]["type"], "image_url")
        self.assertEqual(sent[6]["content"][1]["type"], "image_url")
        self.assertEqual(sent[-1]["role"], "user")
        self.assertEqual(sent[-1]["content"], "这轮的图")
        self.assertIs(sent, query)

    def test_patch_is_reapplied_and_keeps_single_wrapper(self) -> None:
        steward, _ = self._make(image_keep_rounds=2)
        self.main._ACTIVE = steward
        for _ in range(3):
            self.main.apply_payload_patch(ProviderOpenAIOfficial)

        wrapper = ProviderOpenAIOfficial._prepare_chat_payload
        self.assertTrue(hasattr(wrapper, "_his_payload_patch"))
        # 打三次补丁也必须只包一层：内层是未被打过补丁的原始函数
        self.assertFalse(hasattr(wrapper._his_payload_original, "_his_payload_patch"))

        provider = ProviderOpenAIOfficial()
        messages = _history(3)
        before = json.dumps(messages, sort_keys=True)
        asyncio.run(provider._prepare_chat_payload(contexts=messages))
        self.assertEqual(json.dumps(messages, sort_keys=True), before)
        self.assertEqual(provider.calls, 1)

    def test_patch_without_target_is_missing(self) -> None:
        self.assertEqual(self.main.apply_payload_patch(None), "missing")

        class NoTarget:
            pass

        self.assertEqual(self.main.apply_payload_patch(NoTarget), "missing")

    def test_active_none_makes_patch_a_noop(self) -> None:
        self.main.apply_payload_patch(ProviderOpenAIOfficial)
        self.main._ACTIVE = None
        provider = ProviderOpenAIOfficial()
        messages = _history(3)
        before = json.dumps(messages, sort_keys=True)
        payloads, _ = asyncio.run(provider._prepare_chat_payload(contexts=messages))
        self.assertEqual(json.dumps(payloads["messages"], sort_keys=True), before)

    def test_plugin_init_applies_patch_and_sets_active(self) -> None:
        plugin = self.main.HistoryImageStewardPlugin(
            self.context, {"dry_run": False, "image_keep_rounds": 1}
        )
        self.assertIs(self.main._ACTIVE, plugin.steward)
        self.assertEqual(
            getattr(
                ProviderOpenAIOfficial._prepare_chat_payload,
                "_his_payload_patch",
                None,
            ),
            self.main.PATCH_VERSION,
        )
        self.assertIsNotNone(
            getattr(plugin.steward_llm_request, "_hook_priority", None)
        )
        self.assertLess(plugin.steward_llm_request._hook_priority, 1000)


class CaptionSelfLoopTest(StewardTestBase):
    """自环保护：caption 调用自己也会经过 provider → 补丁，绝不能再生 caption。"""

    def test_caption_call_marks_context_var(self) -> None:
        steward, _ = self._make()
        seen = []
        original = self.provider.text_chat

        async def spy(prompt=None, image_urls=None, **kwargs):
            seen.append(self.main._IN_CAPTION_CALL.get())
            return await original(prompt=prompt, image_urls=image_urls, **kwargs)

        self.provider.text_chat = spy
        req = types.SimpleNamespace(
            contexts=[], image_urls=["data:image/jpeg;base64," + _b64(b"x")]
        )
        asyncio.run(self._flow(steward, req))

        self.assertEqual(seen, [True])
        # 用完必须复位，不能污染后续请求
        self.assertFalse(self.main._IN_CAPTION_CALL.get())

    def test_strip_payload_skips_prewarm_inside_caption_call(self) -> None:
        steward, _ = self._make(image_keep_rounds=1)

        def _payload(seed: bytes) -> list:
            messages = copy.deepcopy(_history(3))
            messages.append({"role": "user", "content": [_img(_b64(seed))]})
            return messages

        async def scenario(messages) -> None:
            steward.strip_payload({"messages": messages}, messages)
            await _drain(steward)

        asyncio.run(scenario(_payload(b"cap-1")))
        baseline = len(self.provider.calls)
        self.assertEqual(baseline, 2)  # 正常情况：窗口内 2 张会被预热

        token = self.main._IN_CAPTION_CALL.set(True)
        try:
            asyncio.run(scenario(_payload(b"cap-2")))
        finally:
            self.main._IN_CAPTION_CALL.reset(token)

        self.assertEqual(len(self.provider.calls), baseline)


class LifecycleTest(StewardTestBase):
    def test_shutdown_flushes_cache_and_clears_active(self) -> None:
        steward, _ = self._make()
        self.main._ACTIVE = steward
        steward.cache.put("k", "v")
        asyncio.run(steward.shutdown())
        self.assertIsNone(self.main._ACTIVE)
        cache_file = os.path.join(
            self.data_path, "history_image_steward", "caption_cache.json"
        )
        with open(cache_file, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["entries"]["k"], "v")


if __name__ == "__main__":
    unittest.main(verbosity=2)
