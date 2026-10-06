"""steward_core 的单元测试：纯逻辑，不依赖 AstrBot。

跑法（WSL 或容器内都行）：
    python3 -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import steward_core as core  # noqa: E402


def _img_block(payload: str, mime: str = "image/jpeg") -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{payload}"},
    }


def _url_block(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _history(payloads: list[str]) -> list[dict]:
    """构造一段形如 AstrBot 真实历史的 messages 列表（每轮 user + assistant）。"""
    contexts: list[dict] = []
    for index, payload in enumerate(payloads):
        contexts.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"第{index}轮说的话"},
                    _img_block(payload),
                ],
            }
        )
        contexts.append({"role": "assistant", "content": f"第{index}轮的回话"})
    return contexts


class TestBoundary(unittest.TestCase):
    def test_keep_two_rounds(self) -> None:
        contexts = _history(["A", "B", "C", "D", "E"])
        cut = core.find_keep_boundary(contexts, 2)
        # 倒数第 2 条 user 消息（第 4 轮 / 下标 6）是边界
        self.assertEqual(contexts[cut]["content"][0]["text"], "第3轮说的话")
        self.assertEqual(core.count_image_blocks(contexts, 0, cut), 3)
        self.assertEqual(core.count_image_blocks(contexts, cut), 2)

    def test_keep_zero_strips_everything(self) -> None:
        contexts = _history(["A", "B"])
        self.assertEqual(core.find_keep_boundary(contexts, 0), len(contexts))
        self.assertEqual(core.find_keep_boundary(contexts, -5), len(contexts))

    def test_not_enough_rounds_keeps_all(self) -> None:
        contexts = _history(["A", "B"])
        self.assertEqual(core.find_keep_boundary(contexts, 5), 0)

    def test_assistant_and_tool_do_not_count(self) -> None:
        contexts = [
            {"role": "user", "content": "u1"},
            {"role": "tool", "content": "t1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        cut = core.find_keep_boundary(contexts, 1)
        self.assertEqual(cut, 3)

    def test_non_list_is_safe(self) -> None:
        self.assertEqual(core.find_keep_boundary(None, 2), 0)
        self.assertEqual(core.find_keep_boundary("nope", 2), 0)


class TestImageRefKey(unittest.TestCase):
    def test_data_url_payload_is_mime_independent(self) -> None:
        payload = base64.b64encode(b"hello-image").decode()
        a = core.image_ref_key(f"data:image/jpeg;base64,{payload}")
        b = core.image_ref_key(f"data:image/png;base64,{payload}")
        self.assertIsNotNone(a)
        self.assertEqual(a, b)

    def test_base64_scheme_matches_data_url(self) -> None:
        payload = base64.b64encode(b"hello-image").decode()
        self.assertEqual(
            core.image_ref_key(f"base64://{payload}"),
            core.image_ref_key(f"data:image/jpeg;base64,{payload}"),
        )

    def test_urlsafe_and_padding_variants_match(self) -> None:
        raw = bytes(range(256))
        payload = base64.b64encode(raw).decode()
        urlsafe = base64.urlsafe_b64encode(raw).decode()
        self.assertEqual(
            core.image_ref_key(f"data:image/jpeg;base64,{payload}"),
            core.image_ref_key(f"data:image/jpeg;base64,{urlsafe.rstrip(chr(61))}"),
        )

    def test_local_file_matches_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pic.jpg")
            blob = b"\xff\xd8\xff-fake-jpeg-bytes"
            with open(path, "wb") as fh:
                fh.write(blob)
            expected = core.image_ref_key(
                "data:image/jpeg;base64," + base64.b64encode(blob).decode()
            )
            self.assertEqual(core.image_ref_key(path), expected)
            self.assertEqual(core.image_ref_key("file://" + path), expected)

    def test_http_and_missing_are_none(self) -> None:
        self.assertIsNone(core.image_ref_key("https://example.com/a.png"))
        self.assertIsNone(core.image_ref_key("/no/such/file.png"))
        self.assertIsNone(core.image_ref_key(""))
        self.assertIsNone(core.image_ref_key(None))

    def test_materialize_local_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pic.jpg")
            with open(path, "wb") as fh:
                fh.write(b"abc")
            out = core.materialize_ref(path)
            self.assertTrue(out.startswith("data:image/jpeg;base64,"))
            self.assertEqual(out.split(",", 1)[1], base64.b64encode(b"abc").decode())
        self.assertIsNone(core.materialize_ref("/no/such/file.jpg"))
        self.assertEqual(
            core.materialize_ref("data:image/png;base64,AAAA"),
            "data:image/png;base64,AAAA",
        )


class TestStrip(unittest.TestCase):
    def test_dry_run_does_not_touch_history(self) -> None:
        contexts = _history(["A", "B", "C"])
        snapshot = json.dumps(contexts, sort_keys=True)
        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: "一只猫", dry_run=True
        )
        self.assertEqual(stats.stripped, 2)
        self.assertEqual(stats.captioned, 2)
        self.assertEqual(stats.kept, 1)
        self.assertEqual(json.dumps(contexts, sort_keys=True), snapshot)

    def test_strip_replaces_old_images_only(self) -> None:
        contexts = _history(["A", "B", "C"])
        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: "一只猫"
        )
        self.assertEqual(stats.stripped, 2)
        self.assertEqual(stats.uncaptioned, 0)
        first = contexts[0]["content"]
        self.assertEqual(first[1], {"type": "text", "text": "[图片：一只猫]"})
        # 最后一段（最近 1 轮）的图原样保留
        last_user = contexts[4]["content"][1]
        self.assertEqual(last_user["type"], "image_url")
        # 消息条数不变
        self.assertEqual(len(contexts), 6)

    def test_missing_caption_falls_back_to_placeholder(self) -> None:
        contexts = _history(["A", "B"])
        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: None
        )
        self.assertEqual(stats.stripped, 1)
        self.assertEqual(stats.uncaptioned, 1)
        self.assertEqual(
            contexts[0]["content"][1], {"type": "text", "text": "[图片]"}
        )

    def test_caption_disabled_uses_placeholder(self) -> None:
        contexts = _history(["A", "B"])
        stats = core.strip_history_images(
            contexts,
            keep_rounds=1,
            lookup=lambda key: "一只猫",
            enable_caption=False,
        )
        self.assertEqual(stats.uncaptioned, 1)
        self.assertEqual(
            contexts[0]["content"][1], {"type": "text", "text": "[图片]"}
        )

    def test_image_only_message_stays_non_empty(self) -> None:
        contexts = [
            {"role": "user", "content": [_img_block("AAAA")]},
            {"role": "assistant", "content": "嗯"},
            {"role": "user", "content": [_img_block("BBBB")]},
            {"role": "assistant", "content": "好"},
        ]
        core.strip_history_images(contexts, keep_rounds=1, lookup=lambda key: None)
        content = contexts[0]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0], {"type": "text", "text": "[图片]"})
        self.assertTrue(content[0]["text"].strip())

    def test_caption_truncated(self) -> None:
        contexts = _history(["A", "B"])
        long_caption = "描" * 400
        core.strip_history_images(
            contexts,
            keep_rounds=1,
            lookup=lambda key: long_caption,
            max_chars=10,
        )
        text = contexts[0]["content"][1]["text"]
        self.assertEqual(text, "[图片：" + "描" * 10 + "…]")

    def test_lookup_receives_content_key(self) -> None:
        payload = base64.b64encode(b"keyed").decode()
        contexts = _history([payload, payload])
        seen: list[str] = []
        core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: seen.append(key) or "x"
        )
        self.assertEqual(seen, [core.image_ref_key(f"data:image/jpeg;base64,{payload}")])

    def test_volume_drops_like_the_real_incident(self) -> None:
        """还原 200000000 的形态：95 轮 × 3 张 ~1MB 图，剥完只剩最近 K 轮。"""
        big = base64.b64encode(b"x" * 300_000).decode()  # 每张 ~400KB 的 base64
        contexts: list[dict] = []
        for index in range(95):
            contexts.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"第{index}轮"},
                        _img_block(big),
                        _img_block(big),
                        _img_block(big),
                    ],
                }
            )
            contexts.append({"role": "assistant", "content": "ok"})

        before = len(json.dumps(contexts))
        stats = core.strip_history_images(
            contexts, keep_rounds=2, lookup=lambda key: None
        )
        after = len(json.dumps(contexts))
        self.assertEqual(stats.stripped, 93 * 3)
        self.assertEqual(stats.kept, 2 * 3)
        self.assertLess(after, before / 20)


class TestCaptionCache(unittest.TestCase):
    def test_lru_eviction(self) -> None:
        cache = core.CaptionCache(max_entries=2)
        cache.put("a", "1")
        cache.put("b", "2")
        self.assertEqual(cache.get("a"), "1")  # a 变成最近使用
        cache.put("c", "3")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "1")
        self.assertEqual(cache.get("c"), "3")

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "caption_cache.json")
            cache = core.CaptionCache(path=path, max_entries=10)
            cache.put("k1", "一只猫在窗台上")
            cache.put("k2", "一张风景照")
            self.assertTrue(cache.save())
            self.assertTrue(os.path.exists(path))

            reloaded = core.CaptionCache(path=path, max_entries=10)
            self.assertEqual(reloaded.load(), 2)
            self.assertEqual(reloaded.get("k1"), "一只猫在窗台上")
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["version"], 1)

    def test_load_tolerates_junk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "caption_cache.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{ not json")
            self.assertEqual(core.CaptionCache(path=path).load(), 0)

            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"entries": {"a": "x", "b": "", "c": None}}, fh)
            cache = core.CaptionCache(path=path)
            self.assertEqual(cache.load(), 1)
            self.assertEqual(cache.get("a"), "x")

    def test_put_rejects_empty(self) -> None:
        cache = core.CaptionCache()
        self.assertFalse(cache.put("k", ""))
        self.assertFalse(cache.put(None, "x"))
        self.assertFalse(cache.put("k", "   "))
        self.assertEqual(len(cache), 0)

    def test_save_without_path_is_false(self) -> None:
        self.assertFalse(core.CaptionCache(path=None).save())


class TestHelpers(unittest.TestCase):
    def test_truncate_caption_collapses_whitespace(self) -> None:
        self.assertEqual(core.truncate_caption("a\n\nb   c", 0), "a b c")
        self.assertEqual(core.truncate_caption("abcdef", 3), "abc…")
        self.assertEqual(core.truncate_caption("abc", 3), "abc")

    def test_collect_image_refs(self) -> None:
        self.assertEqual(
            core.collect_image_refs(["a", "", None, 3, " b "]), ["a", "b"]
        )
        self.assertEqual(core.collect_image_refs(None), [])

    def test_iter_image_blocks_tolerates_string_payload(self) -> None:
        contexts = [{"role": "user", "content": [{"type": "image_url", "image_url": "http://x/1.png"}]}]
        found = list(core.iter_image_blocks(contexts))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][2], "http://x/1.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestKeptRefs(unittest.TestCase):
    """kept_refs 是 v1.0.1 caption 预热主路径的输入。

    线上事实：OnLLMRequestEvent 钩子触发时 req.image_urls 还是空的
    （internal.py 里 prepare_request_images 就在钩子调用之后一行），
    所以预热只能从「请求体里保留下来的那批图」出发 —— 而它们正是将来
    滑出 K 窗口时从历史里读出来的同一份字节。
    """

    def test_kept_refs_lists_only_the_preserved_window(self) -> None:
        contexts = _history(["A", "B", "C", "D"])
        stats = core.strip_history_images(contexts, keep_rounds=2, lookup=lambda key: None)
        self.assertEqual(stats.stripped, 2)
        self.assertEqual(stats.kept, 2)
        self.assertEqual(
            stats.kept_refs,
            ["data:image/jpeg;base64,C", "data:image/jpeg;base64,D"],
        )

    def test_kept_refs_empty_when_everything_is_stripped(self) -> None:
        contexts = _history(["A", "B"])
        stats = core.strip_history_images(contexts, keep_rounds=0, lookup=lambda key: None)
        self.assertEqual(stats.stripped, 2)
        self.assertEqual(stats.kept_refs, [])

    def test_kept_refs_filled_in_dry_run_without_mutation(self) -> None:
        contexts = _history(["A", "B", "C"])
        snapshot = json.dumps(contexts, sort_keys=True)
        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: None, dry_run=True
        )
        self.assertEqual(stats.kept_refs, ["data:image/jpeg;base64,C"])
        self.assertEqual(json.dumps(contexts, sort_keys=True), snapshot)

    def test_kept_refs_with_current_round_included(self) -> None:
        contexts = _history(["A", "B", "C"])
        contexts.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "这轮"}, _img_block("cur")],
            }
        )
        stats = core.strip_history_images(
            contexts,
            keep_rounds=1,
            lookup=lambda key: None,
            current_round_included=True,
        )
        # K+1 口径：保留最近 2 条 user 消息 -> 历史上的 C 与当前轮的 cur
        self.assertEqual(
            stats.kept_refs,
            ["data:image/jpeg;base64,C", "data:image/jpeg;base64,cur"],
        )
        self.assertEqual(stats.stripped, 2)


class TestDedupFolding(unittest.TestCase):
    """v1.1 同图去重折叠：旧历史区里指纹相同的图只留第一份。

    背景：群聊流水/引用图会把同一张图逐条灌进历史（实测同一张图 5~9 份），
    每份都换同一句 caption 会让模型把那段描述复读放大。
    """

    @staticmethod
    def _dup_contexts(payload: str, dup_count: int = 3) -> list[dict]:
        """旧历史区里同一张图出现 dup_count 次，最后再留一轮作为保留窗口。"""
        contexts: list[dict] = []
        for index in range(dup_count):
            contexts.append({"role": "user", "content": [_img_block(payload)]})
            contexts.append({"role": "assistant", "content": f"第{index}次"})
        contexts.append(
            {"role": "user", "content": [{"type": "text", "text": "最近一轮"}]}
        )
        contexts.append({"role": "assistant", "content": "ok"})
        return contexts

    def test_placeholder_constant(self) -> None:
        self.assertEqual(core.DUP_PLACEHOLDER, "[同前图]")

    def test_dup_images_folded(self) -> None:
        payload = base64.b64encode(b"same-image").decode()
        contexts = self._dup_contexts(payload, 3)
        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: "一只猫"
        )
        # 3 张都在旧历史区、指纹相同 -> 全部计入 stripped，其中 2 张是折叠
        self.assertEqual(stats.stripped, 3)
        self.assertEqual(stats.deduped, 2)
        self.assertEqual(stats.captioned, 1)
        self.assertEqual(stats.uncaptioned, 0)
        # 第一份走原逻辑换 caption
        self.assertEqual(
            contexts[0]["content"][0], {"type": "text", "text": "[图片：一只猫]"}
        )
        # 后两份折叠成 [同前图]
        self.assertEqual(
            contexts[2]["content"][0], {"type": "text", "text": "[同前图]"}
        )
        self.assertEqual(
            contexts[4]["content"][0], {"type": "text", "text": "[同前图]"}
        )
        # 消息条数不变
        self.assertEqual(len(contexts), 8)

    def test_dup_images_folded_without_caption(self) -> None:
        """查不到 caption 时第一份落 [图片] 占位，重复份仍是 [同前图]。"""
        payload = base64.b64encode(b"no-caption").decode()
        contexts = self._dup_contexts(payload, 3)
        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: None
        )
        self.assertEqual(stats.stripped, 3)
        self.assertEqual(stats.deduped, 2)
        self.assertEqual(stats.uncaptioned, 1)
        self.assertEqual(contexts[0]["content"][0]["text"], "[图片]")
        self.assertEqual(contexts[2]["content"][0]["text"], "[同前图]")
        self.assertEqual(contexts[4]["content"][0]["text"], "[同前图]")

    def test_dup_not_in_keep_window(self) -> None:
        """保留窗口内的重复图不参与去重，原样保留。"""
        old = base64.b64encode(b"old-unique").decode()
        keep = base64.b64encode(b"keep-dup").decode()
        contexts = [
            {"role": "user", "content": [_img_block(old)]},
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": [_img_block(keep)]},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": [_img_block(keep)]},
            {"role": "assistant", "content": "3"},
            {"role": "user", "content": [_img_block(keep)]},
            {"role": "assistant", "content": "4"},
        ]
        stats = core.strip_history_images(
            contexts, keep_rounds=2, lookup=lambda key: None
        )
        self.assertEqual(stats.deduped, 0)
        self.assertEqual(stats.stripped, 2)
        # 保留窗口里的重复 keep 图一个都没动
        for index in (4, 6):
            self.assertEqual(contexts[index]["content"][0]["type"], "image_url")
        self.assertEqual(contexts[3], {"role": "assistant", "content": "2"})

    def test_remote_url_not_deduped(self) -> None:
        """http(s) 图算不出指纹（key=None），每份都当独特处理。"""
        contexts: list[dict] = []
        for index in range(3):
            contexts.append(
                {"role": "user", "content": [_url_block("https://example.com/a.png")]}
            )
            contexts.append({"role": "assistant", "content": f"第{index}次"})
        contexts.append({"role": "user", "content": [{"type": "text", "text": "最近"}]})
        contexts.append({"role": "assistant", "content": "ok"})

        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: "不该被调用"
        )
        self.assertEqual(stats.stripped, 3)
        self.assertEqual(stats.deduped, 0)
        self.assertEqual(stats.uncaptioned, 3)
        for index in (0, 2, 4):
            self.assertEqual(contexts[index]["content"][0]["text"], "[图片]")

    def test_dry_run_counts_deduped_without_mutation(self) -> None:
        payload = base64.b64encode(b"dry-dup").decode()
        contexts = self._dup_contexts(payload, 3)
        snapshot = json.dumps(contexts, sort_keys=True)
        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: "一只猫", dry_run=True
        )
        self.assertEqual(stats.stripped, 3)
        self.assertEqual(stats.deduped, 2)
        self.assertEqual(stats.captioned, 1)
        self.assertEqual(json.dumps(contexts, sort_keys=True), snapshot)

    def test_describe_mentions_deduped(self) -> None:
        payload = base64.b64encode(b"describe").decode()
        contexts = self._dup_contexts(payload, 3)
        stats = core.strip_history_images(
            contexts, keep_rounds=1, lookup=lambda key: "一只猫"
        )
        self.assertIn("去重折叠 2 张", stats.describe())

