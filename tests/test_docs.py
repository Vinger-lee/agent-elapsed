"""Documentation parity: every README language file must stay aligned and link-resolvable.

The multi-language READMEs are a promise: a reader in any of the eight languages sees the same
numbers, the same document list and a switcher that works. This test keeps that promise honest
instead of trusting a copy-paste session.
"""
from __future__ import annotations

import posixpath
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Switcher entries in display order: (file, label). The label is what readers see, so only the
# label set differs between languages; every file must link to all of them.
LANGUAGE_FILES = (
    "README.md",
    "README_CN.md",
    "docs/i18n/README.ja.md",
    "docs/i18n/README.ko.md",
    "docs/i18n/README.fr.md",
    "docs/i18n/README.es.md",
    "docs/i18n/README.ru.md",
    "docs/i18n/README.ar.md",
)

DOC_TABLE_TARGETS = (
    "docs/design.md",
    "docs/embedded-widget.md",
    "docs/integration.md",
    "docs/accuracy-audit.md",
    "docs/math-upgrade.md",
    "docs/implementation-notes.md",
    "docs/validation.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
)

COUNT_PATTERN = re.compile(r"tests-(\d+)%20passing")


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _links(body: str) -> list[tuple[str, str]]:
    """(href, target-relative-to-file) for every markdown/HTML link in the body."""
    out = []
    for match in re.finditer(r"\]\(([^)\s]+)\)", body):
        out.append((match.group(1), match.group(1)))
    for match in re.finditer(r'href="([^"]+)"', body):
        out.append((match.group(1), match.group(1)))
    return out


class ReadmeLanguageTest(unittest.TestCase):
    def test_every_language_file_exists(self):
        for relative in LANGUAGE_FILES:
            with self.subTest(file=relative):
                self.assertTrue((ROOT / relative).is_file(), f"缺少 {relative}")

    def test_every_file_carries_the_full_language_switcher(self):
        # posixpath, not os.path: the links in the documents always use forward slashes, while
        # os.path.relpath would hand back backslashes on Windows and fail a correct file.
        root = ROOT.as_posix()
        for relative in LANGUAGE_FILES:
            body = _read(relative)
            hrefs = {target for _, target in _links(body)}
            here = posixpath.dirname(posixpath.join(root, relative))
            with self.subTest(file=relative):
                for wanted in LANGUAGE_FILES:
                    expected = posixpath.relpath(posixpath.join(root, wanted), here)
                    self.assertTrue(expected in hrefs,
                                    f"{relative} 的语言切换缺 {expected}")

    def test_every_relative_link_resolves(self):
        for relative in LANGUAGE_FILES:
            base = (ROOT / relative).parent
            for href, _ in _links(_read(relative)):
                if href.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = href.split("#", 1)[0]
                if not target:
                    continue
                with self.subTest(file=relative, href=href):
                    self.assertTrue((base / target).exists(), f"{relative} 里的 {href} 指向不存在的文件")

    def test_every_language_lists_every_document(self):
        for relative in LANGUAGE_FILES:
            body = _read(relative)
            base = (ROOT / relative).parent
            listed = {
                str((base / href.split("#", 1)[0]).resolve())
                for href, _ in _links(body)
                if href.split("#", 1)[0]
            }
            for doc in DOC_TABLE_TARGETS:
                resolved = str((ROOT / doc).resolve())
                with self.subTest(file=relative, doc=doc):
                    self.assertIn(resolved, listed, f"{relative} 的文档表缺 {doc}")

    def test_documented_files_are_listed_and_vice_versa(self):
        """docs/ 下的每一篇都必须出现在 README 的文档表里，避免新增文档无人知道。"""
        body = _read("README.md")
        for path in sorted((ROOT / "docs").glob("*.md")):
            with self.subTest(doc=path.name):
                self.assertIn(f"docs/{path.name}", body)


class ReadmeNumbersTest(unittest.TestCase):
    def test_all_languages_quote_the_same_test_count(self):
        counts = {}
        for relative in LANGUAGE_FILES:
            body = _read(relative)
            found = COUNT_PATTERN.search(body)
            self.assertIsNotNone(found, f"{relative} 里没有 tests-N 徽章")
            assert found is not None
            counts[relative] = int(found.group(1))
        self.assertEqual(len(set(counts.values())), 1, f"各语言徽章数字不一致：{counts}")

    def test_badge_matches_the_actual_suite_size(self):
        """徽章数字必须等于真实测试数 —— 加测试就得改徽章，和 LEAP 的工具数一样。"""
        suite = unittest.TestLoader().discover(str(ROOT / "tests"),
                                               top_level_dir=str(ROOT),
                                               pattern="test_*.py")
        actual = suite.countTestCases()
        found = COUNT_PATTERN.search(_read("README.md"))
        self.assertIsNotNone(found)
        assert found is not None
        quoted = int(found.group(1))
        self.assertEqual(quoted, actual,
                         f"README 徽章写 {quoted} 项，实际集合有 {actual} 项")

    def test_code_blocks_are_never_translated(self):
        """带语言标签的代码块（命令、示例）必须与英文主 README 逐字一致。

        公式块允许译注（例如 "(independent of φ)" → "(与 φ 无关)"），但命令与代码是复制粘贴
        材料：一旦被"顺手翻译"，读者复制到的就是跑不通的东西。
        """
        def blocks(text):
            out, cur, lang = [], None, None
            for line in text.splitlines():
                if line.startswith("```"):
                    if cur is None:
                        cur, lang = [], line[3:].strip()
                    else:
                        out.append((lang, "\n".join(cur)))
                        cur, lang = None, None
                    continue
                if cur is not None:
                    cur.append(line)
            return [(l, b) for l, b in out if b.strip()]

        def command_part(lang, body):
            """bash 块里的 `# 注释` 允许本地化；命令行本身必须与原版一致。"""
            if lang != "bash":
                return body
            return "\n".join(line.split("#", 1)[0].rstrip() for line in body.splitlines())

        tagged = {(lang, command_part(lang, body))
                  for lang, body in blocks(_read("README.md")) if lang}
        for relative in LANGUAGE_FILES[2:]:        # 六份译文；中英两份是权威版本
            for lang, body in blocks(_read(relative)):
                if not lang:
                    continue                       # 纯公式块允许保持原文 + 译注
                with self.subTest(file=relative, lang=lang):
                    self.assertTrue((lang, command_part(lang, body)) in tagged,
                                    f"{relative} 的 {lang} 代码块与英文主 README 不一致")

    def test_key_figures_appear_in_every_language(self):
        """证据类数字不许在翻译里漂：每份译文都必须保留这些原始数字。"""
        figures = ("0.594", "0.844", "0.969", "242.8", "481.9", "2.234")
        for relative in LANGUAGE_FILES:
            body = _read(relative)
            for figure in figures:
                with self.subTest(file=relative, figure=figure):
                    # assertTrue 而不是 assertIn：失败时不要把这整篇 README 打出来
                    self.assertTrue(figure in body, f"{relative} 缺关键数字 {figure}")


if __name__ == "__main__":
    unittest.main()
