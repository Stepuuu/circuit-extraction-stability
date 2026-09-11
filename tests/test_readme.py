"""Check documentation and the title banner without importing research code."""
import ast
from pathlib import Path
import re
import shlex
import subprocess
import unittest
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SVG_NS = "{http://www.w3.org/2000/svg}"


class ReadmeTests(unittest.TestCase):
    def setUp(self):
        self.readme = (ROOT / "README.md").read_text()
        self.banner = (ROOT / "assets/readme/header.svg").read_text()

    def test_links_and_anchors_resolve(self):
        urls = re.findall(r"\]\(([^)]+)\)", self.readme)
        urls += re.findall(r'(?:src|href)="([^"]+)"', self.readme)
        anchors = {heading.lower().replace(" ", "-") for heading in re.findall(r"^## (.+)$", self.readme, re.M)}
        for url in urls:
            target = urlsplit(url)
            if target.scheme:
                self.assertEqual(target.scheme, "https")
                continue
            if target.path:
                self.assertTrue((ROOT / unquote(target.path)).exists(), url)
            if target.fragment and not target.path:
                self.assertIn(target.fragment, anchors)

    def test_shell_examples_parse(self):
        blocks = re.findall(r"```bash\n(.*?)```", self.readme, re.S)
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            result = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_entry_points_use_argparse(self):
        commands = re.findall(r"^python ([^\n]+\.py --help)$", self.readme, re.M)
        self.assertEqual(len(commands), 3)
        for command in commands:
            script, flag = shlex.split(command)
            self.assertEqual(flag, "--help")
            tree = ast.parse((ROOT / script).read_text())
            parsers = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == "ArgumentParser"]
            self.assertTrue(parsers, script)
            for parser in parsers:
                self.assertFalse(any(k.arg == "add_help" and isinstance(k.value, ast.Constant)
                                     and k.value.value is False for k in parser.keywords), script)

    def test_banner_accessibility_and_title(self):
        root = ET.fromstring(self.banner)
        self.assertEqual(root.attrib["role"], "img")
        self.assertEqual(root.attrib["viewBox"], "0 0 1200 290")
        title = root.find(SVG_NS + "title").text
        visible_title = " ".join(node.text for node in root.iter(SVG_NS + "text"))
        self.assertEqual(title, visible_title)
        self.assertIn(f"title={{{title}}}", self.readme)
        self.assertIn(f'alt="{title}"', self.readme)
        self.assertTrue(root.find(SVG_NS + "desc").text)

    def test_animation_is_self_contained_and_optional(self):
        self.assertIn("@keyframes drift", self.banner)
        self.assertIn("prefers-reduced-motion: reduce", self.banner)
        self.assertIn("animation: none", self.banner)
        self.assertNotRegex(self.banner, r"<script|<foreignObject|(?:href|src)=|\bon\w+=")
        self.assertNotRegex(self.banner, r"https?://(?!www.w3.org/2000/svg)")

    def test_release_scope_and_attribution(self):
        self.assertIn("not included", self.readme)
        self.assertIn("script defaults are not", self.readme)
        self.assertIn("[NOTICE](NOTICE)", self.readme)
        self.assertIn("OpenAI", self.readme)


if __name__ == "__main__":
    unittest.main()
