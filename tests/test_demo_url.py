"""Validate download URL restrictions without optional web dependencies."""

import ast
import re
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

source = Path(__file__).resolve().parents[1] / "demo/app.py"
node = next(
    n
    for n in ast.parse(source.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "youtube_id"
)
namespace = dict(re=re, urlparse=urlparse, parse_qs=parse_qs)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
youtube_id = namespace["youtube_id"]


class YoutubeTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(youtube_id("https://youtu.be/abcdefghijk"), "abcdefghijk")
        self.assertEqual(
            youtube_id("https://www.youtube.com/watch?v=abcdefghijk&list=123"),
            "abcdefghijk",
        )

    def test_reject_local_or_untrusted(self):
        for url in [
            "http://localhost/x",
            "file:///etc/passwd",
            "https://youtube.com.evil.test/watch?v=abcdefghijk",
            "https://evil.test/",
            "https://user@youtube.com/watch?v=abcdefghijk",
            "https://youtube.com:80/watch?v=abcdefghijk",
            "https://youtu.be/../secret",
            "https://youtube.com/playlist?list=x",
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                youtube_id(url)
