"""Lightweight code normalizer for dual-view training.

Produces a 'normalized' version of source code by:
  - stripping comments (single-line and multi-line for common languages)
  - renaming local variable-like identifiers to generic names (var_0, var_1, ...)
  - collapsing excessive whitespace
"""

import keyword
import re
from typing import Set


_IDENTIFIER_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\b")

_LANGUAGE_KEYWORDS: Set[str] = set()
_LANGUAGE_KEYWORDS.update(keyword.kwlist)
_LANGUAGE_KEYWORDS.update([
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "null", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
    "var", "let", "const", "function", "require", "module", "exports",
    "func", "range", "defer", "chan", "go", "select", "map", "make",
    "array", "echo", "foreach", "elseif", "endif", "endfor",
    "end", "def", "self", "nil", "puts", "attr_accessor", "yield",
    "true", "false", "none", "True", "False", "None",
    "print", "len", "str", "int", "list", "dict", "set", "tuple",
    "sorted", "reversed", "enumerate", "zip", "filter",
])

_BUILTIN_NAMES = {
    "sorted", "append", "extend", "remove", "insert", "pop", "keys",
    "values", "items", "get", "update", "join", "split", "strip",
    "replace", "format", "encode", "decode", "read", "write", "close",
    "open", "range", "len", "print", "type", "isinstance", "hasattr",
    "getattr", "setattr", "super", "next", "iter", "map", "filter",
    "min", "max", "sum", "abs", "round", "input", "output",
    "toString", "valueOf", "indexOf", "charAt", "substring",
    "toLowerCase", "toUpperCase", "trim", "startsWith", "endsWith",
    "contains", "isEmpty", "size", "add", "clear", "equals",
    "hashCode", "compareTo", "main", "init", "setup", "test",
}


def strip_comments(code: str) -> str:
    result = []
    i = 0
    in_quote = None
    escape = False

    while i < len(code):
        ch = code[i]

        if in_quote is not None:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_quote:
                in_quote = None
            i += 1
            continue

        if code.startswith('"""', i) or code.startswith("'''", i):
            quote = code[i : i + 3]
            end = code.find(quote, i + 3)
            if end == -1:
                break
            skipped = code[i : end + 3]
            result.append("\n" * skipped.count("\n"))
            i = end + 3
            continue

        if code.startswith("/*", i):
            end = code.find("*/", i + 2)
            if end == -1:
                break
            skipped = code[i : end + 2]
            result.append("\n" * skipped.count("\n"))
            i = end + 2
            continue

        if code.startswith("//", i) or ch == "#":
            end = code.find("\n", i)
            if end == -1:
                break
            result.append("\n")
            i = end + 1
            continue

        if ch in ("'", '"'):
            in_quote = ch
            result.append(ch)
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def normalize_identifiers(code: str) -> str:
    seen = {}
    counter = [0]

    def _replacer(match: re.Match) -> str:
        name = match.group(1)
        if name in _LANGUAGE_KEYWORDS or name in _BUILTIN_NAMES:
            return name
        if len(name) <= 1:
            return name
        if name not in seen:
            seen[name] = f"var_{counter[0]}"
            counter[0] += 1
        return seen[name]

    result = []
    segment = []
    in_quote = None
    escape = False

    def flush_segment() -> None:
        if segment:
            result.append(_IDENTIFIER_RE.sub(_replacer, "".join(segment)))
            segment.clear()

    for ch in code:
        if in_quote is not None:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_quote:
                in_quote = None
            continue

        if ch in ("'", '"'):
            flush_segment()
            in_quote = ch
            result.append(ch)
            continue

        segment.append(ch)

    flush_segment()
    return "".join(result)


def collapse_whitespace(code: str) -> str:
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def normalize_code(code: str) -> str:
    code = strip_comments(code)
    code = normalize_identifiers(code)
    code = collapse_whitespace(code)
    return code
