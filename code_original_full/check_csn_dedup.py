import json
import os
import hashlib
from collections import Counter, defaultdict

languages = ['ruby', 'go', 'php', 'python', 'java', 'javascript']
root_path = "../data/CSN"
splits = ["train", "valid", "test"]


def md5_text(text):
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_docstring_tokens(row):
    tokens = row.get("docstring_tokens", [])
    if isinstance(tokens, list):
        return " ".join(tokens).strip()
    return str(tokens).strip()


def normalize_code_tokens(row):
    tokens = row.get("code_tokens")
    if tokens is None:
        tokens = row.get("function_tokens", [])
    if isinstance(tokens, list):
        return " ".join(tokens).strip()
    return str(tokens).strip()


def analyze_split(split):
    print("\n" + "=" * 80)
    print(f"分析 split: {split}")
    print("=" * 80)

    lang_counts = {}
    total_count = 0

    url_counter = Counter()
    doc_counter = Counter()
    pair_counter = Counter()
    url_langs = defaultdict(set)

    # 只保留少量样例，避免占内存
    url_examples = {}
    cross_lang_examples = {}

    for lang in languages:
        file_path = os.path.join(root_path, lang, f"{split}.jsonl")
        count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                row = json.loads(line)

                url = row.get("url", "").strip()
                doc = normalize_docstring_tokens(row)
                code = normalize_code_tokens(row)

                # 用 hash 降低内存
                doc_hash = md5_text(doc)
                pair_hash = md5_text(doc + "\n<SEP>\n" + code)

                total_count += 1
                count += 1

                url_counter[url] += 1
                doc_counter[doc_hash] += 1
                pair_counter[pair_hash] += 1
                url_langs[url].add(lang)

                if url and url not in url_examples:
                    url_examples[url] = {
                        "lang": lang,
                        "row_idx": row_idx,
                        "doc_preview": doc[:200],
                        "code_preview": code[:200],
                    }

        lang_counts[lang] = count

    print("\n各语言样本数：")
    for lang in languages:
        print(f"- {lang}: {lang_counts[lang]}")
    print(f"\nmerge 后总样本数: {total_count}")

    dup_urls = {k: v for k, v in url_counter.items() if v > 1}
    dup_url_extra = sum(v - 1 for v in dup_urls.values())

    dup_docs = {k: v for k, v in doc_counter.items() if v > 1}
    dup_doc_extra = sum(v - 1 for v in dup_docs.values())

    dup_pairs = {k: v for k, v in pair_counter.items() if v > 1}
    dup_pair_extra = sum(v - 1 for v in dup_pairs.values())

    cross_lang_dup_urls = {
        url: sorted(list(langs))
        for url, langs in url_langs.items()
        if len(langs) > 1
    }

    print("\n[按 url 统计]")
    print(f"- unique url 数量: {len(url_counter)}")
    print(f"- 重复 url 键数: {len(dup_urls)}")
    print(f"- 因 url 重复而多出来的样本数: {dup_url_extra}")
    print(f"- url 重复率: {dup_url_extra / total_count * 100:.6f}%")
    print(f"- 跨语言重复 url 数量: {len(cross_lang_dup_urls)}")

    print("\n[按 query/docstring 统计]")
    print(f"- unique query(hash) 数量: {len(doc_counter)}")
    print(f"- 重复 query(hash) 键数: {len(dup_docs)}")
    print(f"- 因 query 重复而多出来的样本数: {dup_doc_extra}")
    print(f"- query 重复率: {dup_doc_extra / total_count * 100:.6f}%")

    print("\n[按 (query, code) 统计]")
    print(f"- unique pair(hash) 数量: {len(pair_counter)}")
    print(f"- 重复 pair(hash) 键数: {len(dup_pairs)}")
    print(f"- 因 pair 重复而多出来的样本数: {dup_pair_extra}")
    print(f"- pair 重复率: {dup_pair_extra / total_count * 100:.6f}%")

    print(f"\n==== 重复 url 示例（前 10 个） ====")
    shown = 0
    for url, cnt in dup_urls.items():
        example = url_examples.get(url, {})
        print(f"{shown + 1}. count={cnt}")
        print(f"   url={url}")
        print(f"   first_seen_lang={example.get('lang')}")
        print(f"   doc_preview={example.get('doc_preview', '')}")
        print(f"   code_preview={example.get('code_preview', '')}")
        shown += 1
        if shown >= 10:
            break
    if shown == 0:
        print("无")

    print(f"\n==== 跨语言重复 url 示例（前 10 个） ====")
    shown = 0
    for url, langs in cross_lang_dup_urls.items():
        print(f"{shown + 1}. langs={langs} | url={url}")
        shown += 1
        if shown >= 10:
            break
    if shown == 0:
        print("无")

    print("\n结论建议：")
    if dup_url_extra == 0:
        print("- 从 url 维度看，merge 后基本没有重复，可认为与论文 dedup 口径接近。")
    else:
        print("- 从 url 维度看，merge 后存在重复；若要严格对齐论文，建议 combined multilingual 训练前按 url 去重。")

    if len(cross_lang_dup_urls) > 0:
        print("- 检测到跨语言重复 url，这很像论文所说的 duplicate source identifiers。")
    else:
        print("- 没有检测到明显跨语言重复 url。")

    if dup_doc_extra > dup_url_extra:
        print("- query 级重复比 url 级重复更多，说明还有文本级重复，但这不一定是论文要求去掉的对象。")


if __name__ == "__main__":
    for split in splits:
        analyze_split(split)