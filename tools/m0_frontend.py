#!/usr/bin/env python3
"""Reproduce pinned Qwen3.8 frontend inputs locally; no network or model execution."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import stat
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "bench/reference/frontend-lock.json"
CACHE = ROOT / ".cache/m0-frontend"
WORKLOADS = ROOT / "bench/workloads"
MAX_TARGET = 32768
MAX_RESERVE = 8192
REASONING_EFFORTS = {"low", "medium", "xhigh"}
OPTION_KEYS = {
    "add_generation_prompt", "tools", "documents", "date_string",
    "enable_thinking", "reasoning_effort", "preserve_thinking", "add_vision_id",
}


class FrontendDependencyError(RuntimeError):
    pass


def _tokenizer_type():
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise FrontendDependencyError("frontend evidence requires tokenizers==0.22.1; install the project-local environment described in docs/m0-frontend.md") from exc
    return Tokenizer


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_local_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("invalid locked local path")
    posix = PurePosixPath(name)
    if posix.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("locked local path must be canonical and relative")
    base = Path(root).resolve(strict=True)
    current = base
    for part in posix.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink forbidden in locked local path: " + name)
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(base) or not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("locked local path escapes cache or is not a regular file: " + name)
    return resolved


def load_locked_files(lock_path=LOCK, cache_root=CACHE):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("schema") != "m0-frontend-source-lock-v1":
        raise ValueError("unsupported frontend lock schema")
    loaded = {}
    seen = set()
    for record in lock["files"]:
        model, name = record["model"], record["name"]
        if model not in lock["models"] or not isinstance(name, str) or (model, name) in seen:
            raise ValueError("invalid or duplicate locked file identity")
        seen.add((model, name))
        path = _checked_local_file(cache_root, record["local_path"])
        raw = path.read_bytes()
        if len(raw) != record["size"] or sha256(raw) != record["sha256"]:
            raise ValueError("locked file size/hash mismatch: " + record["local_path"])
        loaded[(lock["models"][model]["repo"], name)] = raw
    return lock, loaded


def _object(raw, label):
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object")
    return value


def _tokenizer_specials(config, tokenizer_config, tokenizer):
    special = {}
    added = tokenizer_config.get("added_tokens_decoder", {})
    text = config.get("text_config", {})
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        raw = tokenizer_config.get(key)
        content = raw.get("content") if isinstance(raw, dict) else raw
        token_id = None
        if content is not None:
            token_id = next((int(idx) for idx, value in added.items() if value.get("content") == content), None)
            if token_id is None:
                token_id = tokenizer.token_to_id(content)
        if key == "bos_token" and content is None:
            token_id = text.get("bos_token_id")
            content = tokenizer.id_to_token(token_id) if token_id is not None else None
        special[key] = {"content": content, "id": token_id}
    return special


def compare_frontends(lock_path=LOCK, cache_root=CACHE):
    lock, files = load_locked_files(lock_path, cache_root)
    models = lock["models"]
    original, gsq = models["original"]["repo"], models["gsq"]["repo"]
    common = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        a, b = files[(original, name)], files[(gsq, name)]
        common[name] = {"byte_identical": a == b, "original_sha256": sha256(a), "gsq_sha256": sha256(b)}
    configs = {side: _object(files[(repo, "config.json")], side + " config") for side, repo in (("original", original), ("gsq", gsq))}
    token_configs = {side: _object(files[(repo, "tokenizer_config.json")], side + " tokenizer config") for side, repo in (("original", original), ("gsq", gsq))}
    Tokenizer = _tokenizer_type()
    tokenizers = {side: Tokenizer.from_str(files[(repo, "tokenizer.json")].decode("utf-8")) for side, repo in (("original", original), ("gsq", gsq))}
    text_configs = {side: cfg.get("text_config", {}) for side, cfg in configs.items()}
    geometry_fields = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings")
    generation = {side: _object(files[(repo, "generation_config.json")], side + " generation config") for side, repo in (("original", original), ("gsq", gsq))}
    generation_keys = sorted(set(generation["original"]) | set(generation["gsq"]))
    return {
        "file_comparison": common,
        "tokenizer_vocab_size": {side: tokenizers[side].get_vocab_size(with_added_tokens=True) for side in tokenizers},
                "config_vocab_size": {side: text_configs[side].get("vocab_size") for side in text_configs},
        "tokenizer_special_bindings": {side: _tokenizer_specials(configs[side], token_configs[side], tokenizers[side]) for side in tokenizers},
        "config_geometry": {field: {"original": text_configs["original"].get(field), "gsq": text_configs["gsq"].get(field), "equal": text_configs["original"].get(field) == text_configs["gsq"].get(field)} for field in geometry_fields},
        "generation_config": {"byte_identical": files[(original, "generation_config.json")] == files[(gsq, "generation_config.json")], "fields": {key: {"original": generation["original"].get(key), "gsq": generation["gsq"].get(key), "equal": generation["original"].get(key) == generation["gsq"].get(key)} for key in generation_keys}},
        "compatibility_limits": ["Byte-identical frontend files do not prove identical runtime loading or component-wide model compatibility.", "Config geometry is reported for comparison only; weights, runtime support, dispatch, MTP head/weights, generation and quality are not verified.", "Frontend inputs alone are not a frozen suite comparison."],
    }


def _render(tokenizer, template, messages, options, special_tokens):
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:
        raise FrontendDependencyError("frontend rendering requires Jinja2==3.1.6; install the project-local environment described in docs/m0-frontend.md") from exc
    env = ImmutableSandboxedEnvironment(autoescape=False, keep_trailing_newline=False)
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    def raise_exception(message):
        raise ValueError(str(message))
    rendered = env.from_string(template).render(
        messages=messages, add_generation_prompt=options["add_generation_prompt"],
        tools=options["tools"], documents=options["documents"],
        date_string=options["date_string"], enable_thinking=options["enable_thinking"],
        reasoning_effort=options["reasoning_effort"], preserve_thinking=options["preserve_thinking"],
        add_vision_id=options["add_vision_id"], bos_token=special_tokens["bos_token"]["content"],
        eos_token=special_tokens["eos_token"]["content"], raise_exception=raise_exception)
    return rendered


def _validate_candidate(spec):
    if not isinstance(spec, dict):
        raise ValueError("workload must be a JSON object")
    candidate = spec.get("id")
    if not isinstance(candidate, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", candidate):
        raise ValueError("candidate id must be 1-80 safe ASCII letters/digits/hyphens/underscores")
    target, reserve = spec.get("target_prompt_tokens"), spec.get("max_new_tokens")
    if target is not None and (type(target) is not int or not 1 <= target <= MAX_TARGET):
        raise ValueError("target_prompt_tokens must be a positive integer <= " + str(MAX_TARGET))
    if type(reserve) is not int or not 1 <= reserve <= MAX_RESERVE:
        raise ValueError("max_new_tokens must be a positive integer <= " + str(MAX_RESERVE))
    options = spec.get("template_options")
    if not isinstance(options, dict) or set(options) != OPTION_KEYS:
        raise ValueError("template options must explicitly provide the complete supported set")
    if type(options["add_generation_prompt"]) is not bool or type(options["enable_thinking"]) is not bool or type(options["preserve_thinking"]) is not bool or type(options["add_vision_id"]) is not bool:
        raise ValueError("boolean template options must be booleans")
    if not isinstance(options["reasoning_effort"], str) or options["reasoning_effort"] not in REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be one of low, medium, or xhigh")
    if not isinstance(options["tools"], list) or not isinstance(options["documents"], list) or not isinstance(options["date_string"], str):
        raise ValueError("invalid deterministic template options")
    try:
        datetime.date.fromisoformat(options["date_string"])
    except ValueError as exc:
        raise ValueError("date_string must be a valid YYYY-MM-DD date") from exc
    messages = spec.get("messages")
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("messages must be a non-empty list of role/content text objects")
    return messages


def build_candidate(workload, model="original", lock_path=LOCK, cache_root=CACHE):
    lock, files = load_locked_files(lock_path, cache_root)
    if model not in lock["models"]:
        raise ValueError("model must be original or gsq")
    repo = lock["models"][model]["repo"]
    Tokenizer = _tokenizer_type()
    tokenizer = Tokenizer.from_str(files[(repo, "tokenizer.json")].decode("utf-8"))
    config = _object(files[(repo, "config.json")], "model config")
    tokenizer_config = _object(files[(repo, "tokenizer_config.json")], "tokenizer config")
    specials = _tokenizer_specials(config, tokenizer_config, tokenizer)
    template = files[(repo, "chat_template.jinja")].decode("utf-8")
    spec = json.loads(Path(workload).read_text(encoding="utf-8"))
    messages = json.loads(json.dumps(_validate_candidate(spec), ensure_ascii=False))
    options = spec["template_options"]
    target = spec.get("target_prompt_tokens")
    if target is not None:
        if len(messages) != 1:
            raise ValueError("targeted workload candidates require exactly one text message")
        base = messages[0]["content"]
        def measure(repetitions):
            messages[0]["content"] = base + (" x" * repetitions)
            rendered_text = _render(tokenizer, template, messages, options, specials)
            return len(tokenizer.encode(rendered_text, add_special_tokens=False).ids), rendered_text
        low, high = 0, max(target + 16, 32)
        while measure(high)[0] < target and high < target * 4 + 4096:
            high *= 2
        best = None
        while low <= high:
            mid = (low + high) // 2
            count, text = measure(mid)
            if count == target:
                best = text
                break
            if count < target:
                low = mid + 1
            else:
                high = mid - 1
        if best is None:
            for repetitions in range(max(0, high - 8), high + 10):
                count, text = measure(repetitions)
                if count == target:
                    best = text
                    break
        if best is None:
            raise ValueError("could not hit exact prompt token target using bounded x-filler search")
        rendered = best
    else:
        rendered = _render(tokenizer, template, messages, options, specials)
    ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    if target is not None and len(ids) != target:
        raise ValueError("rendered prompt token count does not equal requested target")
    if not spec.get("max_new_tokens") + len(ids) <= MAX_TARGET + MAX_RESERVE:
        raise ValueError("prompt plus output reserve exceeds supported bound")
    return {"schema": "m0-frontend-candidate-v1", "candidate": spec["id"], "model": model,
            "source_revision": lock["models"][model]["revision"], "workload_class": spec.get("class"),
            "target_prompt_tokens": target, "actual_prompt_tokens": len(ids),
            "target_length_matches_actual": target is not None and len(ids) == target,
            "rendered_text": rendered, "rendered_utf8_hex": rendered.encode("utf-8").hex(),
            "token_ids": ids, "max_new_tokens": spec["max_new_tokens"],
            "total_context_tokens": len(ids) + spec["max_new_tokens"],
            "template_options": options,
            "qualification": "frontend input evidence only; suite remains UNFROZEN"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--model", choices=("original", "gsq"), default="original")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        outputs = []
        if args.compare:
            outputs.append(("frontend-comparison.json", compare_frontends()))
        for workload in args.workload:
            result = build_candidate(workload, args.model)
            outputs.append((result["candidate"] + ".json", result))
        if not outputs:
            parser.error("select --compare and/or --workload")
        for name, value in outputs:
            raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / name).write_text(raw, encoding="utf-8")
            else:
                print(raw, end="")
        return 0
    except FrontendDependencyError as exc:
        print("m0_frontend: " + str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError, UnicodeError) as exc:
        print("m0_frontend: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
