#!/usr/bin/env python3
"""Quantize VLA GGUF files from a complete YAML plan."""

from __future__ import annotations

import argparse
import ctypes
import fnmatch
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VLA_ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = VLA_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if not GGUF_PY.exists():
    raise SystemExit("third_party/llama.cpp/gguf-py is required; initialize the llama.cpp submodule")

np: Any = None
gguf: Any = None
ggml_base: Any = None


def ensure_gguf_imported() -> None:
    global np, gguf
    if gguf is not None:
        return
    sys.path.insert(0, str(GGUF_PY))
    try:
        import numpy as numpy_module
        import gguf as gguf_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"missing Python dependency: {exc.name}. Run this tool in the same environment used by the GGUF converters."
        ) from exc
    np = numpy_module
    gguf = gguf_module


FLOAT_TYPES = {"f32", "f16", "bf16"}
GGUF_PY_QUANTIZE_TYPES = {"f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"}
GGML_BULK_QUANTIZE_FUNCS = {
    "q2_k": "quantize_q2_K",
    "q3_k": "quantize_q3_K",
    "q4_k": "quantize_q4_K",
    "q5_k": "quantize_q5_K",
    "q6_k": "quantize_q6_K",
}

TYPE_ALIASES = {
    "f32": "F32",
    "float32": "F32",
    "f16": "F16",
    "float16": "F16",
    "bf16": "BF16",
    "q4_0": "Q4_0",
    "q4_1": "Q4_1",
    "q5_0": "Q5_0",
    "q5_1": "Q5_1",
    "q8_0": "Q8_0",
    "q2_k": "Q2_K",
    "q3_k": "Q3_K",
    "q4_k": "Q4_K",
    "q5_k": "Q5_K",
    "q6_k": "Q6_K",
    "iq4_nl": "IQ4_NL",
    "iq4_xs": "IQ4_XS",
}


@dataclass
class TensorPlan:
    component: str
    group: str
    name: str
    quantizable: bool
    target_type: str
    reason: str = ""


def parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "":
        return ""
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "None", "~"}:
        return None
    if text == "{}":
        return {}
    if text == "[]":
        return []
    if text[0:1] in {"'", '"'}:
        return json.loads(text) if text[0] == '"' else text[1:-1]
    try:
        if text.startswith("0") and text not in {"0"} and not text.startswith("0."):
            raise ValueError
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_yaml_subset(path: Path) -> Any:
    """Parse YAML.

    PyYAML is used when available. The fallback only supports the map/list/scalar
    shape used by the checked-in config files.
    """
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        pass

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines: list[tuple[int, str]] = []
    for raw in raw_lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        is_list = lines[index][1].startswith("-")
        if is_list:
            out: list[Any] = []
            while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("-"):
                item = lines[index][1][1:].strip()
                if item == "":
                    value, index = parse_block(index + 1, lines[index + 1][0])
                else:
                    value = parse_scalar(item)
                    index += 1
                out.append(value)
            return out, index

        out: dict[str, Any] = {}
        while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("-"):
            item = lines[index][1]
            if ":" not in item:
                raise ValueError(f"invalid YAML line: {item}")
            key, rest = item.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            if rest:
                out[key] = parse_scalar(rest)
                index += 1
            else:
                if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
                    out[key] = {}
                    index += 1
                else:
                    out[key], index = parse_block(index + 1, lines[index + 1][0])
        return out, index

    parsed, end = parse_block(0, lines[0][0] if lines else 0)
    if end != len(lines):
        raise ValueError(f"failed to parse YAML near line {end + 1}")
    return parsed


def expand_var_expr(match: re.Match[str]) -> str:
    expr = match.group(1)
    if ":-" in expr:
        name, fallback = expr.split(":-", 1)
        return os.environ.get(name, fallback)
    if expr not in os.environ:
        raise SystemExit(f"environment variable {expr} is required by the quant YAML")
    return os.environ[expr]


def expand_path(value: Any) -> Path:
    if value is None or str(value) == "":
        raise SystemExit("empty path in quant YAML")
    text = re.sub(r"\$\{([^}]+)\}", expand_var_expr, str(value))
    path = Path(os.path.expanduser(text))
    if not path.is_absolute():
        path = VLA_ROOT / path
    return path


def parse_type(type_name: str) -> Any:
    ensure_gguf_imported()
    key = str(type_name).lower()
    enum_name = TYPE_ALIASES.get(key, key.upper())
    try:
        return getattr(gguf.GGMLQuantizationType, enum_name)
    except AttributeError as exc:
        raise ValueError(f"unsupported tensor type: {type_name}") from exc


def type_name(qtype: Any) -> str:
    return qtype.name.lower()


def is_quant_type(type_name_: str) -> bool:
    return str(type_name_).lower() not in FLOAT_TYPES


def is_gguf_py_quantize_supported(qtype: Any) -> bool:
    return type_name(qtype) in GGUF_PY_QUANTIZE_TYPES


def is_ggml_ref_quantize_supported(qtype: Any) -> bool:
    return type_name(qtype) in GGML_BULK_QUANTIZE_FUNCS


def is_quantize_supported(qtype: Any) -> bool:
    return is_gguf_py_quantize_supported(qtype) or is_ggml_ref_quantize_supported(qtype)


def ggml_base_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("GGML_BASE_LIB")
    if env_path:
        candidates.append(Path(env_path))
    patterns = (
        "libggml-base*.dylib",
        "libggml-base*.so*",
        "ggml-base*.dll",
        "libggml-base*.dll",
    )
    for build_dir in sorted(VLA_ROOT.glob("build*")):
        for path in build_dir.rglob("*"):
            if any(fnmatch.fnmatchcase(path.name, pattern) for pattern in patterns):
                candidates.append(path)
    return candidates


def ensure_ggml_base_loaded() -> Any:
    global ggml_base
    if ggml_base is not None:
        return ggml_base
    errors: list[str] = []
    for path in ggml_base_candidates():
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        for func_name in GGML_BULK_QUANTIZE_FUNCS.values():
            func = getattr(lib, func_name)
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
            ]
            func.restype = ctypes.c_size_t
        ggml_base = lib
        return ggml_base
    detail = "\n".join(errors)
    if detail:
        detail = "\nLoad errors:\n" + detail
    raise SystemExit(
        "K-quant output requires libggml-base from a local llama.cpp/vla.cpp build. "
        "Run tools/quant/shell/quant.sh to build it automatically, or set "
        "GGML_BASE_LIB=/path/to/libggml-base runtime library." + detail
    )


def logical_shape(tensor: Any) -> list[int]:
    return [int(v) for v in reversed(tensor.shape.tolist())]


def tensor_nbytes_for_shape(shape: list[int], qtype: Any) -> int:
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if not shape:
        return 0
    if shape[-1] % block_size != 0:
        raise ValueError(
            f"shape {shape} is incompatible with {qtype.name}: last dim is not a multiple of {block_size}"
        )
    elems = 1
    for dim in shape:
        elems *= int(dim)
    return elems * type_size // block_size


def copy_metadata(reader: Any, writer: Any) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = field.contents()
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY and len(field.types) > 1 else None
        if key == "general.alignment":
            writer.add_custom_alignment(int(value))
        else:
            writer.add_key_value(key, value, vtype, sub_type=sub_type)


def tensor_to_f32(tensor: Any) -> Any:
    qtype = tensor.tensor_type
    if qtype == gguf.GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32)
    if qtype == gguf.GGMLQuantizationType.F16:
        return np.asarray(tensor.data, dtype=np.float16).astype(np.float32)
    if qtype == gguf.GGMLQuantizationType.F64:
        return np.asarray(tensor.data, dtype=np.float64).astype(np.float32)
    return gguf.dequantize(np.asarray(tensor.data), qtype).astype(np.float32)


def convert_tensor_data(tensor: Any, target: Any) -> Any:
    if tensor.tensor_type == target:
        return np.asarray(tensor.data)
    data = tensor_to_f32(tensor)
    if target == gguf.GGMLQuantizationType.F32:
        return data.astype(np.float32, copy=False)
    if target == gguf.GGMLQuantizationType.F16:
        return data.astype(np.float16, copy=False)
    if is_ggml_ref_quantize_supported(target):
        return quantize_with_ggml_ref(data, target)
    return gguf.quantize(data, target)


def quantize_with_ggml_ref(data: Any, target: Any) -> Any:
    qname = type_name(target)
    block_size, type_size = gguf.GGML_QUANT_SIZES[target]
    if data.shape[-1] % block_size != 0:
        raise ValueError(f"shape {list(data.shape)} is incompatible with {qname}: last dim is not a multiple of {block_size}")

    lib = ensure_ggml_base_loaded()
    func = getattr(lib, GGML_BULK_QUANTIZE_FUNCS[qname])
    source = np.ascontiguousarray(data, dtype=np.float32)
    out_shape = (*source.shape[:-1], source.shape[-1] // block_size * type_size)
    out = np.empty(out_shape, dtype=np.uint8)

    source_rows = source.reshape((-1, source.shape[-1]))
    expected_nbytes = int(out.size)
    written = int(
        func(
            source_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(source_rows.shape[0]),
            ctypes.c_int64(source_rows.shape[1]),
            ctypes.cast(None, ctypes.POINTER(ctypes.c_float)),
        )
    )
    if written != expected_nbytes:
        raise RuntimeError(f"{qname} bulk quantizer wrote {written} bytes, expected {expected_nbytes}")
    return out


def load_plan(path: Path) -> dict[str, Any]:
    plan = load_yaml_subset(path)
    if not isinstance(plan, dict):
        raise SystemExit("plan root must be a mapping")
    if int(plan.get("version", 0)) != 1:
        raise SystemExit("unsupported plan version")
    if "components" not in plan or not isinstance(plan["components"], dict):
        raise SystemExit("plan must contain components")
    return plan


def plan_requires_ggml_ref_quantizer(plan: dict[str, Any]) -> bool:
    components = plan.get("components", {})
    if not isinstance(components, dict):
        return False
    for component, plan_component in components.items():
        if not isinstance(plan_component, dict):
            raise SystemExit(f"component {component}: component must be a mapping")
        groups = plan_component.get("groups", {})
        if not isinstance(groups, dict):
            raise SystemExit(f"component {component}: missing groups")
        for group_name, group in groups.items():
            if not isinstance(group, dict):
                raise SystemExit(f"component {component} group {group_name}: group must be a mapping")
            target = parse_type(str(group.get("type", "f32")).lower())
            if is_ggml_ref_quantize_supported(target):
                return True
    return False


def build_tensor_plan(component: str, plan_component: dict[str, Any], actual_names: set[str]) -> dict[str, TensorPlan]:
    by_tensor: dict[str, TensorPlan] = {}
    groups = plan_component.get("groups")
    if not isinstance(groups, dict):
        raise SystemExit(f"component {component}: missing groups")
    for group_name, group in groups.items():
        if not isinstance(group, dict):
            raise SystemExit(f"component {component} group {group_name}: group must be a mapping")
        tensors = group.get("tensors", [])
        if not isinstance(tensors, list):
            raise SystemExit(f"component {component} group {group_name}: tensors must be a list")
        patterns = group.get("patterns", [])
        if "pattern" in group:
            patterns = [group["pattern"], *patterns]
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            raise SystemExit(f"component {component} group {group_name}: patterns must be a list")
        target_type = str(group.get("type", "f32")).lower()
        parse_type(target_type)
        quantizable = bool(group.get("quantizable", False))
        reason = str(group.get("reason", ""))

        names = [str(name) for name in tensors]
        for pattern in patterns:
            pattern = str(pattern)
            matched = sorted(name for name in actual_names if fnmatch.fnmatchcase(name, pattern))
            if not matched:
                raise SystemExit(f"component {component} group {group_name}: pattern matched no tensors: {pattern}")
            names.extend(matched)

        for name in names:
            name = str(name)
            if name in by_tensor:
                prev = by_tensor[name]
                raise SystemExit(
                    f"component {component}: tensor {name} appears in both {prev.group} and {group_name}"
                )
            by_tensor[name] = TensorPlan(component, str(group_name), name, quantizable, target_type, reason)
    return by_tensor


def validate_component(
    component: str,
    plan_component: dict[str, Any],
    allow_unsafe: bool,
    allow_requantize: bool,
) -> tuple[Any, dict[str, TensorPlan], list[dict[str, Any]]]:
    input_path = expand_path(plan_component.get("input"))
    if not input_path.exists():
        raise SystemExit(f"component {component}: input GGUF does not exist: {input_path}")
    reader = gguf.GGUFReader(input_path)
    actual_names = {tensor.name for tensor in reader.tensors}
    by_tensor = build_tensor_plan(component, plan_component, actual_names)
    planned_names = set(by_tensor)
    missing = sorted(actual_names - planned_names)
    extra = sorted(planned_names - actual_names)
    if missing:
        raise SystemExit(f"component {component}: YAML does not cover tensors: {missing[:10]}")
    if extra:
        raise SystemExit(f"component {component}: YAML references missing tensors: {extra[:10]}")

    report_rows: list[dict[str, Any]] = []
    for tensor in reader.tensors:
        tp = by_tensor[tensor.name]
        target = parse_type(tp.target_type)
        original = tensor.tensor_type
        shape = logical_shape(tensor)
        if is_quant_type(tp.target_type) and not tp.quantizable and not allow_unsafe:
            raise SystemExit(
                f"component {component}: group {tp.group} is quantizable=false but tensor {tensor.name} targets {tp.target_type}; use --allow-unsafe"
            )
        if original != target and original not in {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.F64,
            gguf.GGMLQuantizationType.BF16,
        } and not allow_requantize:
            raise SystemExit(
                f"component {component}: tensor {tensor.name} is already {original.name}; use --allow-requantize"
            )
        if original != target and not is_quantize_supported(target):
            supported = ", ".join(sorted(GGUF_PY_QUANTIZE_TYPES | set(GGML_BULK_QUANTIZE_FUNCS)))
            raise SystemExit(
                f"component {component}: tensor {tensor.name} targets {type_name(target)}, "
                f"but this tool can only write: {supported}."
            )
        try:
            target_nbytes = tensor_nbytes_for_shape(shape, target)
        except ValueError as exc:
            raise SystemExit(f"component {component}: tensor {tensor.name}: {exc}") from exc
        report_rows.append(
            {
                "component": component,
                "group": tp.group,
                "tensor": tensor.name,
                "shape": shape,
                "quantizable": tp.quantizable,
                "original_type": type_name(original),
                "target_type": type_name(target),
                "original_nbytes": int(tensor.n_bytes),
                "target_nbytes": int(target_nbytes),
                "changed": original != target,
                "reason": tp.reason,
            }
        )
    return reader, by_tensor, report_rows


def summarize(rows: list[dict[str, Any]]) -> None:
    totals: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    component_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        key = (row["component"], row["group"], row["original_type"], row["target_type"])
        totals[key][0] += 1
        totals[key][1] += int(row["original_nbytes"])
        totals[key][2] += int(row["target_nbytes"])
        component_totals[row["component"]][0] += 1
        component_totals[row["component"]][1] += int(row["original_nbytes"])
        component_totals[row["component"]][2] += int(row["target_nbytes"])

    for component, values in component_totals.items():
        count, before, after = values
        print(f"{component}: tensors={count} size={before / 1048576:.2f}MB -> {after / 1048576:.2f}MB")
        for key, group_values in sorted(totals.items()):
            comp, group, original, target = key
            if comp != component:
                continue
            n, b, a = group_values
            print(f"  {group}: {n} tensors {original}->{target} {b / 1048576:.2f}MB -> {a / 1048576:.2f}MB")


def write_report(plan: dict[str, Any], rows: list[dict[str, Any]], explicit_path: Path | None) -> None:
    report_path = explicit_path
    if report_path is None:
        report = plan.get("report", {})
        if isinstance(report, dict) and report.get("path"):
            report_path = expand_path(report["path"])
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model_type": plan.get("model_type", ""),
        "tensors": rows,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote quant report: {report_path}")


def apply_component(
    component: str,
    plan_component: dict[str, Any],
    reader: Any,
    by_tensor: dict[str, TensorPlan],
) -> None:
    output_path = expand_path(plan_component.get("output"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arch_field = reader.get_field("general.architecture")
    arch = str(arch_field.contents()) if arch_field is not None else "vlacpp"
    writer = gguf.GGUFWriter(output_path, arch=arch, use_temp_file=True)
    copy_metadata(reader, writer)
    for tensor in reader.tensors:
        target = parse_type(by_tensor[tensor.name].target_type)
        try:
            data = convert_tensor_data(tensor, target)
        except NotImplementedError as exc:
            raise SystemExit(
                f"component {component}: failed to convert tensor {tensor.name} to {type_name(target)}: {exc}"
            ) from exc
        raw_shape = list(data.shape) if data.dtype == np.uint8 else logical_shape(tensor)
        writer.add_tensor(tensor.name, data, raw_shape=raw_shape, raw_dtype=target)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {component}: {output_path}")


def command_quantize(args: argparse.Namespace) -> None:
    ensure_gguf_imported()
    plan = load_plan(args.plan)
    if plan_requires_ggml_ref_quantizer(plan):
        ensure_ggml_base_loaded()
    all_rows: list[dict[str, Any]] = []
    validated: list[tuple[str, dict[str, Any], Any, dict[str, TensorPlan]]] = []
    for component, plan_component in plan["components"].items():
        reader, by_tensor, rows = validate_component(
            str(component),
            plan_component,
            allow_unsafe=args.allow_unsafe,
            allow_requantize=args.allow_requantize,
        )
        all_rows.extend(rows)
        validated.append((str(component), plan_component, reader, by_tensor))

    summarize(all_rows)
    write_report(plan, all_rows, args.report)
    if args.dry_run:
        print("dry-run complete; no GGUF files written")
        return
    for component, plan_component, reader, by_tensor in validated:
        apply_component(component, plan_component, reader, by_tensor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="complete YAML quant plan")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing GGUF outputs")
    parser.add_argument("--allow-unsafe", action="store_true", help="allow quantized target types for quantizable=false groups")
    parser.add_argument("--allow-requantize", action="store_true", help="allow converting tensors that are already quantized")
    parser.add_argument("--report", type=Path, default=None, help="override report JSON path")
    parser.set_defaults(func=command_quantize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args() 
    args.func(args) # default: command_quantize


if __name__ == "__main__":
    main()
