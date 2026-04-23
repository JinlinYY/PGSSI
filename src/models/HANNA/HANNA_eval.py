"""
在 CSV（默认 dataset/all/all_merged_test.csv）上调用 RPTU MLPROP 远程 HANNA 接口，
对无穷稀释溶质活度系数 ln(gamma^∞) 做预测并计算 MAE / R2 / RMSE。

接口与网页一致：https://ml-prop.mv.rptu.de/application.html
后端默认：https://ml-prop.mv.rptu.de:8000 （/make_smiles_canonical, /predict）

CSV 约定（与 MPNN-Cat-GH 评测相同）：
  - 列：Solvent_SMILES, Solute_SMILES, T, log-gamma
  - log-gamma 为溶质在溶剂中无穷稀释时的 ln(gamma)
  - 传给 HANNA：smiles_1 = 溶质，smiles_2 = 溶剂（与 MLPROP 中 x1→0 时 ln γ1 一致）
  - T 默认按摄氏度理解，并换算为开尔文再调用 API（HANNA 表单为 Temperature / K）

用法（在仓库根目录）:
  py -3 src/models/HANNA/HANNA_eval.py --max-rows 20
  py -3 src/models/HANNA/HANNA_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    import numpy as np

    y_true_a = np.asarray(y_true, dtype=np.float64)
    y_pred_a = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true_a) & np.isfinite(y_pred_a)
    if mask.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    yt = y_true_a[mask]
    yp = y_pred_a[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _post_json(url: str, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _canonical_smiles(base: str, smiles: str, *, timeout: float) -> str:
    out = _post_json(f"{base.rstrip('/')}/make_smiles_canonical", {"smiles": smiles}, timeout=timeout)
    if out.get("error"):
        raise RuntimeError(str(out["error"]))
    return str(out["canonical_smiles"])


def _predict_ac_hanna(
    base: str,
    *,
    model_id: str,
    smiles_1: str,
    smiles_2: str,
    temperature_k: float,
    timeout: float,
) -> Dict[str, Any]:
    body = {
        "model_id": model_id,
        "data": {
            "temperature": float(temperature_k),
            "smiles_1": smiles_1,
            "smiles_2": smiles_2,
        },
    }
    return _post_json(f"{base.rstrip('/')}/predict", body, timeout=timeout)


MOL_X_KEY = "Mol Fractions of Component 1"
LN_G1_KEY = "Logarithmic Activity Coefficient of Component 1"


def _ln_gamma_infinite_dilution_comp1(result: Dict[str, Any]) -> float:
    """取 x1→0 时组分 1 的 ln γ（API 返回离散网格，取摩尔分数最接近 0 的点）。"""
    if MOL_X_KEY not in result or LN_G1_KEY not in result:
        raise KeyError(f"Unexpected predict keys: {list(result.keys())}")
    xs = [float(x.strip()) for x in str(result[MOL_X_KEY]).split(",") if x.strip()]
    ln1 = [float(x.strip()) for x in str(result[LN_G1_KEY]).split(",") if x.strip()]
    if len(xs) != len(ln1) or not xs:
        raise ValueError("Mol fractions and ln-gamma series length mismatch or empty.")
    i = min(range(len(xs)), key=lambda j: abs(xs[j]))
    return float(ln1[i])


def _to_kelvin(t: float, unit: str) -> float:
    u = unit.strip().lower()
    if u in ("c", "celsius", "degc"):
        return float(t) + 273.15
    if u in ("k", "kelvin"):
        return float(t)
    raise ValueError(f"Unknown temperature unit: {unit!r} (use celsius or kelvin)")


@dataclass
class _Row:
    solvent: str
    solute: str
    t_raw: float
    y: float
    index: int


def _read_rows(csv_path: str, max_rows: Optional[int]) -> List[_Row]:
    import csv

    required = {"Solvent_SMILES", "Solute_SMILES", "log-gamma", "T"}
    rows: List[_Row] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV header mismatch in {csv_path}. Found: {reader.fieldnames}")
        for i, r in enumerate(reader):
            if max_rows is not None and i >= int(max_rows):
                break
            rows.append(
                _Row(
                    solvent=str(r["Solvent_SMILES"]).strip(),
                    solute=str(r["Solute_SMILES"]).strip(),
                    t_raw=float(r["T"]),
                    y=float(r["log-gamma"]),
                    index=i,
                )
            )
    return rows


def _one_row(
    base: str,
    model_id: str,
    row: _Row,
    *,
    t_unit: str,
    canonicalize: bool,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> Tuple[int, float, Optional[str]]:
    err: Optional[str] = None
    for attempt in range(int(retries) + 1):
        try:
            s1 = row.solute
            s2 = row.solvent
            if canonicalize:
                s1 = _canonical_smiles(base, s1, timeout=timeout)
                s2 = _canonical_smiles(base, s2, timeout=timeout)
            tk = _to_kelvin(row.t_raw, t_unit)
            res = _predict_ac_hanna(
                base,
                model_id=model_id,
                smiles_1=s1,
                smiles_2=s2,
                temperature_k=tk,
                timeout=timeout,
            )
            if res.get("error"):
                raise RuntimeError(str(res["error"]))
            pred = _ln_gamma_infinite_dilution_comp1(res)
            return row.index, pred, None
        except Exception as e:  # noqa: BLE001 — 远程评测需吞掉单行错误
            err = f"{type(e).__name__}: {e}"
            if attempt < int(retries):
                time.sleep(float(retry_sleep) * (attempt + 1))
    return row.index, float("nan"), err


def main() -> None:
    repo = _repo_root()
    default_csv = os.path.join(repo, "dataset", "all", "all_merged_test.csv")
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "HANNA_eval.json")

    parser = argparse.ArgumentParser(description="MLPROP HANNA：远程预测并在 CSV 上评测")
    parser.add_argument("--csv", type=str, default=default_csv, help="测试 CSV 路径")
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://ml-prop.mv.rptu.de:8000",
        help="MLPROP FastAPI 根地址（与网页前端 machineUrl 一致）",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="ac_hanna",
        help="MLPROP model_id，默认 ac_hanna（新 HANNA）；可选 ac_hanna_legacy",
    )
    parser.add_argument(
        "--t-unit",
        type=str,
        default="celsius",
        choices=("celsius", "kelvin"),
        help="CSV 列 T 的单位（默认 celsius，与 all_merged_test 常见写法一致）",
    )
    parser.add_argument("--out-json", type=str, default=default_out, help="指标 JSON 输出路径")
    parser.add_argument("--max-rows", type=int, default=None, help="仅评测前 N 行（调试）")
    parser.add_argument("--workers", type=int, default=2, help="并发线程数（请勿过大以免压垮公共服务）")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次 HTTP 超时（秒）")
    parser.add_argument("--retries", type=int, default=2, help="单行失败后的重试次数")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="重试基础间隔（秒）")
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="每条样本调用 /make_smiles_canonical（与网页一致，但请求量约为 3 倍）",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="每完成 N 行向 stdout 打印一行进度（0 关闭）；tqdm 进度条在 stderr",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    rows = _read_rows(csv_path, args.max_rows)
    if not rows:
        raise RuntimeError("No rows to evaluate.")

    preds: List[Optional[float]] = [None] * len(rows)
    labels = [r.y for r in rows]
    errors: List[Dict[str, Any]] = []

    t_unit = "celsius" if args.t_unit == "celsius" else "kelvin"

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {
            ex.submit(
                _one_row,
                args.api_base,
                args.model_id,
                row,
                t_unit=t_unit,
                canonicalize=bool(args.canonicalize),
                timeout=float(args.timeout),
                retries=int(args.retries),
                retry_sleep=float(args.retry_sleep),
            ): row
            for row in rows
        }
        pbar = tqdm(
            total=len(futs),
            desc="HANNA (remote)",
            ascii=True,
            dynamic_ncols=True,
            unit="row",
            mininterval=0.25,
            file=sys.stderr,
        )
        log_every = max(0, int(args.log_every))
        done = 0
        for fut in as_completed(futs):
            row = futs[fut]
            idx, pred, err = fut.result()
            preds[idx] = pred
            if err is not None:
                errors.append({"index": row.index, "solute": row.solute, "solvent": row.solvent, "error": err})
            done += 1
            pbar.update(1)
            if log_every and done % log_every == 0:
                print(
                    f"[HANNA_eval] {done}/{len(futs)} rows, failures={len(errors)}",
                    flush=True,
                )
        pbar.close()

    y_true: List[float] = []
    y_pred: List[float] = []
    for r, p in zip(rows, preds):
        if p is not None and isinstance(p, float) and not (p != p):  # not NaN
            y_true.append(r.y)
            y_pred.append(float(p))

    metrics = _metrics(y_true, y_pred)
    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    report = {
        "model": "HANNA",
        "remote": "MLPROP",
        "application_url": "https://ml-prop.mv.rptu.de/application.html",
        "api_base": args.api_base,
        "model_id": args.model_id,
        "csv": csv_path,
        "t_unit": t_unit,
        "canonicalize": bool(args.canonicalize),
        "metrics": {
            "num_samples": int(len(y_true)),
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
        },
        "num_failed": int(len(errors)),
        "errors_sample": errors[:50],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(report["metrics"], indent=2))
    print(f"Wrote: {out_path}")
    if errors:
        print(f"Failed rows (showing up to 5): {errors[:5]}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"Network error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
