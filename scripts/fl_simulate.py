#!/usr/bin/env python3
"""
scripts/fl_simulate.py

Federated Learning Simulation บนเครื่องเดียว — Manual FL Loop (ไม่ต้องการ Ray)
ใช้ 2 datasets เป็น 2 clients (จำลองโรงพยาบาล 2 แห่ง)

Client 0 — Hospital_A_QA   : data/processed/ner_qa_train.jsonl  (29,709 records)
Client 1 — Hospital_B_Label: data/processed/ner_bio_th.jsonl     (44 records)

Usage:
    python scripts/fl_simulate.py
    python scripts/fl_simulate.py --base-model outputs/merged_model_3_ner_1 --rounds 5
    python scripts/fl_simulate.py --config configs/fl_config.yaml --rounds 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.federated.client import NERLoRAClient
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__, log_dir=Path("outputs/logs"))

DEFAULT_LABEL2ID = {
    "O": 0, "B-DRUG": 1, "I-DRUG": 2,
    "B-DISEASE": 3, "I-DISEASE": 4,
    "B-SYMPTOM": 5, "I-SYMPTOM": 6,
}
DEFAULT_ID2LABEL = {v: k for k, v in DEFAULT_LABEL2ID.items()}


def load_label_schema(schema_path: str) -> tuple:
    p = Path(schema_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            schema = json.load(f)
        label2id = schema["label2id"]
        id2label = {int(k): v for k, v in schema["id2label"].items()}
        return label2id, id2label
    logger.warning(f"Label schema not found: {schema_path}, using defaults.")
    return DEFAULT_LABEL2ID, DEFAULT_ID2LABEL


def fedavg(
    weights_list: list[list[np.ndarray]],
    num_samples_list: list[int],
    strategy: str = "weighted",
) -> list[np.ndarray]:
    """
    Federated Averaging: รวม LoRA weights จากทุก clients
    - weighted: ถ่วงน้ำหนักด้วยจำนวน samples ของแต่ละ client
    - uniform : ให้ทุก client มีน้ำหนักเท่ากัน (1/N) ป้องกัน client เล็กถูกละเลย
    """
    if strategy == "uniform":
        weights = [1.0 / len(num_samples_list)] * len(num_samples_list)
    else:
        total_samples = sum(num_samples_list)
        weights = [n / total_samples for n in num_samples_list] if total_samples > 0 else [1.0 / len(num_samples_list)] * len(num_samples_list)

    averaged = []
    for layer_weights in zip(*weights_list):
        weighted_sum = sum(
            w * weight
            for w, weight in zip(layer_weights, weights)
        )
        averaged.append(weighted_sum)
    return averaged


def save_round_adapter(global_weights: list[np.ndarray], output_dir: Path, round_num: int):
    """บันทึก global LoRA adapter weights ของ round นั้นๆ"""
    round_dir = output_dir / f"round_{round_num:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(round_dir / "global_adapter_weights.npy"),
            np.array(global_weights, dtype=object))
    logger.info(f"💾 Global adapter saved → {round_dir}")


def main():
    parser = argparse.ArgumentParser(description="FL Simulation: 2 clients บนเครื่องเดียว")
    parser.add_argument("--config", default="configs/fl_config.yaml")
    parser.add_argument("--base-model", default=None, help="Path to base NER model")
    parser.add_argument("--rounds", type=int, default=None, help="Override FL rounds")
    args = parser.parse_args()

    # ─── โหลด Config ─────────────────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fl_cfg       = cfg["federated"]
    client_cfgs  = cfg["clients"]
    lora_cfg     = cfg["lora"]
    training_cfg = cfg["client"]
    model_cfg    = cfg["model"]
    data_cfg     = cfg["data"]

    agg_strategy= fl_cfg.get("aggregation", "weighted")

    num_rounds  = args.rounds or fl_cfg["num_rounds"]
    base_model  = args.base_model or model_cfg["base_model_path"]
    output_dir  = Path(model_cfg["output_dir"])
    seed        = cfg.get("training", {}).get("seed", 42)

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FEDERATED LEARNING SIMULATION (NER LoRA — Manual Loop)")
    logger.info("=" * 60)
    logger.info(f"Base Model   : {base_model}")
    logger.info(f"FL Rounds    : {num_rounds}")
    logger.info(f"Aggregation  : {agg_strategy}")
    logger.info(f"Clients      : {len(client_cfgs)}")
    for c in client_cfgs:
        epochs = c.get("local_epochs", training_cfg.get("local_epochs", 2))
        logger.info(f"  - Client {c['id']} [{c['name']}]: {c['train_file']} (local_epochs={epochs})")
    logger.info(f"Output Dir   : {output_dir}")
    logger.info("=" * 60)

    # ─── Label Schema ─────────────────────────────────────────────────────────
    label2id, id2label = load_label_schema(data_cfg["label_schema_file"])
    logger.info(f"Labels ({len(label2id)}): {list(label2id.keys())}")

    # ─── สร้าง Clients ────────────────────────────────────────────────────────
    clients: list[NERLoRAClient] = []
    for c in client_cfgs:
        c_training_cfg = dict(training_cfg)
        if "local_epochs" in c:
            c_training_cfg["local_epochs"] = c["local_epochs"]

        client = NERLoRAClient(
            client_id   = c["id"],
            client_name = c["name"],
            model_path  = base_model,
            train_file  = c["train_file"],
            lora_cfg    = lora_cfg,
            training_cfg= c_training_cfg,
            label2id    = label2id,
            id2label    = id2label,
            seed        = seed + c["id"],
        )
        clients.append(client)
        logger.info(f"✅ Client {c['id']} ready — {c['name']} "
                    f"(train={len(client.train_dataset)}, val={len(client.eval_dataset)}, epochs={c_training_cfg['local_epochs']})")

    # ─── Initial global weights (LoRA weights จาก Client 0) ──────────────────
    global_weights = clients[0].get_parameters({})
    logger.info(f"\n📐 LoRA adapter size: {len(global_weights)} weight tensors "
                f"({sum(w.nbytes for w in global_weights) / 1024:.1f} KB)")

    # ─── Manual FL Loop ────────────────────────────────────────────────────────
    round_logs = []

    logger.info(f"\n🚀 Starting FL — {num_rounds} rounds × {len(clients)} clients\n")

    for fl_round in range(1, num_rounds + 1):
        logger.info(f"{'─'*50}")
        logger.info(f"  ROUND {fl_round}/{num_rounds}")
        logger.info(f"{'─'*50}")

        round_weights = []
        round_sizes   = []
        round_f1s     = []

        # ── 1. แต่ละ client รับ global weights → เทรน → ส่งกลับ ─────────────
        for client in clients:
            logger.info(f"\n  [{client.client_name}] fit() Round {fl_round}...")

            updated_weights, n_samples, metrics = client.fit(
                parameters=global_weights,
                config={"fl_round": fl_round},
            )
            client_f1 = metrics.get("eval_f1", 0.0)

            logger.info(f"  [{client.client_name}] ✓ n_samples={n_samples} | F1={client_f1:.4f}")

            round_weights.append(updated_weights)
            round_sizes.append(n_samples)
            round_f1s.append(client_f1)

        # ── 2. FedAvg — รวม LoRA weights (ตาม aggregation strategy) ───────────
        global_weights = fedavg(round_weights, round_sizes, strategy=agg_strategy)

        # ── 3. Broadcast global weights กลับให้ทุก client → evaluate ──────────
        eval_f1s = []
        for client in clients:
            loss, n_eval, eval_metrics = client.evaluate(
                parameters=global_weights,
                config={"fl_round": fl_round},
            )
            eval_f1s.append((eval_metrics.get("eval_f1", 0.0), n_eval))

        if agg_strategy == "uniform":
            global_f1 = sum(f1 for f1, _ in eval_f1s) / len(eval_f1s) if eval_f1s else 0.0
        else:
            total_eval = sum(n for _, n in eval_f1s)
            global_f1 = sum(f1 * n for f1, n in eval_f1s) / total_eval if total_eval > 0 else 0.0

        logger.info(f"\n  🌐 [Round {fl_round}] Global F1 ({agg_strategy}): {global_f1:.4f}")
        logger.info(f"  Client F1s: {[f'{f:.4f}' for f in round_f1s]}")

        # ── 4. บันทึก round result ────────────────────────────────────────────
        round_result = {
            "round": fl_round,
            "global_f1": global_f1,
            "client_f1s": round_f1s,
            "client_names": [c.client_name for c in clients],
        }
        round_logs.append(round_result)

        log_path = output_dir / "fl_training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(round_logs, f, indent=2, ensure_ascii=False)

        save_round_adapter(global_weights, output_dir, fl_round)

    # ─── สรุปผลสุดท้าย ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FL SIMULATION COMPLETE")
    logger.info("=" * 60)
    logger.info("F1 per round:")
    for r in round_logs:
        clients_str = " | ".join(
            f"{n}: {f:.4f}" for n, f in zip(r["client_names"], r["client_f1s"])
        )
        logger.info(f"  Round {r['round']:2d} → Global: {r['global_f1']:.4f}  [{clients_str}]")

    best = max(round_logs, key=lambda x: x["global_f1"])
    logger.info(f"\n🏆 Best Round: {best['round']} — Global F1: {best['global_f1']:.4f}")
    logger.info(f"\n📁 All results saved in: {output_dir}/")
    logger.info(f"   - fl_training_log.json")
    logger.info(f"   - round_01/ ... round_{num_rounds:02d}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
