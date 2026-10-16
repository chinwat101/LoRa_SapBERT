"""
src/federated/aggregator.py

Custom FedAvg Strategy ที่ log F1 แต่ละ round
และบันทึก global LoRA adapter ลงดิสก์
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FedAvgWithLogging(fl.server.strategy.FedAvg):
    """
    FedAvg Strategy ที่:
    1. Log F1 score ของแต่ละ client และค่าเฉลี่ย global ทุก round
    2. บันทึก global LoRA adapter parameters ลงดิสก์ทุก N rounds
    """

    def __init__(self, output_dir: str, save_every_n_rounds: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_rounds = save_every_n_rounds
        self.round_results: List[dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """รวม LoRA weights จากทุก clients ด้วย FedAvg (weighted by dataset size)"""
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            # Log F1 ของแต่ละ client
            client_f1s = []
            for _, fit_res in results:
                client_f1 = fit_res.metrics.get("eval_f1", 0.0)
                n_samples = fit_res.num_examples
                client_f1s.append((client_f1, n_samples))
                logger.info(f"  [Round {server_round}] Client ({n_samples} samples) — F1: {client_f1:.4f}")

            # Weighted average F1
            total_samples = sum(n for _, n in client_f1s)
            avg_f1 = sum(f1 * n for f1, n in client_f1s) / total_samples if total_samples > 0 else 0.0
            logger.info(f"[Round {server_round}] 🌐 Global Weighted F1: {avg_f1:.4f}")

            # บันทึก round result
            self.round_results.append({"round": server_round, "global_f1": avg_f1})
            self._save_round_log()

            # บันทึก LoRA adapter weights
            if server_round % self.save_every_n_rounds == 0:
                self._save_global_adapter(server_round, aggregated_params)

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """รวม evaluation metrics จากทุก clients"""
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if results:
            f1_scores = [(r.metrics.get("eval_f1", 0.0), r.num_examples) for _, r in results]
            total = sum(n for _, n in f1_scores)
            avg_f1 = sum(f1 * n for f1, n in f1_scores) / total if total > 0 else 0.0
            logger.info(f"[Round {server_round}] 📊 Eval Global F1: {avg_f1:.4f}")
            metrics["eval_f1"] = avg_f1

        return aggregated_loss, metrics

    def _save_round_log(self):
        """บันทึก F1 history ของแต่ละ round เป็น JSON"""
        log_path = self.output_dir / "fl_training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.round_results, f, indent=2, ensure_ascii=False)

    def _save_global_adapter(self, server_round: int, parameters: Parameters):
        """บันทึก global LoRA adapter weights ลงดิสก์ (เป็น numpy arrays)"""
        round_dir = self.output_dir / f"round_{server_round:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        ndarrays = parameters_to_ndarrays(parameters)
        np.save(str(round_dir / "global_adapter_weights.npy"), np.array(ndarrays, dtype=object))
        logger.info(f"[Round {server_round}] 💾 Global adapter saved → {round_dir}")
