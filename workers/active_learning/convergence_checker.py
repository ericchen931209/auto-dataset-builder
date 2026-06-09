"""
Convergence Checker — Active Learning termination conditions.

The active learning loop stops when ANY of these conditions is met:

  1. DQS >= dqs_threshold          (dataset quality is good enough)
  2. iterations >= max_iterations  (safety limit)
  3. delta_dqs < min_delta         (DQS improvement stalled)
  4. uncertain_count == 0          (no more uncertain images to re-annotate)
"""
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StopReason(str, Enum):
    DQS_THRESHOLD     = "dqs_threshold"
    MAX_ITERATIONS    = "max_iterations"
    DQS_STALLED       = "dqs_stalled"
    NO_UNCERTAIN      = "no_uncertain_images"
    NOT_STOPPED       = "not_stopped"


@dataclass
class IterationRecord:
    iteration: int
    dqs_score: float
    uncertain_count: int
    delta_dqs: float = 0.0      # change from previous iteration
    notes: str = ""


@dataclass
class ConvergenceResult:
    should_stop: bool
    reason: StopReason
    iteration: int
    history: list[IterationRecord] = field(default_factory=list)


class ConvergenceChecker:
    """
    Stateful checker that tracks DQS history across iterations.

    Usage::

        checker = ConvergenceChecker(dqs_threshold=0.75, max_iterations=10)
        while True:
            dqs  = evaluate_dqs(dataset)
            uncertain = sample_uncertain(dataset)
            result = checker.step(dqs, len(uncertain))
            if result.should_stop:
                break
            re_annotate(uncertain)
    """

    def __init__(
        self,
        dqs_threshold: float = 0.75,
        max_iterations: int = 10,
        min_delta: float = 0.005,
        window: int = 2,
    ):
        """
        Args:
            dqs_threshold: Stop when DQS >= this value.
            max_iterations: Hard iteration cap.
            min_delta: Stop if DQS improvement over last `window` iterations
                       is less than this value (stall detection).
            window: Number of past iterations used for stall detection.
        """
        self.dqs_threshold  = dqs_threshold
        self.max_iterations = max_iterations
        self.min_delta      = min_delta
        self.window         = window
        self._history: list[IterationRecord] = []

    @property
    def iteration(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[IterationRecord]:
        return list(self._history)

    def step(self, dqs_score: float, uncertain_count: int) -> ConvergenceResult:
        """
        Record one active-learning iteration and decide whether to stop.

        Args:
            dqs_score: Current DQS score (0.0–1.0).
            uncertain_count: Number of uncertain images found this iteration.

        Returns:
            ConvergenceResult indicating whether the loop should stop.
        """
        prev_dqs  = self._history[-1].dqs_score if self._history else None
        delta_dqs = (dqs_score - prev_dqs) if prev_dqs is not None else 0.0

        record = IterationRecord(
            iteration=self.iteration + 1,
            dqs_score=dqs_score,
            uncertain_count=uncertain_count,
            delta_dqs=delta_dqs,
        )
        self._history.append(record)

        logger.info(
            f"[AL iteration {record.iteration}] "
            f"DQS={dqs_score:.4f} Δ={delta_dqs:+.4f} uncertain={uncertain_count}"
        )

        # ── Termination checks ────────────────────────────────────────────────

        if dqs_score >= self.dqs_threshold:
            record.notes = f"DQS {dqs_score:.4f} >= threshold {self.dqs_threshold}"
            logger.info(f"[AL] Converged: {record.notes}")
            return self._stop(StopReason.DQS_THRESHOLD)

        if self.iteration >= self.max_iterations:
            record.notes = f"Reached max_iterations={self.max_iterations}"
            logger.info(f"[AL] Stopping: {record.notes}")
            return self._stop(StopReason.MAX_ITERATIONS)

        if uncertain_count == 0:
            record.notes = "No uncertain images remain"
            logger.info(f"[AL] Stopping: {record.notes}")
            return self._stop(StopReason.NO_UNCERTAIN)

        if len(self._history) > self.window:
            window_scores = [r.dqs_score for r in self._history[-self.window:]]
            window_delta  = window_scores[-1] - window_scores[0]
            if abs(window_delta) < self.min_delta:
                record.notes = (
                    f"DQS stalled: Δ={window_delta:.5f} < "
                    f"min_delta={self.min_delta} over {self.window} iterations"
                )
                logger.info(f"[AL] Stopping: {record.notes}")
                return self._stop(StopReason.DQS_STALLED)

        return ConvergenceResult(
            should_stop=False,
            reason=StopReason.NOT_STOPPED,
            iteration=self.iteration,
            history=self.history,
        )

    def _stop(self, reason: StopReason) -> ConvergenceResult:
        return ConvergenceResult(
            should_stop=True,
            reason=reason,
            iteration=self.iteration,
            history=self.history,
        )

    def summary(self) -> dict:
        """Return a JSON-serialisable summary of the run."""
        return {
            "total_iterations": self.iteration,
            "final_dqs": self._history[-1].dqs_score if self._history else None,
            "dqs_threshold": self.dqs_threshold,
            "history": [
                {
                    "iteration": r.iteration,
                    "dqs_score": round(r.dqs_score, 4),
                    "delta_dqs": round(r.delta_dqs, 4),
                    "uncertain_count": r.uncertain_count,
                    "notes": r.notes,
                }
                for r in self._history
            ],
        }
