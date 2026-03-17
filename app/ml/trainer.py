"""
DocLens ML Trainer
Training pipeline that learns from accumulated user corrections.

Training modes:
1. Pattern extraction — aggregates corrections into reusable rules
2. Substitution matrix — builds character-level error probability model
3. Field model — trains per-field correction model (bigram-based)

The trainer runs as a background task (Celery) or can be triggered via API.
"""
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.models import (
    FieldCorrection, CorrectionPattern, TrainingRun, MLModel,
    CorrectionStatus, TrainingStatus, ModelStatus,
)
from app.ml.feedback import get_feedback_collector
from app.ml.corrector import MODEL_DIR

logger = logging.getLogger(__name__)


class MLTrainer:
    """Trains correction models from user feedback data."""

    def __init__(self):
        self.min_corrections_for_training = 20  # Minimum data to train
        self.test_split = 0.2

    async def run_training(
        self,
        db: AsyncSession,
        document_type: str = None,
        force: bool = False,
    ) -> TrainingRun:
        """Execute a full training run.

        Steps:
        1. Collect approved corrections from DB
        2. Extract patterns (rule-based)
        3. Build substitution matrix (statistical)
        4. Build field-level bigram model
        5. Save model artifacts
        6. Record training results

        Args:
            document_type: Train only for specific doc type (None = all)
            force: Train even with insufficient data

        Returns:
            TrainingRun record with results
        """
        # Create training run record
        training_run = TrainingRun(status=TrainingStatus.RUNNING)
        training_run.started_at = datetime.now(timezone.utc)
        db.add(training_run)
        await db.flush()

        try:
            # Step 1: Load corrections
            logger.info("Training Step 1: Loading corrections...")
            corrections = await self._load_corrections(db, document_type)
            training_run.corrections_count = len(corrections)

            if len(corrections) < self.min_corrections_for_training and not force:
                training_run.status = TrainingStatus.FAILED
                training_run.error_message = (
                    f"Insufficient data: {len(corrections)} corrections, "
                    f"need {self.min_corrections_for_training}"
                )
                await db.flush()
                return training_run

            # Step 2: Extract patterns
            logger.info("Training Step 2: Extracting patterns...")
            feedback = get_feedback_collector()
            patterns = await feedback.extract_patterns(db, document_type)
            training_run.patterns_generated = len(patterns)

            # Step 3: Build substitution matrix
            logger.info("Training Step 3: Building substitution matrix...")
            sub_matrix, vocab = self._build_substitution_matrix(corrections)

            # Step 4: Build field error probability model
            logger.info("Training Step 4: Building field error model...")
            field_error_probs = self._build_field_error_model(corrections)

            # Step 5: Build character bigram model
            logger.info("Training Step 5: Building bigram model...")
            bigram_probs = self._build_bigram_model(corrections)

            # Step 6: Evaluate model quality
            logger.info("Training Step 6: Evaluating...")
            train_data, test_data = self._split_data(corrections)
            accuracy_before = self._evaluate_baseline(test_data)
            accuracy_after = self._evaluate_model(
                test_data, sub_matrix, vocab, field_error_probs
            )

            # Step 7: Save model
            logger.info("Training Step 7: Saving model...")
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = str(MODEL_DIR / f"corrector_{version}.npz")

            np.savez(
                model_path,
                substitution_matrix=sub_matrix,
                vocab=vocab,
                char_bigram_probs=bigram_probs,
                field_error_probs=field_error_probs,
                metadata=json.dumps({
                    "version": version,
                    "corrections_count": len(corrections),
                    "patterns_count": len(patterns),
                    "document_type": document_type,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                }),
            )

            # Also save as "latest"
            latest_path = str(MODEL_DIR / "corrector_latest.npz")
            np.savez(
                latest_path,
                substitution_matrix=sub_matrix,
                vocab=vocab,
                char_bigram_probs=bigram_probs,
                field_error_probs=field_error_probs,
                metadata=json.dumps({
                    "version": version,
                    "corrections_count": len(corrections),
                    "patterns_count": len(patterns),
                    "document_type": document_type,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                }),
            )

            # Step 8: Register model in DB
            ml_model = MLModel(
                name=f"field_corrector_{document_type or 'all'}",
                version=version,
                status=ModelStatus.ACTIVE,
                document_type=document_type,
                description=f"Trained on {len(corrections)} corrections, "
                            f"{len(patterns)} patterns",
                model_path=model_path,
                training_run_id=training_run.id,
                accuracy=accuracy_after,
            )
            db.add(ml_model)

            # Deactivate previous models of same type
            await db.execute(
                update(MLModel)
                .where(
                    MLModel.name == ml_model.name,
                    MLModel.version != version,
                    MLModel.status == ModelStatus.ACTIVE,
                )
                .values(status=ModelStatus.INACTIVE)
            )

            # Mark corrections as used in training
            correction_ids = [c.id for c in corrections]
            if correction_ids:
                await db.execute(
                    update(FieldCorrection)
                    .where(FieldCorrection.id.in_(correction_ids))
                    .values(status=CorrectionStatus.USED_IN_TRAINING)
                )

            # Complete training run
            training_run.status = TrainingStatus.COMPLETED
            training_run.model_version = version
            training_run.model_path = model_path
            training_run.accuracy_before = accuracy_before
            training_run.accuracy_after = accuracy_after
            training_run.completed_at = datetime.now(timezone.utc)
            training_run.duration_seconds = int(
                (training_run.completed_at - training_run.started_at).total_seconds()
            )

            # Per-field accuracy breakdown
            field_accs = self._compute_field_accuracies(
                test_data, sub_matrix, vocab, field_error_probs
            )
            training_run.field_accuracies = field_accs

            await db.commit()

            logger.info(
                f"Training completed: {version}, "
                f"accuracy {accuracy_before:.1%} -> {accuracy_after:.1%}, "
                f"duration {training_run.duration_seconds}s"
            )
            return training_run

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            training_run.status = TrainingStatus.FAILED
            training_run.error_message = str(e)
            training_run.completed_at = datetime.now(timezone.utc)
            await db.commit()
            return training_run

    async def _load_corrections(
        self,
        db: AsyncSession,
        document_type: str = None,
    ) -> list[FieldCorrection]:
        """Load corrections for training."""
        query = select(FieldCorrection).where(
            FieldCorrection.status.in_([
                CorrectionStatus.APPROVED,
                CorrectionStatus.PENDING,
            ])
        )
        if document_type:
            query = query.where(FieldCorrection.document_type == document_type)

        result = await db.execute(query)
        return list(result.scalars().all())

    def _build_substitution_matrix(
        self,
        corrections: list[FieldCorrection],
    ) -> tuple[np.ndarray, dict]:
        """Build a character substitution probability matrix.

        Matrix[i][j] = probability that character i should be replaced with j.

        Returns:
            (matrix, vocab_dict) where vocab_dict maps char -> index
        """
        # Build vocabulary from all corrections
        all_chars = set()
        for c in corrections:
            all_chars.update(c.original_value)
            all_chars.update(c.corrected_value)

        vocab = {ch: i for i, ch in enumerate(sorted(all_chars))}
        n = len(vocab)

        if n == 0:
            return np.eye(1), {}

        # Count substitutions
        counts = np.zeros((n, n), dtype=np.float64)

        for c in corrections:
            orig = c.original_value
            corr = c.corrected_value

            # Align characters using edit distance alignment
            aligned = self._align_strings(orig, corr)
            for orig_ch, corr_ch in aligned:
                if orig_ch in vocab and corr_ch in vocab:
                    counts[vocab[orig_ch]][vocab[corr_ch]] += 1

        # Normalize rows to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = counts / row_sums

        # Add identity bias (keep original char if no strong signal)
        identity = np.eye(n) * 0.3
        matrix = matrix * 0.7 + identity
        # Re-normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums

        return matrix, vocab

    def _build_field_error_model(
        self,
        corrections: list[FieldCorrection],
    ) -> dict:
        """Build per-field error probability model.

        Returns dict: {
            "passport_rf:last_name": {
                "error_rate": 0.15,
                "common_errors": [("О", "0", 0.3), ...],
                "avg_confidence_when_wrong": 0.72,
            }
        }
        """
        field_stats = defaultdict(lambda: {
            "total": 0,
            "errors": Counter(),
            "confidence_sum": 0.0,
        })

        for c in corrections:
            key = f"{c.document_type}:{c.field_name}"
            stats = field_stats[key]
            stats["total"] += 1
            stats["confidence_sum"] += c.original_confidence

            # Track character-level errors
            aligned = self._align_strings(c.original_value, c.corrected_value)
            for orig_ch, corr_ch in aligned:
                if orig_ch != corr_ch:
                    stats["errors"][(orig_ch, corr_ch)] += 1

        result = {}
        for key, stats in field_stats.items():
            total = stats["total"]
            common = [
                (orig, corr, count / total)
                for (orig, corr), count in stats["errors"].most_common(10)
            ]
            result[key] = {
                "error_rate": 1.0,  # All entries are corrections
                "common_errors": common,
                "avg_confidence_when_wrong": (
                    stats["confidence_sum"] / total if total > 0 else 0
                ),
                "sample_count": total,
            }

        return result

    def _build_bigram_model(
        self,
        corrections: list[FieldCorrection],
    ) -> dict:
        """Build character bigram probability model from correct values.

        Used to detect unlikely character sequences in OCR output.
        """
        bigrams = Counter()
        total = 0

        for c in corrections:
            text = c.corrected_value  # Use corrected (ground truth) values
            for i in range(len(text) - 1):
                bigrams[(text[i], text[i + 1])] += 1
                total += 1

        # Convert to probabilities
        result = {}
        for (ch1, ch2), count in bigrams.items():
            key = f"{ch1}{ch2}"
            result[key] = count / total if total > 0 else 0

        return result

    def _align_strings(
        self,
        s1: str,
        s2: str,
    ) -> list[tuple[str, str]]:
        """Align two strings character by character using edit distance.

        Returns list of (original_char, corrected_char) pairs.
        Insertions/deletions use empty string.
        """
        m, n = len(s1), len(s2)

        # Simple case: same length
        if m == n:
            return list(zip(s1, s2))

        # DP alignment
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1],  # substitution
                    )

        # Backtrack to get alignment
        aligned = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and (
                s1[i - 1] == s2[j - 1] or
                dp[i][j] == dp[i - 1][j - 1] + 1
            ):
                aligned.append((s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                aligned.append((s1[i - 1], ""))
                i -= 1
            else:
                aligned.append(("", s2[j - 1]))
                j -= 1

        aligned.reverse()
        # Filter out empty pairs for substitution matrix
        return [(a, b) for a, b in aligned if a and b]

    def _split_data(
        self,
        corrections: list[FieldCorrection],
    ) -> tuple[list, list]:
        """Split corrections into train/test sets."""
        n = len(corrections)
        test_n = max(1, int(n * self.test_split))

        # Sort by creation date, use newest as test
        sorted_corrections = sorted(corrections, key=lambda c: c.created_at or datetime.min)
        train = sorted_corrections[:-test_n]
        test = sorted_corrections[-test_n:]

        return train, test

    def _evaluate_baseline(self, test_data: list[FieldCorrection]) -> float:
        """Evaluate accuracy without any correction (baseline)."""
        if not test_data:
            return 0.0

        correct = sum(
            1 for c in test_data
            if c.original_value.strip() == c.corrected_value.strip()
        )
        return correct / len(test_data)

    def _evaluate_model(
        self,
        test_data: list[FieldCorrection],
        sub_matrix: np.ndarray,
        vocab: dict,
        field_error_probs: dict,
    ) -> float:
        """Evaluate model accuracy on test set."""
        if not test_data:
            return 0.0

        correct = 0
        for c in test_data:
            predicted = self._apply_model(
                c.original_value, c.document_type, c.field_name,
                sub_matrix, vocab
            )
            if predicted.strip() == c.corrected_value.strip():
                correct += 1

        return correct / len(test_data)

    def _compute_field_accuracies(
        self,
        test_data: list[FieldCorrection],
        sub_matrix: np.ndarray,
        vocab: dict,
        field_error_probs: dict,
    ) -> dict:
        """Compute per-field accuracy on test set."""
        field_results = defaultdict(lambda: {"correct": 0, "total": 0})

        for c in test_data:
            key = f"{c.document_type}:{c.field_name}"
            predicted = self._apply_model(
                c.original_value, c.document_type, c.field_name,
                sub_matrix, vocab
            )
            field_results[key]["total"] += 1
            if predicted.strip() == c.corrected_value.strip():
                field_results[key]["correct"] += 1

        return {
            key: {
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                "total": data["total"],
            }
            for key, data in field_results.items()
        }

    def _apply_model(
        self,
        value: str,
        document_type: str,
        field_name: str,
        sub_matrix: np.ndarray,
        vocab: dict,
    ) -> str:
        """Apply substitution matrix model to a value."""
        corrected = []
        for ch in value:
            if ch in vocab:
                ch_idx = vocab[ch]
                if ch_idx < sub_matrix.shape[0]:
                    row = sub_matrix[ch_idx]
                    best_idx = np.argmax(row)
                    if row[best_idx] > 0.6 and best_idx != ch_idx:
                        reverse_vocab = {v: k for k, v in vocab.items()}
                        if best_idx in reverse_vocab:
                            corrected.append(reverse_vocab[best_idx])
                            continue
            corrected.append(ch)
        return "".join(corrected)

    async def get_training_status(self, db: AsyncSession) -> dict:
        """Get current training system status."""
        # Last training run
        result = await db.execute(
            select(TrainingRun)
            .order_by(TrainingRun.created_at.desc())
            .limit(1)
        )
        last_run = result.scalar_one_or_none()

        # Active models count
        model_result = await db.execute(
            select(MLModel).where(MLModel.status == ModelStatus.ACTIVE)
        )
        active_models = model_result.scalars().all()

        # Pending corrections count
        pending_result = await db.execute(
            select(FieldCorrection).where(
                FieldCorrection.status.in_([
                    CorrectionStatus.PENDING,
                    CorrectionStatus.APPROVED,
                ])
            )
        )
        pending = pending_result.scalars().all()

        return {
            "last_training_run": {
                "id": str(last_run.id) if last_run else None,
                "status": last_run.status.value if last_run else None,
                "accuracy_before": last_run.accuracy_before if last_run else None,
                "accuracy_after": last_run.accuracy_after if last_run else None,
                "corrections_used": last_run.corrections_count if last_run else 0,
                "completed_at": (
                    last_run.completed_at.isoformat() if last_run and last_run.completed_at
                    else None
                ),
            },
            "active_models": [
                {
                    "name": m.name,
                    "version": m.version,
                    "accuracy": m.accuracy,
                    "document_type": m.document_type,
                }
                for m in active_models
            ],
            "pending_corrections": len(pending),
            "ready_for_training": len(pending) >= self.min_corrections_for_training,
        }


# Singleton
_trainer = None


def get_trainer() -> MLTrainer:
    global _trainer
    if _trainer is None:
        _trainer = MLTrainer()
    return _trainer
