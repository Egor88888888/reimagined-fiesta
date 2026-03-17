"""
DocLens Pipeline Orchestrator
Coordinates the full recognition pipeline.
"""
import time
import logging
import numpy as np
from dataclasses import dataclass, field, asdict

from app.core.preprocessor import DocumentPreprocessor
from app.core.ocr_engine import OCREngine, OCRResult
from app.core.classifier import DocumentClassifier
from app.models.database import DocumentType
from app.parsers import get_parser
from app.validators.checksum import validate_document_fields
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class FieldResult:
    """Single extracted field."""
    value: str
    confidence: float
    source: str = "ocr"           # "ocr" | "mrz" | "enrichment"
    is_valid: bool | None = None  # None = not validated
    auto_fill: bool = False       # Above auto-fill threshold?


@dataclass
class RecognitionResult:
    """Full recognition result."""
    document_type: str = "unknown"
    classification_confidence: float = 0.0
    overall_confidence: float = 0.0
    fields: dict[str, dict] = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    processing_time_ms: int = 0
    raw_ocr_text: str = ""

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type,
            "classification_confidence": round(self.classification_confidence, 3),
            "overall_confidence": round(self.overall_confidence, 3),
            "fields": self.fields,
            "validation": self.validation,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
        }


class RecognitionPipeline:
    """Main document recognition pipeline."""

    def __init__(self):
        self.preprocessor = DocumentPreprocessor(max_size=settings.OCR_MAX_IMAGE_SIZE)
        self.ocr_engine = OCREngine(use_gpu=settings.OCR_USE_GPU)
        self.classifier = DocumentClassifier()

    def process(self, image_bytes: bytes, document_type_hint: str = None) -> RecognitionResult:
        """Process a document image through the full pipeline.

        Args:
            image_bytes: Raw image bytes
            document_type_hint: Optional type hint from client

        Returns:
            RecognitionResult with extracted fields
        """
        start_time = time.time()
        result = RecognitionResult()

        try:
            # Step 1: Preprocess image
            logger.info("Step 1: Preprocessing image...")
            image = self.preprocessor.process(image_bytes)

            # Step 2: OCR extraction
            logger.info("Step 2: Running OCR...")
            ocr_result = self.ocr_engine.extract(image, languages=["ru", "en"])
            result.raw_ocr_text = ocr_result.full_text

            if not ocr_result.lines:
                result.warnings.append("No text detected in image")
                return result

            # Step 3: Classify document
            logger.info("Step 3: Classifying document...")
            doc_type, cls_confidence = self.classifier.classify(ocr_result, hint=document_type_hint)
            result.document_type = doc_type.value
            result.classification_confidence = cls_confidence

            if doc_type == DocumentType.UNKNOWN:
                result.warnings.append("Could not determine document type")
                return result

            # Step 4: Parse fields with type-specific parser
            logger.info(f"Step 4: Parsing fields for {doc_type.value}...")
            parser = get_parser(doc_type)
            if parser:
                parsed_fields = parser.parse(ocr_result, image)
                result.fields = self._format_fields(parsed_fields)
            else:
                result.warnings.append(f"No parser available for {doc_type.value}")

            # Step 5: Validate extracted fields
            logger.info("Step 5: Validating fields...")
            validation = validate_document_fields(doc_type, result.fields)
            result.validation = validation

            # Apply validation results to warnings
            for field_name, val_result in validation.items():
                if not val_result.get("valid", True):
                    result.warnings.append(
                        f"{field_name}: {val_result.get('message', 'validation failed')}"
                    )

            # Calculate overall confidence
            if result.fields:
                confidences = [
                    f.get("confidence", 0)
                    for f in result.fields.values()
                    if isinstance(f, dict)
                ]
                result.overall_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0
                )

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            result.warnings.append(f"Processing error: {str(e)}")

        finally:
            elapsed = (time.time() - start_time) * 1000
            result.processing_time_ms = int(elapsed)
            logger.info(f"Pipeline completed in {elapsed:.0f}ms")

        return result

    def _format_fields(self, parsed_fields: dict[str, FieldResult]) -> dict:
        """Format parsed fields with auto-fill flags."""
        formatted = {}
        for name, field_result in parsed_fields.items():
            if isinstance(field_result, FieldResult):
                auto_fill = field_result.confidence >= settings.CONFIDENCE_AUTO_FILL
                needs_review = (
                    settings.CONFIDENCE_REVIEW <= field_result.confidence < settings.CONFIDENCE_AUTO_FILL
                )
                formatted[name] = {
                    "value": field_result.value,
                    "confidence": round(field_result.confidence, 3),
                    "source": field_result.source,
                    "auto_fill": auto_fill,
                    "needs_review": needs_review,
                }
            elif isinstance(field_result, dict):
                formatted[name] = field_result
        return formatted


# Singleton
_pipeline = None


def get_pipeline() -> RecognitionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RecognitionPipeline()
    return _pipeline
