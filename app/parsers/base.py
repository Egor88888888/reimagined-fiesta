"""
DocLens Base Document Parser
"""
import re
import numpy as np
from abc import ABC, abstractmethod
from app.core.ocr_engine import OCRResult, OCRLine
from app.core.orchestrator import FieldResult


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, ocr_result: OCRResult, image: np.ndarray) -> dict[str, FieldResult]:
        """Extract structured fields from OCR result.

        Args:
            ocr_result: OCR extraction result
            image: Preprocessed image (for zone-based extraction)

        Returns:
            Dictionary of field_name -> FieldResult
        """
        ...

    # ========================
    # Helper methods
    # ========================

    def find_by_label(self, lines: list[OCRLine], label_pattern: str,
                      same_line: bool = True, next_line: bool = True,
                      right_of: bool = True) -> OCRLine | None:
        """Find value associated with a label.

        Tries multiple strategies:
        1. Value in same line after label (e.g., "ФИО: Иванов")
        2. Value in next line below label
        3. Value to the right of label
        """
        for i, line in enumerate(lines):
            match = re.search(label_pattern, line.text, re.IGNORECASE)
            if not match:
                continue

            # Strategy 1: Value after label in same line
            if same_line:
                remaining = line.text[match.end():].strip()
                remaining = re.sub(r'^[:\s]+', '', remaining)
                if remaining and len(remaining) > 1:
                    return OCRLine(
                        text=remaining,
                        confidence=line.confidence,
                        bbox=line.bbox,
                    )

            # Strategy 2: Next line below
            if next_line and i + 1 < len(lines):
                next_l = lines[i + 1]
                # Check if next line is close vertically
                if abs(next_l.center_x - line.center_x) < line.width * 0.5:
                    return next_l

            # Strategy 3: Line to the right
            if right_of:
                for other in lines:
                    if (other.center_y - line.center_y) < line.height * 0.5 and \
                       other.center_x > line.center_x + line.width * 0.3 and \
                       other is not line:
                        return other

        return None

    def find_pattern(self, lines: list[OCRLine], pattern: str) -> tuple[str, float] | None:
        """Find first line matching a regex pattern.

        Returns:
            Tuple of (matched_text, confidence) or None
        """
        for line in lines:
            m = re.search(pattern, line.text)
            if m:
                return m.group(0), line.confidence
        return None

    def extract_date(self, lines: list[OCRLine], context_pattern: str = None) -> tuple[str, float] | None:
        """Extract a date from OCR lines.

        Args:
            context_pattern: Optional pattern to narrow search area
        """
        date_pattern = r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})"

        search_lines = lines
        if context_pattern:
            # Find lines near the context
            for i, line in enumerate(lines):
                if re.search(context_pattern, line.text, re.IGNORECASE):
                    # Search in a window around the context
                    start = max(0, i - 1)
                    end = min(len(lines), i + 3)
                    search_lines = lines[start:end]
                    break

        for line in search_lines:
            m = re.search(date_pattern, line.text)
            if m:
                date_str = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
                return date_str, line.confidence
        return None

    def lines_in_zone(self, ocr_result: OCRResult,
                      x_range: tuple[float, float],
                      y_range: tuple[float, float]) -> list[OCRLine]:
        """Get OCR lines within a normalized zone (0.0-1.0)."""
        return ocr_result.lines_in_region(
            x_min=x_range[0], x_max=x_range[1],
            y_min=y_range[0], y_max=y_range[1],
        )

    def clean_text(self, text: str) -> str:
        """Clean OCR artifacts from text."""
        # Remove common OCR noise
        text = re.sub(r'[|_~`]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
