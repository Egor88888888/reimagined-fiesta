"""
DocLens OCR Engine
Wraps PaddleOCR with structured output.
"""
import numpy as np
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lazy-load PaddleOCR to avoid import overhead
_ocr_instances = {}


def get_ocr(lang: str = "ru", use_gpu: bool = False):
    """Get or create PaddleOCR instance (singleton per language)."""
    key = f"{lang}_{use_gpu}"
    if key not in _ocr_instances:
        from paddleocr import PaddleOCR
        _ocr_instances[key] = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            use_gpu=use_gpu,
            det_db_thresh=0.3,
            rec_batch_num=6,
            show_log=False,
        )
    return _ocr_instances[key]


@dataclass
class OCRLine:
    """Single line of recognized text."""
    text: str
    confidence: float
    bbox: list  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def __post_init__(self):
        if self.bbox:
            xs = [p[0] for p in self.bbox]
            ys = [p[1] for p in self.bbox]
            self.center_x = sum(xs) / len(xs)
            self.center_y = sum(ys) / len(ys)
            self.width = max(xs) - min(xs)
            self.height = max(ys) - min(ys)


@dataclass
class OCRResult:
    """Full OCR result for an image."""
    lines: list[OCRLine] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    languages_used: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def lines_in_region(self, x_min: float, x_max: float,
                        y_min: float, y_max: float) -> list[OCRLine]:
        """Get lines within a normalized region (0.0-1.0)."""
        result = []
        for line in self.lines:
            norm_x = line.center_x / self.image_width if self.image_width else 0
            norm_y = line.center_y / self.image_height if self.image_height else 0
            if x_min <= norm_x <= x_max and y_min <= norm_y <= y_max:
                result.append(line)
        return result

    def find_text(self, pattern: str, case_insensitive: bool = True) -> list[OCRLine]:
        """Find lines matching a text pattern."""
        import re
        flags = re.IGNORECASE if case_insensitive else 0
        return [line for line in self.lines if re.search(pattern, line.text, flags)]


class OCREngine:
    """Document OCR engine using PaddleOCR."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    def extract(self, image: np.ndarray, languages: list[str] = None) -> OCRResult:
        """Extract text from image.

        Args:
            image: OpenCV image (BGR)
            languages: List of languages to try ["ru", "en"]

        Returns:
            OCRResult with extracted lines
        """
        if languages is None:
            languages = ["ru"]

        h, w = image.shape[:2]
        all_lines: list[OCRLine] = []

        for lang in languages:
            try:
                ocr = get_ocr(lang=lang, use_gpu=self.use_gpu)
                result = ocr.ocr(image, cls=True)

                if result and result[0]:
                    for line in result[0]:
                        bbox = line[0]
                        text = line[1][0].strip()
                        confidence = float(line[1][1])

                        if text and confidence > 0.1:  # Filter noise
                            all_lines.append(OCRLine(
                                text=text,
                                confidence=confidence,
                                bbox=bbox,
                            ))
            except Exception as e:
                logger.error(f"OCR failed for lang={lang}: {e}")

        # Deduplicate overlapping lines (from multiple language runs)
        all_lines = self._deduplicate(all_lines)

        # Sort: top-to-bottom, left-to-right
        all_lines.sort(key=lambda l: (round(l.center_y / 20) * 20, l.center_x))

        return OCRResult(
            lines=all_lines,
            image_width=w,
            image_height=h,
            languages_used=languages,
        )

    def _deduplicate(self, lines: list[OCRLine]) -> list[OCRLine]:
        """Remove duplicate lines from multiple language runs."""
        if not lines:
            return lines

        unique = []
        for line in lines:
            is_dup = False
            for existing in unique:
                # Check if centers are close (within 20px)
                if (abs(line.center_x - existing.center_x) < 20 and
                        abs(line.center_y - existing.center_y) < 20):
                    # Keep the one with higher confidence
                    if line.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(line)
                    is_dup = True
                    break
            if not is_dup:
                unique.append(line)
        return unique

    def extract_mrz_zone(self, image: np.ndarray) -> OCRResult:
        """Extract text specifically from MRZ zone (bottom of document)."""
        h, w = image.shape[:2]
        # MRZ is typically in the bottom 25% of the document
        mrz_region = image[int(h * 0.7):h, :]
        return self.extract(mrz_region, languages=["en"])
