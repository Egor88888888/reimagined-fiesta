"""
DocLens Image Preprocessor
Handles image normalization before OCR.
"""
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Preprocesses document images for optimal OCR results."""

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size

    def process(self, image_bytes: bytes) -> np.ndarray:
        """Full preprocessing pipeline."""
        image = self._bytes_to_cv2(image_bytes)
        if image is None:
            raise ValueError("Cannot decode image")

        image = self._resize_if_needed(image)
        image = self._correct_orientation(image)
        image = self._detect_and_crop_document(image)
        image = self._deskew(image)
        image = self._enhance_contrast(image)

        return image

    def _bytes_to_cv2(self, image_bytes: bytes) -> np.ndarray | None:
        """Convert bytes to OpenCV image."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize if image is too large."""
        h, w = image.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def _correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct image orientation."""
        # PaddleOCR has built-in angle classifier, so we do basic checks only
        h, w = image.shape[:2]
        # If image is portrait but wider than tall, it might be rotated
        # We leave complex rotation to PaddleOCR's cls module
        return image

    def _detect_and_crop_document(self, image: np.ndarray) -> np.ndarray:
        """Detect document boundaries and crop."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / (image.shape[0] * image.shape[1])

        # Only crop if document takes significant portion of image
        if area_ratio < 0.2:
            return image

        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) == 4:
            return self._four_point_transform(image, approx.reshape(4, 2))

        # Fallback: use bounding rectangle
        x, y, w, h = cv2.boundingRect(largest)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        return image[y:y + h, x:x + w]

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform to get top-down view of document."""
        rect = self._order_points(pts.astype(np.float32))
        (tl, tr, br, bl) = rect

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct text skew."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        if abs(angle) < 0.5:
            return image
        if abs(angle) > 15:
            return image  # Too much rotation, probably wrong detection

        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def cv2_to_bytes(image: np.ndarray, format: str = ".png") -> bytes:
        """Convert OpenCV image back to bytes."""
        success, buffer = cv2.imencode(format, image)
        if not success:
            raise ValueError(f"Failed to encode image to {format}")
        return buffer.tobytes()
