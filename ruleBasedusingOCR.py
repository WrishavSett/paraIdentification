from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


# --- Image and OCR ---

def load_image(path: str) -> Image.Image:
    """Load an image and convert to RGB."""
    return Image.open(path).convert('RGB')


def run_ocr(image_path: str, use_gpu: bool = False):
    """Run PaddleOCR on the given image path."""
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
    results = ocr.ocr(image_path, cls=True)[0]
    return results


# --- Metadata Processing ---

def extract_metadata(results) -> list:
    """Extract bounding box and text metadata from OCR results."""
    lines = []
    for box, (text, score) in results:
        if not text.strip():
            continue
        x_coords, y_coords = zip(*box)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        lines.append({
            'text': text,
            'x': x_min,
            'y': y_min,
            'w': x_max - x_min,
            'h': y_max - y_min,
            'box': box
        })
    return lines


def cluster_columns(lines: list, threshold: int = 50) -> list:
    """Group lines into columns based on horizontal proximity."""
    columns = []
    for line in sorted(lines, key=lambda l: l['x']):
        for col in columns:
            if abs(line['x'] - col['x']) < threshold:
                col['lines'].append(line)
                col['x'] = (col['x'] + line['x']) / 2  # update centroid
                break
        else:
            columns.append({'x': line['x'], 'lines': [line]})
    return columns


def group_paragraphs(column_lines: list, gap_factor: float = 1.5, merge_ratio: float = 0.5) -> list:
    """Group lines into paragraphs using vertical spacing and overlap."""
    column_lines.sort(key=lambda l: l['y'])
    paragraphs = []

    if not column_lines:
        return paragraphs

    current_para = [column_lines[0]]
    prev_line = column_lines[0]

    for line in column_lines[1:]:
        gap = line['y'] - (prev_line['y'] + prev_line['h'])
        height_thresh = gap_factor * np.mean([prev_line['h'], line['h']])
        if gap < height_thresh and gap >= -merge_ratio * line['h']:
            current_para.append(line)
        else:
            paragraphs.append(current_para)
            current_para = [line]
        prev_line = line

    if current_para:
        paragraphs.append(current_para)

    return paragraphs


# --- Visualization ---

def draw_paragraph_boxes(image: Image.Image, paragraphs: list, color=(255, 0, 0), thickness=2) -> np.ndarray:
    """Draw rectangles around grouped paragraphs."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for para in paragraphs:
        x_min = int(min(line['x'] for line in para))
        y_min = int(min(line['y'] for line in para))
        x_max = int(max(line['x'] + line['w'] for line in para))
        y_max = int(max(line['y'] + line['h'] for line in para))
        cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), color, thickness)
    return image_cv


# --- Pipeline ---

def detect_paragraphs(image_path: str, column_thresh: int = 50, gap_factor: float = 1.5, use_gpu: bool = False):
    """Full pipeline to detect paragraphs and visualize them."""
    image = load_image(image_path)
    results = run_ocr(image_path, use_gpu=use_gpu)
    lines = extract_metadata(results)
    lines.sort(key=lambda l: l['y'])

    columns = cluster_columns(lines, threshold=column_thresh)

    all_paragraphs = []
    for col in columns:
        all_paragraphs.extend(group_paragraphs(col['lines'], gap_factor=gap_factor))

    image_with_boxes = draw_paragraph_boxes(image, all_paragraphs)

    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Refined Paragraph Detection")
    plt.axis('off')
    plt.show()

    return all_paragraphs


# --- Run ---
if __name__ == "__main__":
    detect_paragraphs('./sampleImages/5.png', column_thresh=50, gap_factor=0.4, use_gpu=False)