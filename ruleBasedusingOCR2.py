from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image(path):
    return Image.open(path).convert('RGB')

def extract_metadata(results):
    lines = []
    for box, (text, score) in results:
        if not text.strip():
            continue
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
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

def cluster_columns(lines, threshold=50):
    columns = []
    for line in sorted(lines, key=lambda l: l['x']):
        added = False
        for col in columns:
            if abs(line['x'] - col['x']) < threshold:
                col['lines'].append(line)
                col['x'] = (col['x'] + line['x']) / 2  # update centroid
                added = True
                break
        if not added:
            columns.append({'x': line['x'], 'lines': [line]})
    return columns

# def group_paragraphs(column_lines, gap_factor=1.5):
#     column_lines.sort(key=lambda l: l['y'])
#     if len(column_lines) < 2:
#         return [column_lines] if column_lines else []
#
#     # Calculate dynamic average line height
#     line_heights = [l['h'] for l in column_lines]
#     avg_height = np.median(line_heights)
#     para_gap_threshold = gap_factor * avg_height
#
#     paragraphs = []
#     para = [column_lines[0]]
#
#     for i in range(1, len(column_lines)):
#         prev = column_lines[i - 1]
#         curr = column_lines[i]
#         gap = curr['y'] - (prev['y'] + prev['h'])
#
#         if gap > para_gap_threshold:
#             paragraphs.append(para)
#             para = [curr]
#         else:
#             para.append(curr)
#     if para:
#         paragraphs.append(para)
#
#     return paragraphs

def group_paragraphs(column_lines, gap_factor=1.5, merge_ratio=0.5):
    """
    Groups lines into paragraphs using dynamic spacing and line height.
    """
    column_lines.sort(key=lambda l: l['y'])
    paragraphs = []
    if not column_lines:
        return paragraphs

    current_para = [column_lines[0]]
    prev_line = column_lines[0]

    for i in range(1, len(column_lines)):
        line = column_lines[i]
        gap = line['y'] - (prev_line['y'] + prev_line['h'])

        # Adaptive threshold based on previous line height
        height_thresh = gap_factor * np.mean([prev_line['h'], line['h']])

        # Heuristic: merge if gap is small or overlap occurs
        if gap < height_thresh and gap >= -merge_ratio * line['h']:
            current_para.append(line)
        else:
            paragraphs.append(current_para)
            current_para = [line]

        prev_line = line

    if current_para:
        paragraphs.append(current_para)

    return paragraphs

def draw_paragraph_boxes(image, paragraphs, color=(255, 0, 0), thickness=2):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for para in paragraphs:
        x_vals = [l['x'] for l in para]
        y_vals = [l['y'] for l in para]
        x_ends = [l['x'] + l['w'] for l in para]
        y_ends = [l['y'] + l['h'] for l in para]
        x_min, y_min = int(min(x_vals)), int(min(y_vals))
        x_max, y_max = int(max(x_ends)), int(max(y_ends))
        cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), color, thickness)
    return image_cv

# Main pipeline
def detect_paragraphs(image_path, column_thresh=60, gap_factor=1.5):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    image = load_image(image_path)
    results = ocr.ocr(image_path, cls=True)[0]

    lines = extract_metadata(results)
    lines.sort(key=lambda l: l['y'])

    columns = cluster_columns(lines, threshold=column_thresh)

    all_paragraphs = []
    for col in columns:
        paras = group_paragraphs(col['lines'], gap_factor=gap_factor)
        all_paragraphs.extend(paras)

    image_with_boxes = draw_paragraph_boxes(image, all_paragraphs)

    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Refined Paragraph Detection")
    plt.axis('off')
    plt.show()

    return all_paragraphs

# Run
paragraphs = detect_paragraphs('./sampleImages/5.png')