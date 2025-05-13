import os
import uuid
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Root folder and output folder
root_folder = 'C:/Users/datacore/Downloads/Alignment'
output_folder = './ProcessedOutput'
os.makedirs(output_folder, exist_ok=True)

# Loop through folders 1 to 50
for folder_num in range(1, 51):
    folder_path = os.path.join(root_folder, str(folder_num))
    
    # Check for image existence
    jpg_path = os.path.join(folder_path, 'Correct.jpg')
    png_path = os.path.join(folder_path, 'Correct.png')
    
    image_path = None
    if os.path.exists(jpg_path):
        image_path = jpg_path
    elif os.path.exists(png_path):
        image_path = png_path
    else:
        print(f"[{folder_num}] No image found.")
        continue

    print(f"Processing: {image_path}")
    
    # Load and OCR
    image = Image.open(image_path).convert('RGB')
    ocr_result = ocr.ocr(image_path, cls=True)

    line_boxes = [line[0] for line in ocr_result[0] if line[1][0].strip()]

    def get_center_y(box): return (box[0][1] + box[3][1]) / 2
    def get_start_x(box): return min(box[0][0], box[3][0])

    lines_with_meta = [(box, get_center_y(box), get_start_x(box)) for box in line_boxes]
    lines_sorted = sorted(lines_with_meta, key=lambda x: x[1])

    y_centers = [line[1] for line in lines_sorted]
    vertical_gaps = [y2 - y1 for y1, y2 in zip(y_centers[:-1], y_centers[1:])]
    median_gap = np.median(vertical_gaps) if vertical_gaps else 0
    vertical_thresh = median_gap * 1.3

    start_xs = [line[2] for line in lines_sorted]
    q1, q3 = np.percentile(start_xs, [25, 75]) if start_xs else (0, 0)
    iqr = q3 - q1
    indent_thresh = iqr * 1.5

    paragraphs = []
    if lines_sorted:
        current_paragraph = [lines_sorted[0][0]]
        prev_y = lines_sorted[0][1]
        prev_x = lines_sorted[0][2]

        for i in range(1, len(lines_sorted)):
            box, curr_y, curr_x = lines_sorted[i]
            if abs(curr_y - prev_y) > vertical_thresh or (curr_x - prev_x) > indent_thresh:
                paragraphs.append(current_paragraph)
                current_paragraph = [box]
            else:
                current_paragraph.append(box)
            prev_y, prev_x = curr_y, curr_x

        paragraphs.append(current_paragraph)

    def merge_boxes(boxes):
        xs = [pt[0] for box in boxes for pt in box]
        ys = [pt[1] for box in boxes for pt in box]
        return [[min(xs), min(ys)], [max(xs), min(ys)],
                [max(xs), max(ys)], [min(xs), max(ys)]]

    paragraph_boxes = [merge_boxes(p) for p in paragraphs]

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for box in paragraph_boxes:
        draw.polygon([tuple(pt) for pt in box], outline='blue', width=1)

    # Save image with UUID
    unique_filename = f"{uuid.uuid4()}.png"
    output_path = os.path.join(output_folder, unique_filename)
    image.save(output_path)
    print(f"Saved: {output_path}")