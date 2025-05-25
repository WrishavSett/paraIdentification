import os
import uuid
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

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

    # ----------- Step 2: Create Metadata -----------
    metadata_list = []
    for line in ocr_result[0]:
        box, (text, score) = line
        if not text.strip():
            continue  # Skip empty lines

        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        metadata_list.append({
            'text': text.strip(),
            'x': x_min,
            'y': y_min,
            'w': x_max - x_min,
            'h': y_max - y_min,
            'box': box,
            'center_y': (box[0][1] + box[3][1]) / 2,
            'start_x': min(box[0][0], box[3][0])
        })

    # ----------- Step 3: DBSCAN for Column Detection -----------
    x_starts = np.array([[m['start_x']] for m in metadata_list])
    db = DBSCAN(eps=40, min_samples=2).fit(x_starts)
    labels = db.labels_

    for meta, label in zip(metadata_list, labels):
        meta['column_order'] = label

    # Keep only clustered lines (ignore noise)
    valid_metadata = [m for m in metadata_list if m['column_order'] != -1]
    unique_columns = sorted(set(m['column_order'] for m in valid_metadata))


    # ----------- Step 4: Enhanced Paragraph Grouping -----------
    def group_into_paragraphs(lines):
        if not lines:
            return []

        lines_sorted = sorted(lines, key=lambda x: x['center_y'])

        # Compute vertical gap threshold
        y_centers = [line['center_y'] for line in lines_sorted]
        vertical_gaps = [y2 - y1 for y1, y2 in zip(y_centers[:-1], y_centers[1:])]
        median_gap = np.median(vertical_gaps) if vertical_gaps else 0
        vertical_thresh = median_gap * 1.3

        # Compute indent threshold via IQR
        start_xs = [line['start_x'] for line in lines_sorted]
        if len(start_xs) >= 4:
            q1, q3 = np.percentile(start_xs, [25, 75])
            iqr = q3 - q1
            indent_thresh = iqr * 1.5
        else:
            indent_thresh = 40  # fallback

        paragraphs = []
        current_paragraph = [lines_sorted[0]]
        prev_y = lines_sorted[0]['center_y']
        prev_x = lines_sorted[0]['start_x']

        for curr in lines_sorted[1:]:
            curr_y = curr['center_y']
            curr_x = curr['start_x']

            is_new_para = (
                    abs(curr_y - prev_y) > vertical_thresh or
                    (curr_x - prev_x) > indent_thresh
            )

            if is_new_para:
                paragraphs.append(current_paragraph)
                current_paragraph = [curr]
            else:
                current_paragraph.append(curr)

            prev_y, prev_x = curr_y, curr_x

        paragraphs.append(current_paragraph)  # Add the last paragraph
        return paragraphs


    # ----------- Step 5: Group Lines by Column and Paragraphs -----------
    all_paragraphs = []
    for col_id in unique_columns:
        col_lines = [m for m in valid_metadata if m['column_order'] == col_id]
        col_paragraphs = group_into_paragraphs(col_lines)
        all_paragraphs.extend(col_paragraphs)


    # ----------- Step 6: Merge Boxes for Paragraphs -----------
    def merge_boxes(paragraph):
        all_pts = [pt for line in paragraph for pt in line['box']]
        xs = [pt[0] for pt in all_pts]
        ys = [pt[1] for pt in all_pts]
        return [[min(xs), min(ys)], [max(xs), min(ys)],
                [max(xs), max(ys)], [min(xs), max(ys)]]


    paragraph_boxes = [merge_boxes(p) for p in all_paragraphs]

    # ----------- Step 7: Draw on Image -----------
    draw = ImageDraw.Draw(image)
    for box in paragraph_boxes:
        draw.polygon([tuple(pt) for pt in box], outline='blue', width=2)

    # Save image with UUID
    unique_filename = f"segment_{uuid.uuid4()}.png"
    output_path = os.path.join(output_folder, unique_filename)
    image.save(output_path)
    print(f"Saved: {output_path}")

    from collections import Counter

    # ----------- Step 8: Compute Mode of Paragraph Start X -----------

    # Extract starting x-coordinates of each paragraph box (left edge)
    start_xs = [min(pt[0] for pt in box) for box in paragraph_boxes]

    # Count frequencies and find mode
    x_counts = Counter(start_xs)
    most_common_x, _ = x_counts.most_common(1)[0]

    # Draw vertical line at mode x
    draw.line([(most_common_x, 0), (most_common_x, image.height)], fill='red', width=2)

    # Save image with UUID
    unique_filename = f"line_{uuid.uuid4()}.png"
    output_path = os.path.join(output_folder, unique_filename)
    image.save(output_path)
    print(f"Saved: {output_path}")

    # ----------- Step 9: Highlight Misaligned Paragraphs Based on X Alignment -----------

    from collections import Counter

    # Define threshold for misalignment (tweakable)
    deviation_threshold = 25  # pixels

    # Draw paragraph boxes in red or green
    draw = ImageDraw.Draw(image)
    for box in paragraph_boxes:
        para_start_x = min(pt[0] for pt in box)
        deviation = abs(para_start_x - most_common_x)
        
        # Choose color based on alignment
        color = 'green' if deviation <= deviation_threshold else 'red'
        draw.polygon([tuple(pt) for pt in box], outline=color, width=2)

    # Draw the vertical alignment reference line
    draw.line([(most_common_x, 0), (most_common_x, image.height)], fill='blue', width=2)

    # Save image with UUID
    unique_filename = f"align_{uuid.uuid4()}.png"
    output_path = os.path.join(output_folder, unique_filename)
    image.save(output_path)
    print(f"Saved: {output_path}")