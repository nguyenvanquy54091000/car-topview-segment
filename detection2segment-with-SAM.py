import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import SAM
import supervision as sv

if os.path.exists('sam3_b.pt'):
    print("🗑️ Đang xóa file model cũ để tải lại bản mới...")
    os.remove('sam2_b.pt')

IMG_DIR = ''
LBL_DIR = ''
OUT_IMG_DIR = ''
OUT_LBL_DIR = ''
OUT_MASK_DIR = ''

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# Khởi tạo SAM Model
model = SAM("sam2_b.pt") 

def generate_masks(img_dir, label_dir, mask_dir, img_size=(416, 416)):
    """
    Công dụng: Chuyển đổi tọa độ Polygon (YOLOv11) sang ảnh nhị phân (Mask).
    Nó giải mã tọa độ chuẩn hóa và vẽ vùng chứa đối tượng bằng cv2.fillPoly.
    """
    os.makedirs(mask_dir, exist_ok=True)
    
    img_files = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))
    print(f"Tìm thấy {len(img_files)} ảnh. Đang tiến hành tạo masks...")

    for img_path in tqdm(img_files, desc="Processing"):
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w, _ = img.shape
        
        mask = np.zeros((h, w), dtype=np.uint8)

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = list(map(float, line.strip().split()))
                    if len(data) < 3: continue 
                    

                    poly = np.array(data[1:]).reshape(-1, 2)
                    poly[:, 0] *= w
                    poly[:, 1] *= h
                    
                    # Vẽ vùng xe màu trắng (255) lên nền đen
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

        mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(mask_dir, f"{img_path.stem}.png"), mask_resized)

def yolo_to_bbox(line, img_w, img_h):
    parts = list(map(float, line.strip().split()))
    if len(parts) < 5:
        return None, None
    cls = int(parts[0])
    x_c, y_c, w, h = parts[1:5]
    
    xmin = int((x_c - w / 2) * img_w)
    ymin = int((y_c - h / 2) * img_h)
    xmax = int((x_c + w / 2) * img_w)
    ymax = int((y_c + h / 2) * img_h)
    
    return cls, [xmin, ymin, xmax, ymax]

mask_annotator = sv.MaskAnnotator()

image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(IMG_DIR, img_name)
    base_name = os.path.splitext(img_name)[0]
    
    label_in_path = os.path.join(LBL_DIR, f"{base_name}.txt")
    label_out_path = os.path.join(OUT_LBL_DIR, f"{base_name}.txt")
    img_out_path = os.path.join(OUT_IMG_DIR, f"{base_name}_segmented.jpg")
    mask_out_path = os.path.join(OUT_MASK_DIR, f"{base_name}.png")

    if not os.path.exists(label_in_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue
    h_img, w_img, _ = image.shape

    # Đọc Bounding Boxes
    classes, input_boxes = [], []
    with open(label_in_path, 'r') as f:
        for line in f:
            cls, bbox = yolo_to_bbox(line, w_img, h_img)
            if bbox:
                classes.append(cls)
                input_boxes.append(bbox)

    if not input_boxes:
         continue

    # SAM 2 Dự đoán Segmentation
    results = model.predict(source=image, bboxes=input_boxes, verbose=False)
    result = results[0]

    if result.masks is not None:
        detections = sv.Detections.from_ultralytics(result)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        cv2.imwrite(img_out_path, annotated_image)
    
        mask_img = np.zeros((h_img, w_img), dtype=np.uint8)
        
        new_labels = []
        for i, polygon_coords in enumerate(result.masks.xyn):
            cls = classes[i] if i < len(classes) else classes[0]
            if len(polygon_coords) < 3: 
                continue
            
            poly = np.array(polygon_coords).reshape(-1, 2)
            poly_scaled = poly.copy()
            poly_scaled[:, 0] *= w_img
            poly_scaled[:, 1] *= h_img
            cv2.fillPoly(mask_img, [poly_scaled.astype(np.int32)], 255)
            
            poly_str = " ".join([f"{coord:.6f}" for pair in polygon_coords for coord in pair])
            new_labels.append(f"{cls} {poly_str}")

        with open(label_out_path, 'w') as f:
            f.write("\n".join(new_labels))
            
        mask_resized = cv2.resize(mask_img, (416, 416), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_out_path, mask_resized)
            
        print(f"✅ Đã lưu: {img_out_path}")
        print(f"✅ Đã lưu label: {label_out_path}")
        print(f"✅ Đã lưu mask: {mask_out_path}")

print("\n🎉 Hoàn tất quá trình Segmentation & sinh Label mới!")

