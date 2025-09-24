# -*- coding: utf-8 -*-
# @Author: Your name
# @Date: 2025-07-27 15:22:17
# @Last Modified by:   Your name
# @Last Modified time: 2025-09-24 14:48:04

from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import threading
from collections import deque
import time
from PIL import Image, ImageDraw, ImageFont
import os
import json

app = Flask(__name__)

MODEL_MODE = "rgb"
MODEL_DIR = "models"
YOLO_MODEL = os.path.join(MODEL_DIR, "best.pt")


try:
    yolo = YOLO(YOLO_MODEL)
    print(f"Model YOLO OBB berhasil dimuat: {YOLO_MODEL}")
except Exception as e:
    print(f"Gagal memuat model YOLO: {e}")
    exit(1)

interpreter = None
input_details = None
output_details = None
input_shape = None
CLASS_NAMES = ['Tidak Layak Petik', 'Layak Petik']
font = None
font_small = None
model_lock = threading.Lock()

def load_tflite_model(mode):
    global interpreter, input_details, output_details, input_shape
    tflite_model_path = os.path.join(MODEL_DIR, f"chili_maturity_mobilenetv2_{mode}_float16.tflite")
    
    if not os.path.exists(tflite_model_path):
        raise FileNotFoundError(f"Model TFLite tidak ditemukan di: {tflite_model_path}")
        
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]
    print(f"Model TFLite ({mode.upper()}) siap. Input shape: {input_shape}")


try:
    load_tflite_model(MODEL_MODE)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        print("Font 'arial.ttf' tidak ditemukan. Menggunakan font default.")
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
except Exception as e:
    print(f"Gagal memuat model: {e}")
    exit(1)

STREAM_URL = "http://192.168.218.239:8080/videofeed"


TARGET_FPS = 20
PROCESS_WIDTH = 1280
BUFFER_SIZE = 10
MIN_CONFIDENCE = 0.6  
HISTORY_SIZE = 5

COLORS = {
    'Tidak Layak Petik': (0, 165, 255),  
    'Layak Petik': (0, 255, 0),         
    'Error': (0, 0, 255)                
}

frame_queue = deque(maxlen=BUFFER_SIZE)
output_frame = None
frame_lock = threading.Lock()
is_processing = True
classification_history = {}


def enhance_image_quality(image):
 
    if image is None:
        return None
        
  
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def calculate_image_quality(image):
    if image is None or image.size == 0:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_score = min(fm / 100, 1.0)
    return quality_score

def adaptive_confidence_threshold(quality_score):
    base_threshold = MIN_CONFIDENCE
    adaptive_threshold = base_threshold * (0.7 + 0.3 * quality_score)
    return max(0.3, min(adaptive_threshold, 0.8))

def get_color_features(image):
    if image is None or image.size == 0:
        return (0, 0, 0), (0, 0, 0)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_bgr = np.mean(image, axis=(0, 1)).astype(int)
    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
    mean_hsv = np.mean(hsv_image, axis=(0, 1)).astype(int)
    
    return mean_rgb, mean_hsv

def smooth_classification_results(obj_id, new_class, new_confidence):
    global classification_history
    if obj_id not in classification_history:
        classification_history[obj_id] = {
            'classes': [],
            'confidences': [],
            'last_seen': time.time()
        }
    
    history = classification_history[obj_id]
    history['classes'].append(new_class)
    history['confidences'].append(new_confidence)
    history['last_seen'] = time.time()
    
    if len(history['classes']) > HISTORY_SIZE:
        history['classes'] = history['classes'][-HISTORY_SIZE:]
        history['confidences'] = history['confidences'][-HISTORY_SIZE:]
    
    if len(history['classes']) < 3:
        return new_class, new_confidence
    
    # Voting berbasis confidence
    class_votes = {}
    for cls, conf in zip(history['classes'], history['confidences']):
        if cls not in class_votes:
            class_votes[cls] = 0
        class_votes[cls] += conf
    
    smoothed_class = max(class_votes, key=class_votes.get)
    smoothed_confidence = class_votes[smoothed_class] / sum(class_votes.values())
    
    return smoothed_class, smoothed_confidence

def cleanup_old_history(max_age_seconds=5):
    global classification_history
    current_time = time.time()
    keys_to_delete = []
    
    for obj_id, data in classification_history.items():
        if current_time - data['last_seen'] > max_age_seconds:
            keys_to_delete.append(obj_id)
    
    for key in keys_to_delete:
        del classification_history[key]

def crop_rotated_box(frame, points):
   
    try:
        if len(points) != 4:
            return None
        

        points = points.reshape(4, 2).astype(np.float32)
        
    
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        if width <= 5 or height <= 5:  
            return None
        

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (width, height))
        
        if warped.size == 0:
            return None
            
        return warped
    except Exception as e:
        print(f"Error cropping rotated box: {e}")
        return None

def is_green_chili_dominant(hsv_image):
 
    green_lower1 = np.array([35, 80, 40])    
    green_upper1 = np.array([85, 255, 200])  
    
    green_lower2 = np.array([30, 80, 40])    
    green_upper2 = np.array([95, 255, 180])
    
    
    leaf_lower = np.array([30, 30, 100])     
    leaf_upper = np.array([90, 100, 255])
    
    
    green_mask1 = cv2.inRange(hsv_image, green_lower1, green_upper1)
    green_mask2 = cv2.inRange(hsv_image, green_lower2, green_upper2)
    green_mask = cv2.bitwise_or(green_mask1, green_mask2)
    
  
    leaf_mask = cv2.inRange(hsv_image, leaf_lower, leaf_upper)
    

    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
    green_pixels = cv2.countNonZero(green_mask)
    leaf_pixels = cv2.countNonZero(leaf_mask)
    
    green_ratio = green_pixels / total_pixels
    leaf_ratio = leaf_pixels / total_pixels
    

    if green_pixels > 0:
        mean_h = np.mean(hsv_image[:,:,0][green_mask > 0])
        mean_s = np.mean(hsv_image[:,:,1][green_mask > 0])
        mean_v = np.mean(hsv_image[:,:,2][green_mask > 0])
        
     
        is_green_hue = (35 <= mean_h <= 85)
        is_high_saturation = mean_s > 60  
        is_medium_value = 40 < mean_v < 180  
        
    
        return (green_ratio > 0.3 and leaf_ratio < 0.2 and 
                is_green_hue and is_high_saturation and is_medium_value)
    
    return False

def is_red_chili_dominant(hsv_image):


    red_lower1 = np.array([0, 80, 40])      
    red_upper1 = np.array([10, 255, 255])
    
    red_lower2 = np.array([170, 80, 40])    
    red_upper2 = np.array([180, 255, 255])
    

    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
    red_pixels = cv2.countNonZero(red_mask)
    red_ratio = red_pixels / total_pixels
    
 
    if red_pixels > 0:
        mean_h = np.mean(hsv_image[:,:,0][red_mask > 0])
        if mean_h > 90: 
            mean_h = 180 - mean_h
        
        mean_s = np.mean(hsv_image[:,:,1][red_mask > 0])
        mean_v = np.mean(hsv_image[:,:,2][red_mask > 0])
        
    
        is_red_hue = (mean_h <= 10) or (mean_h >= 170)
        is_high_saturation = mean_s > 60  
        is_medium_value = 40 < mean_v < 220 
        
        return red_ratio > 0.35 and is_red_hue and is_high_saturation and is_medium_value
    
    return False

def classify_chili(cropped_img):
    
    global interpreter, input_details, output_details, input_shape, MODEL_MODE
    
    if interpreter is None or cropped_img is None:
        return "Error", 0.0
    
    try:
       
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        
      
        is_green = is_green_chili_dominant(hsv)
        is_red = is_red_chili_dominant(hsv)
        
     
        img_resized = cv2.resize(cropped_img, (input_shape[1], input_shape[0]))
        
        if MODEL_MODE == 'hsv':
            img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        else:
            img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
        img_normalized = (img_processed / 127.5) - 1.0
        img_expanded = np.expand_dims(img_normalized, axis=0).astype('float32')
        
        interpreter.set_tensor(input_details[0]['index'], img_expanded)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        probs = tf.nn.softmax(output[0]).numpy()
        model_class_id = np.argmax(probs)
        model_confidence = float(probs[model_class_id])
        model_class = CLASS_NAMES[model_class_id]
        
    
        if is_green and model_class == "Layak Petik":
        
            if model_confidence < 0.7:  
                return "Tidak Layak Petik", max(0.8, model_confidence)
        
        if is_red and model_class == "Tidak Layak Petik":
         
            if model_confidence < 0.7:
                return "Layak Petik", max(0.8, model_confidence)
        

        return model_class, model_confidence
        
    except Exception as e:
        print(f"Classification error: {e}")
    
        try:
            hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            if is_green_chili_dominant(hsv):
                return "Tidak Layak Petik", 0.8
            elif is_red_chili_dominant(hsv):
                return "Layak Petik", 0.8
            else:
                return "Error", 0.5
        except:
            return "Error", 0.5

def process_frame(frame):
 
    if yolo is None or interpreter is None or frame is None:
        return frame
        
    with model_lock:
        try:
        
            h, w = frame.shape[:2]
            scale = PROCESS_WIDTH / w
            new_h = int(h * scale)
            frame_resized = cv2.resize(frame, (PROCESS_WIDTH, new_h))
            
       
            frame_enhanced = enhance_image_quality(frame_resized)
            if frame_enhanced is None:
                return frame_resized
                
            quality_score = calculate_image_quality(frame_enhanced)
            adaptive_conf = adaptive_confidence_threshold(quality_score)
            
           
            results = yolo(frame_enhanced, conf=0.6, verbose=False, imgsz=640)
            cleanup_old_history()
            

            frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(frame_pil)
            
    
            for r in results:
                if hasattr(r, 'obb') and r.obb is not None:
                    for box in r.obb:
                       
                        class_id = int(box.cls)
                        class_name = yolo.names[class_id] if hasattr(yolo, 'names') else "Unknown"
                        
                        if class_name == 'Cabai':
                            points = box.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(int)
                            score = float(box.conf)
                            
                        
                            cropped_bgr = crop_rotated_box(frame_enhanced, points)
                            
                            if cropped_bgr is not None and cropped_bgr.size > 100:  
                                
                                maturity, conf = classify_chili(cropped_bgr)
                                
                             
                                center_x = np.mean(points[:, 0])
                                center_y = np.mean(points[:, 1])
                                obj_id = f"{int(center_x)}_{int(center_y)}"
                                
                                smoothed_maturity, smoothed_conf = smooth_classification_results(
                                    obj_id, maturity, conf
                                )
                                
                               
                                mean_rgb, mean_hsv = get_color_features(cropped_bgr)
                                
                             
                                color = COLORS.get(smoothed_maturity, (255, 255, 255))
                                
                          
                                draw.polygon([tuple(p) for p in points], outline=color, width=3)
                                
                          
                                text_label = f"{smoothed_maturity} ({smoothed_conf:.0%})"
                                text_rgb = f"RGB: {mean_rgb}"
                                text_hsv = f"HSV: {mean_hsv}"
                                
                             
                                text_y_start = max(points[:, 1].min() - 60, 10)
                                
                           
                                text_bbox_label = draw.textbbox((points[0][0], text_y_start), text_label, font=font)
                                text_bbox_rgb = draw.textbbox((points[0][0], text_y_start + 25), text_rgb, font=font_small)
                                text_bbox_hsv = draw.textbbox((points[0][0], text_y_start + 45), text_hsv, font=font_small)
                                
                                bg_x0 = min(text_bbox_label[0], text_bbox_rgb[0], text_bbox_hsv[0]) - 5
                                bg_y0 = text_bbox_label[1] - 5
                                bg_x1 = max(text_bbox_label[2], text_bbox_rgb[2], text_bbox_hsv[2]) + 5
                                bg_y1 = text_bbox_hsv[3] + 5
                                
                               
                                draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=color)
                                draw.text((points[0][0], text_y_start), text_label, font=font, fill=(0, 0, 0))
                                draw.text((points[0][0], text_y_start + 25), text_rgb, font=font_small, fill=(0, 0, 0))
                                draw.text((points[0][0], text_y_start + 45), text_hsv, font=font_small, fill=(0, 0, 0))
            
            return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return frame


def capture_thread():
   
    cap = None
    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(STREAM_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                print(f"Mencoba membuka stream: {STREAM_URL}")
                
            if not is_processing:
                time.sleep(0.5)
                continue
                
            ret, frame = cap.read()
            if ret:
                frame_queue.append(frame)
            else:
                print("Gagal membaca frame, mencoba menghubungkan kembali...")
                if cap:
                    cap.release()
                cap = None
                time.sleep(2)
                
            time.sleep(1 / TARGET_FPS)
            
        except Exception as e:
            print(f"Capture error: {e}")
            if cap:
                cap.release()
            cap = None
            time.sleep(5)

def processing_thread():
   
    global output_frame
    while True:
        if frame_queue:
            frame = frame_queue.popleft()
            processed = process_frame(frame)
            with frame_lock:
                output_frame = processed
        else:
            time.sleep(0.01)


@app.route('/')
def index():
    return render_template('index.html', mode=MODEL_MODE.upper())

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if output_frame is None:
                 
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Menunggu Stream...", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', blank_frame)
                else:
                    _, buffer = cv2.imencode('.jpg', output_frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1 / TARGET_FPS)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle', methods=['POST'])
def toggle_processing():
    global is_processing
    is_processing = not is_processing
    status = 'active' if is_processing else 'paused'
    print(f"Pemrosesan diubah ke: {status}")
    return jsonify({'status': status})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global MODEL_MODE, classification_history
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'})
        
    new_mode = data.get('mode', '').lower()
    if new_mode in ['rgb', 'hsv'] and new_mode != MODEL_MODE:
        MODEL_MODE = new_mode
        try:
            load_tflite_model(MODEL_MODE)
            classification_history.clear()
            print("Riwayat klasifikasi dihapus.")
            return jsonify({
                'status': 'success', 
                'message': f'Mode berhasil diubah ke {new_mode.upper()}'
            })
        except FileNotFoundError as e:
            return jsonify({'status': 'error', 'message': str(e)})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})
    
    return jsonify({
        'status': 'error', 
        'message': 'Mode tidak valid atau sama dengan mode saat ini.'
    })

@app.route('/stats')
def get_stats():
    mature_count = 0
    immature_count = 0
    
    for data in classification_history.values():
        if data['classes']:
            last_class = data['classes'][-1]
            if last_class == 'Layak Petik':
                mature_count += 1
            elif last_class == 'Tidak Layak Petik':
                immature_count += 1
    
    stats = {
        'total_detections': len(classification_history),
        'mature_count': mature_count,
        'immature_count': immature_count,
        'model_mode': MODEL_MODE.upper(),
        'processing_status': 'active' if is_processing else 'paused'
    }
    
    return jsonify(stats)

@app.route('/calibration', methods=['POST'])
def manual_calibration():
    """Endpoint untuk kalibrasi manual"""
    data = request.get_json()
    if data and 'action' in data:
        if data['action'] == 'reset_history':
            classification_history.clear()
            return jsonify({'status': 'success', 'message': 'Riwayat dikosongkan'})
    
    return jsonify({'status': 'error', 'message': 'Aksi tidak valid'})


if __name__ == '__main__':
    try:
 
        threading.Thread(target=capture_thread, daemon=True).start()
        threading.Thread(target=processing_thread, daemon=True).start()
        
        print(f"Server berjalan di http://localhost:5000")
        print(f"Mode: {MODEL_MODE.upper()}")
        print(f"Stream URL: {STREAM_URL}")
        print("Tekan Ctrl+C untuk menghentikan")
        
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False, debug=False)
        
    except KeyboardInterrupt:
        print("\nMenutup aplikasi...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()