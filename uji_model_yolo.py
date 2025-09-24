# # -*- coding: utf-8 -*-
# # @Author: Your name
# # @Date:   2025-07-28 00:01:58
# # @Last Modified by:   Your name
# # @Last Modified time: 2025-07-28 00:28:05
# from ultralytics import YOLO
# import torch

# # Load model dengan approach yang lebih aman
# model = YOLO("C:/Users/ASUS/Documents/Pendeteksian-cabai/models/best.pt")

# # Prediksi gambar
# results = model.predict("C:/Users/ASUS/Documents/tambahan-dataset/photo_1408.jpg", save=True)

# print("Prediksi selesai. Hasil disimpan di folder runs/detect/predict")