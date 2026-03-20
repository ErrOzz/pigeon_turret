from ultralytics import YOLO

# Load the base ImageNet classification model (downloads automatically)
model = YOLO('yolov8n-cls.pt')

# Run inference on your image
results = model('test_pigeon.jpg')

# Get the dictionary of all 1000 known classes
names = results[0].names

# Get the top 3 predictions
top5_probs = results[0].probs.top5
top5_confs = results[0].probs.top5conf.tolist()

print("\n--- TOP PREDICTIONS ---")
for i in range(5):
    class_id = top5_probs[i]
    class_name = names[class_id]
    confidence = top5_confs[i] * 100
    print(f"{i+1}. {class_name}: {confidence:.1f}%")