## Our Pipeline

The project runs in Google Colab on a GPU. It goes from a labelled dataset to a counted output video in seven steps.

### Step 1: Set up the environment

We check that a GPU is available with `nvidia-smi`, then install `ultralytics==8.0.20` to get YOLOv8.

### Step 2: Get the dataset

Our dataset is hosted on Roboflow (workspace `jawad`, project `object-detect-korbo`, version 1). We download it into `{HOME}/datasets` in YOLOv8 format. The download includes the images, their label files and a `data.yaml` that lists the two classes:

| Class ID | Class |
|---|---|
| 0 | car |
| 1 | truck |

```python
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("jawad").project("object-detect-korbo")
dataset = project.version(1).download("yolov8")
```

### Step 3: Train YOLOv8

We fine-tune the pretrained YOLOv8s model on our dataset for 50 epochs at an image size of 800. Training plots and weights are saved in `runs/detect/train/`.

```python
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=50 imgsz=800 plots=True
```

We then check the confusion matrix (`confusion_matrix.png`), the training curves (`results.png`) and sample validation predictions (`val_batch0_pred.jpg`). Finally, we zip the trained weights so we can keep them.

### Step 4: Load the traffic video and the trained model

We download the test video (`vehicle-counting.mp4`) and load our best weights (`best.pt`) with YOLO. We keep only the two classes we trained on (`CLASS_ID = [0, 1]`). As a quick check, we run the model on the first frame and draw the detected boxes.

### Step 5: Set up tracking

Detection alone can't tell whether a car in one frame is the same car in the next, so we add **ByteTrack**. We clone the ByteTrack repository and set its parameters:

| Parameter | Value |
|---|---|
| `track_thresh` | 0.25 |
| `track_buffer` | 30 |
| `match_thresh` | 0.8 |
| `aspect_ratio_thresh` | 3.0 |
| `min_box_area` | 1.0 |

Three helper functions connect YOLO and ByteTrack:

- `detections2boxes`: converts YOLO detections into the format ByteTrack expects.
- `tracks2boxes`: converts ByteTrack tracks back into boxes.
- `match_detections_with_tracks`: matches each track to a detection by IoU, so every detection gets a tracker ID.

### Step 6: Count vehicles

We draw a horizontal counting line across the frame at y = 1500 (from x = 50 to x = 3790). The **supervision** `LineCounter` counts each tracked vehicle once as it crosses this line.

For every frame of the video:

1. YOLOv8 detects the vehicles in the frame.
2. Detections that aren't car or truck are dropped.
3. ByteTrack updates the tracks, and each detection gets its tracker ID.
4. Detections without a tracker ID are dropped.
5. The line counter updates with the tracked vehicles.
6. Boxes, labels and the line with its counts are drawn on the frame.
7. The frame is written to `vehicle-counting-result.mp4`.

### Step 7: Validate the model

We validate `best.pt` on the validation set with `yolo mode=val`. The precision, recall and mAP scores are in the Results section.
