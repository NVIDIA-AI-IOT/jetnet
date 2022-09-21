## Common Types

### Point

```python3
class Point:
    x: int
    y: int
```

### Polygon

```python3
class Polygon:
    points: Sequence[Point]
```

### Classification

```python3
class Classification:
    index: int
    label: Optional[str]
    score: Optional[float]
```

### Detection

```python3
class Detection:
    boundary: Polygon
    classification: Classification
```

### DetectionSet

```python3
class DetectionSet:
    detections: Sequence[Detection]
```

### Keypoint

```python3
class Keypoint:
    x: int
    y: int
    index: int
    label: Optional[str]
    score: Optional[float]
```

### Pose

```python3
class Pose:
    keypoints: Sequence[Keypoint]
```

### PoseSet

```python3
class PoseSet:
    poses: Sequence[Pose]
```

### TextDetection

```python3
class TextDetection:
    boundary: Polygon
    text: Optional[str]
    score: Optional[float]
```

### TextDetectionSet

```python3
class TextDetectionSet:
    detections: Sequence[Detection]
```

## Abstract Types

### ClassificationModel

```python3
class ClassificationModel:

    def init(self):
        pass

    def get_labels(self) -> Sequence[str]:
        raise NotImplementedError
      
    def __call__(self, index: Image) -> Classification:
        raise NotImplementedError
```

### DetectionModel

```python3
class DetectionModel:

    def init(self):
        pass

    def get_labels(self) -> Sequence[str]:
        raise NotImplementedError
      
    def __call__(self, index: Image) -> DetectionSet:
        raise NotImplementedError
```

### PoseModel

```python3
class PoseModel:

    def init(self):
        pass

    def get_keypoints(self) -> Sequence[str]:
        raise NotImplementedError

    def get_skeleton(self) -> Sequence[Tuple[int, int]]:
        raise NotImplementedError

    def __call__(self, index: Image) -> PoseSet:
        raise NotImplementedError
```

### TextDetectionModel

```python3
class TextDetectionModel:

    def init(self):
        pass

    def __call__(self, index: Image) -> TextDetectionSet:
        raise NotImplementedError
```

### ImageDataset

```python3
class ImageDataset:

    def init(self):
        pass
        
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Image:
        raise NotImplementedError
```