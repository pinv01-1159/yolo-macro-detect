import numpy as np
import supervision as sv

from models.inference import BORDER_PX, pad_border, unpad_detections


def test_border_roundtrip():
    frame = np.zeros((640, 640, 3), np.uint8)
    padded = pad_border(frame)
    assert padded.shape == (640 + 2 * BORDER_PX, 640 + 2 * BORDER_PX, 3)
    assert (padded[0, 0] == 114).all()
    # una caja que en la imagen con margen toca el relleno vuelve recortada al borde original
    d = sv.Detections(
        xyxy=np.array([[10.0, 26.0, 100.0, 680.0]]),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
    )
    out = unpad_detections(d, frame.shape)
    assert out.xyxy.tolist() == [[0.0, 10.0, 84.0, 640.0]]
