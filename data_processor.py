from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET
import cv2

PROJECT_ROOT = Path(__file__).parent

VIDEO_PATH = PROJECT_ROOT / "data/AICity_data/train/S03/c010/vdo.avi"
OUTPUT_DIR = PROJECT_ROOT / "parsed_data"
ANNOTATION_PATH = PROJECT_ROOT / "data/ai_challenge_s03_c010-full_annotation.xml"


def extract_frames(video_path: Path, output_dir: Path, *, clean_output: bool = True) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for existing in output_dir.glob("frame_*.jpg"):
            existing.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        # Some codecs report ok=True on the final read but return an empty frame.
        if (not ok) or frame is None or getattr(frame, "size", 0) == 0:
            break

        frame_idx += 1
        frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            raise OSError(f"Failed to write frame to: {frame_path}")

    cap.release()
    return frame_idx + 1


@dataclass(slots=True)
class BoundingBox:
    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    label: str
    track_id: int
    occluded: int = 0
    outside: int = 0
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.xtl, self.ytl, self.xbr, self.ybr)

    def to_element_dict(self) -> dict[str, Any]:
        # Close to the XML element attributes + extras.
        return {
            "frame": self.frame,
            "xtl": self.xtl,
            "ytl": self.ytl,
            "xbr": self.xbr,
            "ybr": self.ybr,
            "label": self.label,
            "track_id": self.track_id,
            "occluded": self.occluded,
            "outside": self.outside,
            **self.attributes,
        }


class AICityFrames:

    def __init__(
        self,
        frames_dir: Path = OUTPUT_DIR,
        annotation_path: Path = ANNOTATION_PATH,
        *,
        image_index_base: int = 1,
        scale: float = 1.0,
    ) -> None:
        self.frames_dir = frames_dir
        self.annotation_path = annotation_path
        self.image_index_base = image_index_base
        self.scale = scale

        self._by_frame: dict[int, list[BoundingBox]] = {}
        self._index_annotations()

    @property
    def frame_count(self) -> int:
        return len(list(self.frames_dir.glob("frame_*.jpg")))

    def _index_annotations(self) -> None:
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation XML not found: {self.annotation_path}")

        context = ET.iterparse(str(self.annotation_path), events=("start", "end"))

        current_track_id: int | None = None
        current_label: str | None = None

        for event, elem in context:
            if event == "start" and elem.tag == "track":
                current_track_id = int(elem.attrib["id"])
                current_label = elem.attrib.get("label", "")
                continue

            if event == "end" and elem.tag == "box":
                if current_track_id is None or current_label is None:
                    elem.clear()
                    continue

                attrs: dict[str, str] = {}
                for a in elem.findall("attribute"):
                    name = a.attrib.get("name")
                    if name:
                        attrs[name] = (a.text or "").strip()

                frame = int(elem.attrib["frame"])
                bb = BoundingBox(
                    frame=frame,
                    xtl=float(elem.attrib["xtl"]),
                    ytl=float(elem.attrib["ytl"]),
                    xbr=float(elem.attrib["xbr"]),
                    ybr=float(elem.attrib["ybr"]),
                    label=current_label,
                    track_id=current_track_id,
                    occluded=int(elem.attrib.get("occluded", "0")),
                    outside=int(elem.attrib.get("outside", "0")),
                    attributes=attrs,
                )

                self._by_frame.setdefault(frame, []).append(bb)
                elem.clear()
                continue

            if event == "end" and elem.tag == "track":
                current_track_id = None
                current_label = None
                elem.clear()

    def frame_path(self, frame: int) -> Path:
        return self.frames_dir / f"frame_{frame + self.image_index_base:06d}.jpg"

    def boxes(self, frame: int, *, include_outside: bool = False) -> list[BoundingBox]:
        boxes = self._by_frame.get(frame, [])
        if include_outside:
            return list(boxes)
        return [b for b in boxes if b.outside == 0]

    def box_elements(self, frame: int, *, include_outside: bool = False) -> list[dict[str, Any]]:
        return [b.to_element_dict() for b in self.boxes(frame, include_outside=include_outside)]

    def image(self, frame: int):

        img_path = self.frames_dir / f"frame_{frame + self.image_index_base:06d}.jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Frame image not found or unreadable: {img_path}")
        if self.scale != 1.0:
            h, w = img.shape
            new_h, new_w = int(h * self.scale), int(w * self.scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def get(self, frame: int, *, load_image: bool = True, include_outside: bool = False):
        img = self.image(frame) if load_image else None
        return img, self.boxes(frame, include_outside=include_outside)

    def frames_with_boxes(self) -> Iterable[int]:
        return sorted(self._by_frame.keys())


def main() -> None:
    count = extract_frames(VIDEO_PATH, OUTPUT_DIR)
    print(f"Wrote {count} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
