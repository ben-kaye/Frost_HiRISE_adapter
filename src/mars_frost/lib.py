from dataclasses import dataclass, fields
from shapely.wkt import loads
from shapely.geometry import Polygon, MultiPolygon
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
from pathlib import Path
import random
from omegaconf import OmegaConf, DictConfig
import imageio as iio
import json


@dataclass
class Bbox:
    min_h: float
    max_h: float
    min_w: float
    max_w: float


@dataclass
class Label:
    labeler: str
    geologic_context: str
    frost_context: list[str]
    frost_type: str
    confidence: str


@dataclass
class ImageMeta:
    coords: Bbox
    bbox: Bbox
    annotations: list[Polygon]
    annotations_meta: list[Label]
    image_size: tuple[int, int]

    @staticmethod
    def from_dict(metadata: DictConfig, image_size: tuple[int, int]) -> "ImageMeta":
        return ImageMeta(
            coords=ImageMeta._image_coords(metadata),
            bbox=ImageMeta._image_bbox(metadata),
            annotations=ImageMeta._annotations(
                metadata,
                intersect_bbox=True,
            ),
            annotations_meta=ImageMeta._annotations_meta(metadata),
            image_size=image_size,
        )

    @staticmethod
    def _image_coords(metadata: DictConfig):
        min_h, max_h, min_w, max_w = metadata.image_coords

        return Bbox(min_h, max_h, min_w, max_w)

    @staticmethod
    def _annotations(metadata: DictConfig, intersect_bbox: bool) -> list[Polygon]:
        intersections = (
            loads(ann.shapely) for ann in metadata.annotations if "shapely" in ann
        )
        if intersect_bbox:
            box: Polygon = loads(metadata.tile_bbox)
            intersections = (poly.intersection(box) for poly in intersections)
        return list(intersections)

    def _annotations_meta(metadata: DictConfig) -> list[Label]:
        keys_ = [f.name for f in fields(Label)]
        return [
            Label(**{k: v for k, v in ann.items() if k in keys_})
            for ann in metadata.annotations
        ]

    @staticmethod
    def _image_bbox(metadata) -> Bbox:
        polygon: Polygon = loads(metadata.tile_bbox)
        corners = list(polygon.exterior.coords)

        # get the max x, y and min x, y
        max_h = max(corners, key=lambda x: x[0])[0]
        min_h = min(corners, key=lambda x: x[0])[0]
        max_w = max(corners, key=lambda x: x[1])[1]
        min_w = min(corners, key=lambda x: x[1])[1]

        return Bbox(min_h, max_h, min_w, max_w)

    def _convert_coord(self, coord: tuple[float, float]) -> tuple[float, float]:
        h_new = (coord[0] - self.bbox.min_h) / (self.bbox.max_h - self.bbox.min_h)
        w_new = (coord[1] - self.bbox.min_w) / (self.bbox.max_w - self.bbox.min_w)
        return (h_new * self.image_size[0], w_new * self.image_size[1])

    def _convert_poly(self, poly: Polygon) -> Polygon:
        if isinstance(poly, MultiPolygon):
            return [self._convert_poly(_poly) for _poly in poly.geoms]
        return Polygon([self._convert_coord(coord) for coord in poly.exterior.coords])

    def plot(
        self,
        ax: plt.Axes,
        colors: list[str] | str | None,
        color_map: dict[str, str] | None = None,
    ):
        if not isinstance(colors, list):
            colors = [colors] * len(self.annotations)
        for polygon, color, label in zip(
            self.annotations, colors, self.annotations_meta
        ):
            if color_map is not None:
                color = color_map[label.frost_type]

            poly = self._convert_poly(polygon)
            if not isinstance(poly, list):
                poly = [poly]
            for poly_ in poly:
                plot_polygon(poly_, ax=ax, color=color)

    def get_coverage(self) -> list[float]:
        area = self.image_size[0] * self.image_size[1]
        areas_ratio = [
            self._convert_poly(polygon).area / area for polygon in self.annotations
        ]
        return areas_ratio


# Loading in and plotting some of the data
def load_text_ids(file_path):
    """Simple helper to load all lines from a text file"""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def obtain_class_balance(subdirs: list[Path]):
    """
    Get the class balance of data split
    :param: data_directory: Main data directory to query
    :param: sub_directories: List of subdirectories that contain data folders
    """
    frost_count = 0
    background_count = 0
    frost_idxs = []
    background_idxs = []

    for i, path_ in enumerate(subdirs):
        sub_path = path_ / "labels"
        label_dir = list(p for p in sub_path.iterdir() if p.is_dir())[-1]
        samples = list(label_dir.iterdir())

        label_type = label_dir.name

        match label_type:
            case "frost":
                frost_count += len(samples)
                frost_idxs.append(i)
            case "background":
                background_count += len(samples)
                background_idxs.append(i)

    return frost_count, background_count, frost_idxs, background_idxs


@dataclass
class ClassBalance:
    frost_count: int
    background_count: int
    frost_idxs: list[int]
    background_idxs: list[int]


@dataclass
class DataUnit:
    base_dir: Path
    dirs: list[Path]
    tile_ids: list[str]
    class_balance: ClassBalance

    @staticmethod
    def from_ids(data_dir: Path, id_path: Path) -> "DataUnit":
        ids_ = set(load_text_ids(id_path))

        valid_subdirs: list[Path] = [
            subdir
            for subdir in data_dir.iterdir()
            if subdir.is_dir() and "_".join(subdir.stem.split("_")[:3]) in ids_
        ]

        tile_ids = sorted(ids_)

        class_balance_ = ClassBalance(*obtain_class_balance(valid_subdirs))

        return DataUnit(
            base_dir=data_dir,
            dirs=valid_subdirs,
            class_balance=class_balance_,
            tile_ids=tile_ids,
        )


def get_metadata(path):
    with Path(path).open() as f:
        data = json.load(f)
    return data


def get_sample(data_unit: DataUnit, is_frost: bool):
    random_subdir_index = (
        random.choice(data_unit.class_balance.frost_idxs)
        if is_frost
        else random.choice(data_unit.class_balance.background_idxs)
    )
    random_subdir = data_unit.dirs[random_subdir_index]
    label_type = "frost" if is_frost else "background"
    meta_dir = random_subdir / f"labels/{label_type}"
    image_dir = random_subdir / f"tiles/{label_type}"
    label_paths = list(meta_dir.glob("*.json"))
    image_paths = list(image_dir.glob("*.png"))
    index_ = random.choice(range(len(label_paths)))

    metadata = OmegaConf.create(get_metadata(label_paths[index_]))
    image = iio.imread(image_paths[index_])

    return image, metadata
