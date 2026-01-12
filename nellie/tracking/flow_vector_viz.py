"""
Helpers to visualize Hu tracking flow vectors and mocap markers in napari.
"""
from __future__ import annotations

import os
import numpy as np

from nellie.im_info.verifier import ImInfo


def load_flow_vector_array(im_info: ImInfo, path: str | None = None) -> np.ndarray:
    flow_path = path or im_info.pipeline_paths["flow_vector_array"]
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow vector array not found: {flow_path}")
    return np.load(flow_path)


def flow_vectors_to_tracks(
    flow_vector_array: np.ndarray,
    *,
    no_z: bool,
    cost_threshold: float | None = None,
    stride: int = 1,
    max_vectors: int | None = None,
) -> tuple[np.ndarray, dict]:
    if flow_vector_array.size == 0:
        track_cols = 4 if no_z else 5
        return np.empty((0, track_cols), dtype=np.float32), {"cost": np.array([], dtype=np.float32)}

    flow = flow_vector_array
    if cost_threshold is not None:
        flow = flow[flow[:, -1] <= cost_threshold]
    if stride > 1:
        flow = flow[::stride]
    if max_vectors is not None and flow.shape[0] > max_vectors:
        flow = flow[:max_vectors]

    if flow.size == 0:
        track_cols = 4 if no_z else 5
        return np.empty((0, track_cols), dtype=np.float32), {"cost": np.array([], dtype=np.float32)}

    track_ids = np.arange(flow.shape[0], dtype=np.int64)
    t0 = flow[:, 0].astype(np.int64)
    cost = flow[:, -1].astype(np.float32)

    if no_z:
        y0 = flow[:, 1].astype(np.float32)
        x0 = flow[:, 2].astype(np.float32)
        dy = flow[:, 3].astype(np.float32)
        dx = flow[:, 4].astype(np.float32)
        coords0 = np.column_stack((y0, x0))
        coords1 = np.column_stack((y0 + dy, x0 + dx))
    else:
        z0 = flow[:, 1].astype(np.float32)
        y0 = flow[:, 2].astype(np.float32)
        x0 = flow[:, 3].astype(np.float32)
        dz = flow[:, 4].astype(np.float32)
        dy = flow[:, 5].astype(np.float32)
        dx = flow[:, 6].astype(np.float32)
        coords0 = np.column_stack((z0, y0, x0))
        coords1 = np.column_stack((z0 + dz, y0 + dy, x0 + dx))

    tracks = np.vstack(
        [
            np.column_stack((track_ids, t0, coords0)),
            np.column_stack((track_ids, t0 + 1, coords1)),
        ]
    ).astype(np.float32)
    properties = {"cost": np.repeat(cost, 2)}
    return tracks, properties


def load_flow_vectors_as_tracks(
    im_info: ImInfo,
    *,
    path: str | None = None,
    cost_threshold: float | None = None,
    stride: int = 1,
    max_vectors: int | None = None,
) -> tuple[np.ndarray, dict]:
    flow = load_flow_vector_array(im_info, path=path)
    return flow_vectors_to_tracks(
        flow,
        no_z=im_info.no_z,
        cost_threshold=cost_threshold,
        stride=stride,
        max_vectors=max_vectors,
    )


def load_mocap_markers_as_points(
    im_info: ImInfo,
    *,
    t_range: tuple[int, int] | None = None,
    time_stride: int = 1,
    point_stride: int = 1,
    max_points: int | None = None,
) -> np.ndarray:
    marker_memmap = im_info.get_memmap(im_info.pipeline_paths["im_marker"])
    t_start, t_end = (0, marker_memmap.shape[0]) if t_range is None else t_range

    points = []
    for t in range(t_start, t_end, time_stride):
        coords = np.argwhere(marker_memmap[t] > 0)
        if coords.size == 0:
            continue
        if point_stride > 1:
            coords = coords[::point_stride]
        t_col = np.full((coords.shape[0], 1), t, dtype=np.int64)
        points.append(np.concatenate((t_col, coords.astype(np.int64)), axis=1))

    if points:
        points = np.vstack(points)
    else:
        point_cols = 3 if im_info.no_z else 4
        points = np.empty((0, point_cols), dtype=np.int64)

    if max_points is not None and points.shape[0] > max_points:
        points = points[:max_points]

    return points


if __name__ == "__main__":
    from nellie.im_info.verifier import FileInfo
    
    test_file = "/Users/austin/test_files/nellie_all_tests/yeast_3d_mitochondria.ome_variants/variant_TCZYX_dup2.ome.tif"
    file_info = FileInfo(test_file)
    file_info.find_metadata()
    file_info.load_metadata()

    im_info = ImInfo(file_info)

    tracks, props = load_flow_vectors_as_tracks(im_info, cost_threshold=None, stride=1)
    markers = load_mocap_markers_as_points(im_info, point_stride=1)

    import napari
    viewer = napari.Viewer()
    viewer.add_tracks(tracks, properties=props, name="flow_vectors")
    viewer.add_points(markers, name="mocap_markers", size=2)
    napari.run()
