import os.path
"""
Hierarchical feature extraction for microscopy images.

This module provides the Hierarchy class for extracting multi-level features (voxels, nodes,
branches, organelles, and images) from segmented and tracked microscopy data.
"""
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi_cpu
from scipy import spatial
from skimage.measure import regionprops

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.measure")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="skimage.measure")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="skimage.morphology")
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Mean of empty slice"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="All-NaN slice encountered"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in divide"
)

from nellie.utils import adaptive_run
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo
from nellie.tracking.flow_interpolation import FlowInterpolator


@dataclass(frozen=True)
class HierarchyConfig:
    """Algorithm configuration for ``Hierarchy``.

    Frozen — represents the user's intent at construction time. Hierarchy
    copies these values into mutable instance attributes that the
    OOM/availability cascade can update mid-run (``device``,
    ``low_memory``). Inspect ``Hierarchy.config`` to see the original
    intent regardless of any cascade-driven runtime fallbacks.

    Note: Hierarchy is asymmetric — it has no ``num_t`` field because
    the value is always derived from ``im_info.shape[0]`` inside
    ``__init__``. Per resolved decision #3 of PRD #112, no ``num_t``
    constructor parameter exists.
    """

    skip_nodes: bool = True
    low_memory: bool = False
    enable_motility: bool = True
    enable_adjacency: bool = True
    device: str = "auto"
    node_chunk_size: int | None = None
    max_node_mask_elems: int = int(5e7)

    def __post_init__(self) -> None:
        adaptive_run.normalize_device(self.device)
        if self.max_node_mask_elems <= 0:
            raise ValueError(
                f"HierarchyConfig.max_node_mask_elems must be > 0, "
                f"got {self.max_node_mask_elems}"
            )
        if self.node_chunk_size is not None and self.node_chunk_size <= 0:
            raise ValueError(
                f"HierarchyConfig.node_chunk_size must be > 0 if set, "
                f"got {self.node_chunk_size}"
            )


class Hierarchy:
    """
    Main orchestration class for hierarchical feature extraction.
    """

    def __init__(
        self,
        im_info: ImInfo,
        config: HierarchyConfig = HierarchyConfig(),
        viewer=None,
    ):
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata object.
        config : HierarchyConfig
            Algorithm configuration. Defaults to ``HierarchyConfig()``.
        viewer : optional
            Viewer object with `.status` attribute for status updates.

        Notes
        -----
        Hierarchy does NOT take a ``num_t`` constructor argument
        (asymmetric per resolved decision #3 of PRD #112). The value is
        always derived from ``self.im_info.shape[0]`` below.
        """
        self.im_info = im_info
        self.config = config
        self.num_t = self.im_info.shape[0]

        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_res["Y"], self.im_info.dim_res["X"])
        else:
            self.spacing = (
                self.im_info.dim_res["Z"],
                self.im_info.dim_res["Y"],
                self.im_info.dim_res["X"],
            )

        self.viewer = viewer

        # Aliases for hot path readability — config remains the source of truth.
        self.skip_nodes = config.skip_nodes
        self.enable_motility = config.enable_motility
        self.enable_adjacency = config.enable_adjacency
        self.node_chunk_size = config.node_chunk_size
        self.max_node_mask_elems = int(config.max_node_mask_elems)

        # Cascade-mutable runtime state. Initial values come from config;
        # ``_set_backend`` and ``_set_low_memory`` may update them on retry.
        self.low_memory = bool(config.low_memory)
        self.device = adaptive_run.normalize_device(config.device or "auto")
        self.xp, _ndi, self.device_type = adaptive_run.resolve_backend(self.device)

        # Memory-mapped images
        self.im_obj_reassigned = None
        self.im_branch_reassigned = None

        # Flow interpolators (lazy init in run)
        self.flow_interpolator_fw: FlowInterpolator | None = None
        self.flow_interpolator_bw: FlowInterpolator | None = None

        # Hierarchy levels
        self.voxels = None
        self.nodes = None
        self.branches = None
        self.components = None
        self.image = None

    def _set_backend(self, device: str):
        device = adaptive_run.normalize_device(device)
        self.device = device
        self.xp, _ndi, self.device_type = adaptive_run.resolve_backend(device)

    def _set_low_memory(self, low_memory):
        self.low_memory = bool(low_memory)

    def _resolve_node_chunk_size(self, num_nodes: int, num_voxels: int) -> int:
        if num_voxels <= 0:
            return 1
        base_chunk = self.node_chunk_size or 10000
        max_mask_elems = self.max_node_mask_elems
        if self.low_memory:
            max_mask_elems = max(1, max_mask_elems // 4)
        if num_nodes > 0 and num_nodes * base_chunk > max_mask_elems:
            base_chunk = max(1, max_mask_elems // num_nodes)
        return int(max(1, min(base_chunk, num_voxels)))

    def _allocate_memory(self):
        """
        Load required image data as memory-mapped arrays.
        """
        self.im_raw = self.im_info.get_memmap(self.im_info.im_path)
        self.im_struct = self.im_info.get_memmap(
            self.im_info.pipeline_paths["im_preprocessed"]
        )
        self.im_distance = self.im_info.get_memmap(
            self.im_info.pipeline_paths["im_distance"]
        )
        self.im_skel = self.im_info.get_memmap(
            self.im_info.pipeline_paths["im_skel"]
        )
        self.label_components = self.im_info.get_memmap(
            self.im_info.pipeline_paths["im_instance_label"]
        )
        self.label_branches = self.im_info.get_memmap(
            self.im_info.pipeline_paths["im_skel_relabelled"]
        )
        self.im_border_mask = self.im_info.get_memmap(
            self.im_info.pipeline_paths["im_border"]
        )
        self.im_pixel_class = self.im_info.get_memmap(
            self.im_info.pipeline_paths["im_pixel_class"]
        )

        if not self.im_info.no_t:
            obj_reassigned_path = self.im_info.pipeline_paths.get(
                "im_obj_label_reassigned"
            )
            branch_reassigned_path = self.im_info.pipeline_paths.get(
                "im_branch_label_reassigned"
            )
            if obj_reassigned_path and branch_reassigned_path:
                if os.path.exists(obj_reassigned_path) and os.path.exists(
                    branch_reassigned_path
                ):
                    self.im_obj_reassigned = self.im_info.get_memmap(
                        obj_reassigned_path
                    )
                    self.im_branch_reassigned = self.im_info.get_memmap(
                        branch_reassigned_path
                    )

    def _get_hierarchies(self):
        """
        Run voxel, node, branch, component and image analyses.
        """
        self.voxels = Voxels(self)
        logger.info("Running voxel analysis")
        start = time.time()
        self.voxels.run()
        v_time = time.time() - start

        self.nodes = Nodes(self)
        logger.info("Running node analysis")
        start = time.time()
        self.nodes.run()
        n_time = time.time() - start

        self.branches = Branches(self)
        logger.info("Running branch analysis")
        start = time.time()
        self.branches.run()
        b_time = time.time() - start

        self.components = Components(self)
        logger.info("Running component analysis")
        start = time.time()
        self.components.run()
        c_time = time.time() - start

        self.image = Image(self)
        logger.info("Running image analysis")
        start = time.time()
        self.image.run()
        i_time = time.time() - start

        logger.debug(f"Voxel analysis took {v_time:.3f} seconds")
        logger.debug(f"Node analysis took {n_time:.3f} seconds")
        logger.debug(f"Branch analysis took {b_time:.3f} seconds")
        logger.debug(f"Component analysis took {c_time:.3f} seconds")
        logger.debug(f"Image analysis took {i_time:.3f} seconds")

    # -------------------------------------------------------------------------
    # Streaming feature saving
    # -------------------------------------------------------------------------

    @staticmethod
    def _iter_feature_arrays(level, labels=None):
        """
        Generator that yields (t, frame_array, headers_without_t_label)
        for a given level, without materializing the full feature table.
        """
        all_attr = []

        node_attr = getattr(level, "aggregate_node_metrics", None)
        if node_attr:
            all_attr.append(node_attr)

        voxel_attr = getattr(level, "aggregate_voxel_metrics", None)
        if voxel_attr:
            all_attr.append(voxel_attr)

        branch_attr = getattr(level, "aggregate_branch_metrics", None)
        if branch_attr:
            all_attr.append(branch_attr)

        component_attr = getattr(level, "aggregate_component_metrics", None)
        if component_attr:
            all_attr.append(component_attr)

        inherent_features = getattr(level, "features_to_save", [])
        for feature in inherent_features:
            feature_vals = getattr(level, feature, None)
            if feature_vals is None:
                continue
            if len(feature_vals) == 0:
                continue
            # match aggregate_* structure: list over t of dicts
            all_attr.append([{feature: feature_vals[t]} for t in range(len(feature_vals))])

        if not all_attr:
            return

        num_frames = len(all_attr[0])

        for t in range(num_frames):
            time_dict = {}
            for attr in all_attr:
                time_dict.update(attr[t])

            time_array, new_headers = append_to_array(time_dict)

            if labels is None:
                labels_t = np.arange(len(time_array[0]), dtype=np.int64)
            else:
                labels_t = np.asarray(labels[t])

            t_col = np.full(labels_t.shape[0], t, dtype=np.int64)

            # prepend t and label columns
            time_array.insert(0, labels_t)
            time_array.insert(0, t_col)

            frame_array = np.array(time_array).T
            yield t, frame_array, new_headers

    def _save_dfs(self):
        """
        Save features to CSVs, streaming per-frame to limit peak memory usage.
        """
        if self.viewer is not None:
            self.viewer.status = "Saving features to csv files."

        # Voxels
        voxel_path = self.im_info.pipeline_paths["features_voxels"]
        first = True
        voxel_headers_full = None
        for _, frame_array, headers in self._iter_feature_arrays(self.voxels, labels=None):
            if first:
                voxel_headers_full = ["t", "label"] + headers
                first = False
                mode = "w"
                header_flag = True
            else:
                mode = "a"
                header_flag = False
            df = pd.DataFrame(frame_array, columns=voxel_headers_full)
            df.to_csv(voxel_path, index=False, mode=mode, header=header_flag)

        # Nodes
        if not self.skip_nodes:
            node_path = self.im_info.pipeline_paths["features_nodes"]
            first = True
            node_headers_full = None
            for _, frame_array, headers in self._iter_feature_arrays(
                self.nodes, labels=None
            ):
                if first:
                    node_headers_full = ["t", "label"] + headers
                    first = False
                    mode = "w"
                    header_flag = True
                else:
                    mode = "a"
                    header_flag = False
                df = pd.DataFrame(frame_array, columns=node_headers_full)
                df.to_csv(node_path, index=False, mode=mode, header=header_flag)

        # Branches
        branch_path = self.im_info.pipeline_paths["features_branches"]
        first = True
        branch_headers_full = None
        for _, frame_array, headers in self._iter_feature_arrays(
            self.branches, labels=self.branches.branch_label
        ):
            if first:
                branch_headers_full = ["t", "label"] + headers
                first = False
                mode = "w"
                header_flag = True
            else:
                mode = "a"
                header_flag = False
            df = pd.DataFrame(frame_array, columns=branch_headers_full)
            df.to_csv(branch_path, index=False, mode=mode, header=header_flag)

        # Components (organelles)
        component_path = self.im_info.pipeline_paths["features_organelles"]
        first = True
        component_headers_full = None
        for _, frame_array, headers in self._iter_feature_arrays(
            self.components, labels=self.components.component_label
        ):
            if first:
                component_headers_full = ["t", "label"] + headers
                first = False
                mode = "w"
                header_flag = True
            else:
                mode = "a"
                header_flag = False
            df = pd.DataFrame(frame_array, columns=component_headers_full)
            df.to_csv(component_path, index=False, mode=mode, header=header_flag)

        # Image-level features
        image_path = self.im_info.pipeline_paths["features_image"]
        first = True
        image_headers_full = None
        for _, frame_array, headers in self._iter_feature_arrays(self.image, labels=None):
            if first:
                image_headers_full = ["t", "label"] + headers
                first = False
                mode = "w"
                header_flag = True
            else:
                mode = "a"
                header_flag = False
            df = pd.DataFrame(frame_array, columns=image_headers_full)
            df.to_csv(image_path, index=False, mode=mode, header=header_flag)

    def _save_adjacency_maps(self):
        """
        Construct adjacency maps (as edge lists) for voxels, nodes, branches, and components.
        This version avoids dense matrices to reduce memory usage.
        """
        v_n = []
        v_b = []
        v_o = []

        for t in range(len(self.voxels.time)):
            num_voxels = len(self.voxels.coords[t])

            # voxel -> node adjacency (voxel index, node index).
            # Vectorize the (voxel, n)-pair construction: per-voxel
            # ragged arrays of node indices become one repeat+concat
            # of `voxel_idx` × `len(nodes_at_that_voxel)` rows.
            if not self.skip_nodes:
                voxel_node_lists = self.voxels.node_labels[t]
                lengths = np.fromiter(
                    (0 if n is None else len(n) for n in voxel_node_lists),
                    dtype=np.int64,
                    count=len(voxel_node_lists),
                )
                total_edges = int(lengths.sum())
                if total_edges:
                    rows = np.repeat(
                        np.arange(len(voxel_node_lists), dtype=np.int64), lengths
                    )
                    cols = np.concatenate(
                        [
                            np.asarray(n, dtype=np.int64)
                            for n in voxel_node_lists
                            if n is not None and len(n) > 0
                        ]
                    )
                    v_n.append(np.column_stack((rows, cols)))
                else:
                    v_n.append(np.zeros((0, 2), dtype=np.int64))

            # voxel -> branch adjacency (voxel index, branch index 0-based)
            branch_labels = np.asarray(self.voxels.branch_labels[t], dtype=np.int64)
            mask = branch_labels > 0
            if np.any(mask):
                rows = np.nonzero(mask)[0]
                cols = branch_labels[mask] - 1
                v_b.append(np.column_stack((rows, cols)))
            else:
                v_b.append(np.zeros((0, 2), dtype=np.int64))

            # voxel -> component adjacency (voxel index, component label)
            component_labels = np.asarray(self.voxels.component_labels[t], dtype=np.int64)
            mask = component_labels > 0
            if np.any(mask):
                rows = np.nonzero(mask)[0]
                cols = component_labels[mask]
                v_o.append(np.column_stack((rows, cols)))
            else:
                v_o.append(np.zeros((0, 2), dtype=np.int64))

        n_b = []
        n_o = []
        if not self.skip_nodes:
            for t in range(len(self.nodes.time)):
                # node -> branch
                node_labels = np.asarray(self.nodes.branch_label[t], dtype=np.int64)
                branch_labels = np.asarray(self.branches.branch_label[t], dtype=np.int64)
                if len(branch_labels) == 0:
                    n_b.append(np.zeros((0, 2), dtype=np.int64))
                else:
                    max_label = int(branch_labels.max())
                    label_to_idx = np.full(max_label + 1, -1, dtype=np.int64)
                    label_to_idx[branch_labels] = np.arange(len(branch_labels), dtype=np.int64)
                    branch_idx = label_to_idx[node_labels]
                    mask = branch_idx >= 0
                    rows = np.nonzero(mask)[0]
                    cols = branch_idx[mask]
                    n_b.append(np.column_stack((rows, cols)))

                # node -> component
                node_comp_labels = np.asarray(self.nodes.component_label[t], dtype=np.int64)
                comp_labels = np.asarray(self.components.component_label[t], dtype=np.int64)
                if len(comp_labels) == 0:
                    n_o.append(np.zeros((0, 2), dtype=np.int64))
                else:
                    max_label_c = int(comp_labels.max())
                    label_to_idx_c = np.full(max_label_c + 1, -1, dtype=np.int64)
                    label_to_idx_c[comp_labels] = np.arange(len(comp_labels), dtype=np.int64)
                    comp_idx = label_to_idx_c[node_comp_labels]
                    mask_c = comp_idx >= 0
                    rows_c = np.nonzero(mask_c)[0]
                    cols_c = comp_idx[mask_c]
                    n_o.append(np.column_stack((rows_c, cols_c)))

        b_o = []
        for t in range(len(self.branches.time)):
            branch_comp_labels = np.asarray(self.branches.component_label[t], dtype=np.int64)
            comp_labels = np.asarray(self.components.component_label[t], dtype=np.int64)
            if len(comp_labels) == 0:
                b_o.append(np.zeros((0, 2), dtype=np.int64))
            else:
                max_label_c = int(comp_labels.max())
                label_to_idx_c = np.full(max_label_c + 1, -1, dtype=np.int64)
                label_to_idx_c[comp_labels] = np.arange(len(comp_labels), dtype=np.int64)
                comp_idx = label_to_idx_c[branch_comp_labels]
                mask_c = comp_idx >= 0
                rows_c = np.nonzero(mask_c)[0]
                cols_c = comp_idx[mask_c]
                b_o.append(np.column_stack((rows_c, cols_c)))

        edges = {
            "v_b": v_b,
            "v_n": v_n,
            "v_o": v_o,
            "n_b": n_b,
            "n_o": n_o,
            "b_o": b_o,
        }

        with open(self.im_info.pipeline_paths["adjacency_maps"], "wb") as f:
            pickle.dump(edges, f)

    def _run_hierarchy(self):
        # Lazily initialize flow interpolators only if needed
        if (
            self.enable_motility
            and not self.im_info.no_t
            and self.num_t is not None
            and self.num_t > 1
        ):
            self.flow_interpolator_fw = FlowInterpolator(self.im_info)
            self.flow_interpolator_bw = FlowInterpolator(self.im_info, forward=False)
        else:
            self.flow_interpolator_fw = None
            self.flow_interpolator_bw = None

        self._allocate_memory()
        self._get_hierarchies()
        self._save_dfs()

        if self.viewer is not None:
            self.viewer.status = "Finalizing run."

        if self.enable_adjacency:
            self._save_adjacency_maps()

        if self.viewer is not None:
            self.viewer.status = "Done!"

    def run(self):
        """
        Main execution method.
        """
        device = adaptive_run.normalize_device(self.device)
        device_order = adaptive_run.device_cascade(self.device)
        if device == "gpu" and device_order == ["cpu"]:
            logger.warning("Hierarchy: GPU requested but not available; falling back to CPU.")

        start_low_memory = bool(self.low_memory) or adaptive_run.should_use_low_memory(
            self.im_info,
            include_gpu="gpu" in device_order,
            device_order=device_order,
        )
        if start_low_memory and not self.low_memory:
            logger.info("Hierarchy: enabling low-memory mode based on estimated usage.")

        last_exc = None
        for dev, low in adaptive_run.mode_candidates(device_order, start_low_memory):
            try:
                self._set_backend(dev)
                self._set_low_memory(low)
                self._run_hierarchy()
                return
            except Exception as exc:
                last_exc = exc
                if adaptive_run.is_gpu_unavailable_error(exc) and dev in ("gpu", "mps"):
                    logger.warning("Hierarchy: GPU backend unavailable; retrying on CPU.")
                    continue
                if adaptive_run.is_oom_error(exc):
                    logger.warning(
                        "Hierarchy: OOM on %s/%s; retrying with lower settings.",
                        dev,
                        "low-memory" if low else "high-memory",
                    )
                    continue
                raise
        raise last_exc


def append_to_array(to_append):
    """
    Convert feature dict into list-of-arrays + headers.
    """
    new_array = []
    new_headers = []
    for feature, stats in to_append.items():
        if not isinstance(stats, dict):
            stats = {"raw": stats}
            stats["raw"] = [np.array(stats["raw"])]
        for stat, vals in stats.items():
            vals = np.array(vals)[0]
            new_array.append(vals)
            new_headers.append(f"{feature}_{stat}")
    return new_array, new_headers


def _sorted_label_groups(labels):
    """Sort `labels` once and split positions by unique label value.

    Returns ``(unique_labels_ascending, list_of_index_arrays)``. Each
    index array holds the positions of one unique label, in ascending
    order — same ordering ``np.argwhere(labels == lbl).flatten()``
    would produce.
    """
    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        return np.empty(0, dtype=labels_arr.dtype), []
    order = np.argsort(labels_arr, kind="stable")
    sorted_labels = labels_arr[order]
    unique, starts = np.unique(sorted_labels, return_index=True)
    groups = np.split(order, starts[1:])
    return unique, groups


def _group_indices_by_label(labels, *, drop_label=0):
    """O(N log N) replacement for
    ``[np.argwhere(labels == lbl).flatten() for lbl in np.unique(labels) if lbl != drop_label]``.

    Returns one ascending-position index array per non-``drop_label``
    unique label, in ascending-label order.
    """
    unique, groups = _sorted_label_groups(labels)
    return [g for u, g in zip(unique, groups) if u != drop_label]


def _group_indices_for_keys(labels, keys):
    """For each key in `keys` (preserving caller-supplied order), return
    positions in `labels` equal to that key (in ascending position order).

    Replaces the
    ``[np.argwhere(other_labels == lbl).flatten() for lbl in np.unique(key_labels) if lbl != 0]``
    pattern in ``Components._get_aggregate_stats`` — where the
    iteration key set comes from one label array but the matched
    positions live in another. O((N + K) log N) via sort + per-key
    ``np.searchsorted``. Keys absent from ``labels`` produce empty
    groups (preserves the 1:1-with-keys list shape).
    """
    labels_arr = np.asarray(labels)
    keys_arr = np.asarray(keys)
    if keys_arr.size == 0:
        return []
    if labels_arr.size == 0:
        return [np.empty(0, dtype=np.intp) for _ in range(keys_arr.size)]
    order = np.argsort(labels_arr, kind="stable")
    sorted_labels = labels_arr[order]
    left = np.searchsorted(sorted_labels, keys_arr, side="left")
    right = np.searchsorted(sorted_labels, keys_arr, side="right")
    return [order[lo:hi] for lo, hi in zip(left, right)]


class Voxels:
    """
    Voxel-level features.
    """

    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.coords = []

        # voxel metrics
        self.x = []
        self.y = []
        self.z = []
        self.intensity = []
        self.structure = []

        self.vec01 = []
        self.vec12 = []

        self.angular_acc = []
        self.angular_vel = []
        self.angular_vel_vector = []
        self.linear_acc = []
        self.linear_vel = []
        self.linear_vel_vector = []

        self.rel_angular_acc = []
        self.rel_angular_vel = []
        self.rel_linear_acc = []
        self.rel_linear_vel = []
        self.rel_directionality = []

        self.node_labels = []
        self.branch_labels = []
        self.component_labels = []
        self.image_name = []

        self.node_dim0_lims = []
        self.node_dim1_lims = []
        self.node_dim2_lims = []
        self.node_voxel_idxs = []

        self.stats_to_aggregate = [
            "linear_vel",
            "angular_vel",
            "linear_acc",
            "angular_acc",
            "rel_linear_vel",
            "rel_angular_vel",
            "rel_linear_acc",
            "rel_angular_acc",
            "rel_directionality",
            "structure",
            "intensity",
        ]

        self.features_to_save = self.stats_to_aggregate + ["x", "y", "z"]

    def _get_node_info(self, t, frame_coords):
        """
        Compute node bounding boxes and voxel->node assignments.
        """
        skeleton_pixels = np.argwhere(self.hierarchy.im_pixel_class[t] > 0)
        skeleton_radius = self.hierarchy.im_distance[t][tuple(skeleton_pixels.T)]

        lims_dim0 = (
            skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 0, np.newaxis]
        ).astype(int)
        lims_dim0[:, 1] += 1
        lims_dim1 = (
            skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 1, np.newaxis]
        ).astype(int)
        lims_dim1[:, 1] += 1

        lims_dim0[lims_dim0 < 0] = 0
        lims_dim1[lims_dim1 < 0] = 0

        if not self.hierarchy.im_info.no_z:
            lims_dim2 = (
                skeleton_radius[:, np.newaxis] * np.array([-1, 1])
                + skeleton_pixels[:, 2, np.newaxis]
            ).astype(int)
            lims_dim2[:, 1] += 1
            lims_dim2[lims_dim2 < 0] = 0
            max_dim0 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index("Z")]
            max_dim1 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index("Y")]
            max_dim2 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index("X")]
            lims_dim2[lims_dim2 > max_dim2] = max_dim2
        else:
            lims_dim2 = None
            max_dim0 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index("Y")]
            max_dim1 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index("X")]

        lims_dim0[lims_dim0 > max_dim0] = max_dim0
        lims_dim1[lims_dim1 > max_dim1] = max_dim1

        self.node_dim0_lims.append(lims_dim0)
        self.node_dim1_lims.append(lims_dim1)
        self.node_dim2_lims.append(lims_dim2)

        frame_coords = np.array(frame_coords)
        num_nodes = len(skeleton_pixels)
        num_voxels = len(frame_coords)
        chunk_size = self.hierarchy._resolve_node_chunk_size(num_nodes, num_voxels)

        def _process_chunks(local_chunk_size):
            chunk_node_voxel_idxs = {idx: [] for idx in range(num_nodes)}
            chunk_nodes_idxs = []
            if num_voxels == 0:
                return chunk_nodes_idxs, chunk_node_voxel_idxs
            num_chunks = int(np.ceil(num_voxels / local_chunk_size))
            for chunk_num in range(num_chunks):
                logger.debug(f"Processing chunk {chunk_num + 1} of {num_chunks}")
                start = chunk_num * local_chunk_size
                end = min((chunk_num + 1) * local_chunk_size, num_voxels)
                chunk_frame_coords = frame_coords[start:end]

                if not self.hierarchy.im_info.no_z:
                    dim0_coords, dim1_coords, dim2_coords = (
                        chunk_frame_coords[:, 0],
                        chunk_frame_coords[:, 1],
                        chunk_frame_coords[:, 2],
                    )
                    dim0_mask = (lims_dim0[:, 0][:, None] <= dim0_coords) & (
                        lims_dim0[:, 1][:, None] >= dim0_coords
                    )
                    dim1_mask = (lims_dim1[:, 0][:, None] <= dim1_coords) & (
                        lims_dim1[:, 1][:, None] >= dim1_coords
                    )
                    dim2_mask = (lims_dim2[:, 0][:, None] <= dim2_coords) & (
                        lims_dim2[:, 1][:, None] >= dim2_coords
                    )
                    mask = dim0_mask & dim1_mask & dim2_mask
                else:
                    dim0_coords, dim1_coords = chunk_frame_coords[:, 0], chunk_frame_coords[:, 1]
                    dim0_mask = (lims_dim0[:, 0][:, None] <= dim0_coords) & (
                        lims_dim0[:, 1][:, None] >= dim0_coords
                    )
                    dim1_mask = (lims_dim1[:, 0][:, None] <= dim1_coords) & (
                        lims_dim1[:, 1][:, None] >= dim1_coords
                    )
                    mask = dim0_mask & dim1_mask

                frame_coord_nodes_idxs = [[] for _ in range(mask.shape[1])]
                rows, cols = np.nonzero(mask)
                for row, col in zip(rows, cols):
                    frame_coord_nodes_idxs[col].append(row)
                frame_coord_nodes_idxs = [
                    np.array(indices) for indices in frame_coord_nodes_idxs
                ]

                chunk_nodes_idxs.extend(frame_coord_nodes_idxs)

                for i in range(num_nodes):
                    chunk_node_voxel_idxs[i].extend(np.nonzero(mask[i])[0] + start)
            return chunk_nodes_idxs, chunk_node_voxel_idxs

        while True:
            try:
                chunk_nodes_idxs, chunk_node_voxel_idxs = _process_chunks(chunk_size)
                break
            except MemoryError:
                if chunk_size <= 1:
                    raise
                logger.warning(
                    "Node assignment OOM with chunk_size=%d; retrying smaller chunks.",
                    chunk_size,
                )
                chunk_size = max(1, chunk_size // 2)

        self.node_labels.append(chunk_nodes_idxs)
        chunk_node_voxel_idxs = [
            np.array(chunk_node_voxel_idxs[i]) for i in range(len(skeleton_pixels))
        ]
        self.node_voxel_idxs.append(chunk_node_voxel_idxs)

    def _get_min_euc_dist(self, t, vec):
        """
        Compute index of voxel with minimal Euclidean distance for each branch label.
        Returns a numpy array of length max_label+1 where element [label] is the
        index into vec/coords for that representative voxel (or NaN if none).
        """
        euc_dist = np.linalg.norm(vec, axis=1)
        branch_labels = np.asarray(self.branch_labels[t], dtype=np.int64)

        if branch_labels.size == 0:
            return np.array([], dtype=float)

        max_label = int(branch_labels.max())
        idxmin = np.full(max_label + 1, np.nan, dtype=float)

        unique_labels = np.unique(branch_labels)
        for lbl in unique_labels:
            mask = branch_labels == lbl
            vals = euc_dist[mask]
            valid = ~np.isnan(vals)
            if not np.any(valid):
                continue
            local_idx = np.nanargmin(vals[valid])
            global_indices = np.nonzero(mask)[0][valid]
            idxmin[lbl] = global_indices[local_idx]

        return idxmin

    def _get_ref_coords(self, coords_a, coords_b, idxmin, t):
        """
        Map every voxel to reference coordinates based on branch label.
        """
        branch_labels = np.asarray(self.branch_labels[t], dtype=np.int64)
        max_label = len(idxmin) - 1
        branch_labels_clipped = np.clip(branch_labels, 0, max_label)
        # `vals_a` and `vals_b` were previously computed as two
        # identical `idxmin[branch_labels_clipped]` gathers — same
        # input, same lookup, same output. Compute once; share the
        # NaN mask too.
        vals = idxmin[branch_labels_clipped]
        nan_mask = np.isnan(vals)
        vals_int = vals.copy()
        vals_int[nan_mask] = 0
        vals_int = vals_int.astype(int)

        ref_a = coords_a[vals_int]
        ref_b = coords_b[vals_int]
        ref_a[nan_mask] = np.nan
        ref_b[nan_mask] = np.nan

        return ref_a, ref_b

    def _get_linear_velocity(self, ra, rb):
        lin_disp = rb - ra
        dt = self.hierarchy.im_info.dim_res["T"]
        lin_vel = lin_disp / dt
        lin_vel_mag = np.linalg.norm(lin_vel, axis=1)
        # orientation not used downstream; don't return it to save memory
        return lin_vel, lin_vel_mag, None

    def _get_angular_velocity_2d(self, ra, rb):
        theta_a = np.arctan2(ra[:, 1], ra[:, 0])
        theta_b = np.arctan2(rb[:, 1], rb[:, 0])
        delta_theta = theta_b - theta_a
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
        dt = self.hierarchy.im_info.dim_res["T"]
        ang_vel = delta_theta / dt
        ang_vel_mag = np.abs(ang_vel)
        ang_vel_orient = np.sign(ang_vel)
        return ang_vel, ang_vel_mag, ang_vel_orient

    def _get_angular_velocity_3d(self, ra, rb):
        cross_product = np.cross(ra, rb, axis=1)
        norm = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
        ang_disp = np.divide(cross_product.T, norm.T).T
        ang_disp[norm == 0] = [np.nan, np.nan, np.nan]

        dt = self.hierarchy.im_info.dim_res["T"]
        ang_vel = ang_disp / dt
        ang_vel_mag = np.linalg.norm(ang_vel, axis=1)
        ang_vel_orient = (ang_vel.T / ang_vel_mag).T
        ang_vel_orient = np.where(
            np.isinf(ang_vel_orient),
            [np.nan, np.nan, np.nan],
            ang_vel_orient,
        )
        return ang_vel, ang_vel_mag, ang_vel_orient

    def _get_angular_velocity(self, ra, rb):
        if self.hierarchy.im_info.no_z:
            return self._get_angular_velocity_2d(ra, rb)
        return self._get_angular_velocity_3d(ra, rb)

    def _get_motility_stats(self, t, coords_1_px):
        """
        Compute motility-related features. Can be skipped entirely by setting
        hierarchy.enable_motility = False.
        """
        coords_1_px = coords_1_px.astype("float32")
        dims = 2 if self.hierarchy.im_info.no_z else 3
        n = len(coords_1_px)

        # If motility is disabled or we don't have flows, fill with NaNs and return.
        if (
            not self.hierarchy.enable_motility
            or self.hierarchy.flow_interpolator_fw is None
            or self.hierarchy.flow_interpolator_bw is None
            or self.hierarchy.num_t is None
            or self.hierarchy.num_t < 2
        ):
            self.vec01.append(np.full((n, dims), np.nan, dtype=np.float32))
            self.vec12.append(np.full((n, dims), np.nan, dtype=np.float32))
            nan_vec = np.full((n, dims), np.nan, dtype=np.float32)
            nan_arr = np.full(n, np.nan, dtype=np.float32)

            self.linear_vel_vector.append(nan_vec)
            self.linear_vel.append(nan_arr)
            self.angular_vel_vector.append(nan_arr if dims == 2 else nan_vec)
            self.angular_vel.append(nan_arr)
            self.rel_linear_vel.append(nan_arr)
            self.rel_angular_vel.append(nan_arr)
            self.rel_directionality.append(nan_arr)
            self.linear_acc.append(nan_arr)
            self.angular_acc.append(nan_arr)
            self.rel_linear_acc.append(nan_arr)
            self.rel_angular_acc.append(nan_arr)
            return

        vec01 = []
        vec12 = []

        if t > 0:
            vec01_px = self.hierarchy.flow_interpolator_bw.interpolate_coord(coords_1_px, t)
            vec01 = vec01_px * self.hierarchy.spacing
            self.vec01.append(vec01.astype(np.float32))
        else:
            self.vec01.append(np.full((n, dims), np.nan, dtype=np.float32))

        if t < self.hierarchy.num_t - 1:
            vec12_px = self.hierarchy.flow_interpolator_fw.interpolate_coord(coords_1_px, t)
            vec12 = vec12_px * self.hierarchy.spacing
            self.vec12.append(vec12.astype(np.float32))
        else:
            self.vec12.append(np.full((n, dims), np.nan, dtype=np.float32))

        coords_1 = coords_1_px * self.hierarchy.spacing

        # Forward and backward in time
        if len(vec01) and len(vec12):
            coords_0_px = coords_1_px - vec01_px
            coords_0 = coords_0_px * self.hierarchy.spacing

            lin_vel_01, lin_vel_mag_01, _ = self._get_linear_velocity(coords_0, coords_1)
            ang_vel_01, ang_vel_mag_01, _ = self._get_angular_velocity(coords_0, coords_1)

            idxmin01 = self._get_min_euc_dist(t, vec01)
            ref_coords01 = self._get_ref_coords(coords_0, coords_1, idxmin01, t)
            ref_coords01[0][np.isnan(vec01)] = np.nan
            ref_coords01[1][np.isnan(vec01)] = np.nan
            r0_rel_01 = coords_0 - ref_coords01[0]
            r1_rel_01 = coords_1 - ref_coords01[1]

            lin_vel_rel_01, lin_vel_mag_rel_01, _ = self._get_linear_velocity(
                r0_rel_01, r1_rel_01
            )
            ang_vel_rel_01, ang_vel_mag_rel_01, _ = self._get_angular_velocity(
                r0_rel_01, r1_rel_01
            )

        if len(vec12):
            coords_2_px = coords_1_px + vec12_px
            coords_2 = coords_2_px * self.hierarchy.spacing

            lin_vel, lin_vel_mag, _ = self._get_linear_velocity(coords_1, coords_2)
            ang_vel, ang_vel_mag, _ = self._get_angular_velocity(coords_1, coords_2)

            idxmin12 = self._get_min_euc_dist(t, vec12)
            ref_coords12 = self._get_ref_coords(coords_1, coords_2, idxmin12, t)
            ref_coords12[0][np.isnan(vec12)] = np.nan
            ref_coords12[1][np.isnan(vec12)] = np.nan
            r1_rel_12 = coords_1 - ref_coords12[0]
            r2_rel_12 = coords_2 - ref_coords12[1]

            lin_vel_rel, lin_vel_mag_rel, _ = self._get_linear_velocity(
                r1_rel_12, r2_rel_12
            )
            ang_vel_rel, ang_vel_mag_rel, _ = self._get_angular_velocity(
                r1_rel_12, r2_rel_12
            )

            r2_rel_mag_12 = np.linalg.norm(r2_rel_12, axis=1)
            r1_rel_mag_12 = np.linalg.norm(r1_rel_12, axis=1)
            denom = r2_rel_mag_12 + r1_rel_mag_12
            directionality_rel = np.divide(
                np.abs(r2_rel_mag_12 - r1_rel_mag_12),
                denom,
                out=np.full_like(denom, np.nan, dtype=np.float32),
                where=denom != 0,
            )
        else:
            lin_vel = np.full((n, dims), np.nan, dtype=np.float32)
            lin_vel_mag = np.full(n, np.nan, dtype=np.float32)
            ang_vel_mag = np.full(n, np.nan, dtype=np.float32)
            lin_vel_rel = np.full((n, dims), np.nan, dtype=np.float32)
            lin_vel_mag_rel = np.full(n, np.nan, dtype=np.float32)
            ang_vel_mag_rel = np.full(n, np.nan, dtype=np.float32)
            directionality_rel = np.full(n, np.nan, dtype=np.float32)
            if dims == 3:
                ang_vel = np.full((n, dims), np.nan, dtype=np.float32)
                ang_vel_rel = np.full((n, dims), np.nan, dtype=np.float32)
            else:
                ang_vel = np.full(n, np.nan, dtype=np.float32)
                ang_vel_rel = np.full(n, np.nan, dtype=np.float32)

        self.linear_vel_vector.append(lin_vel.astype(np.float32))
        self.linear_vel.append(lin_vel_mag.astype(np.float32))
        self.angular_vel_vector.append(ang_vel.astype(np.float32))
        self.angular_vel.append(ang_vel_mag.astype(np.float32))
        self.rel_linear_vel.append(lin_vel_mag_rel.astype(np.float32))
        self.rel_angular_vel.append(ang_vel_mag_rel.astype(np.float32))
        self.rel_directionality.append(directionality_rel.astype(np.float32))

        if len(vec01) and len(vec12):
            dt = self.hierarchy.im_info.dim_res["T"]
            lin_acc = (lin_vel - lin_vel_01) / dt
            lin_acc_mag = np.linalg.norm(lin_acc, axis=1)
            ang_acc = (ang_vel - ang_vel_01) / dt

            lin_acc_rel = (lin_vel_rel - lin_vel_rel_01) / dt
            lin_acc_rel_mag = np.linalg.norm(lin_acc_rel, axis=1)

            ang_acc_rel = (ang_vel_rel - ang_vel_rel_01) / dt
            if self.hierarchy.im_info.no_z:
                ang_acc_mag = np.abs(ang_acc)
                ang_acc_rel_mag = np.abs(ang_acc_rel)
            else:
                ang_acc_mag = np.linalg.norm(ang_acc, axis=1)
                ang_acc_rel_mag = np.linalg.norm(ang_acc_rel, axis=1)
        else:
            lin_acc_mag = np.full(n, np.nan, dtype=np.float32)
            ang_acc_mag = np.full(n, np.nan, dtype=np.float32)
            lin_acc_rel_mag = np.full(n, np.nan, dtype=np.float32)
            ang_acc_rel_mag = np.full(n, np.nan, dtype=np.float32)

        self.linear_acc.append(lin_acc_mag.astype(np.float32))
        self.angular_acc.append(ang_acc_mag.astype(np.float32))
        self.rel_linear_acc.append(lin_acc_rel_mag.astype(np.float32))
        self.rel_angular_acc.append(ang_acc_rel_mag.astype(np.float32))

    def _run_frame(self, t=None):
        frame_coords = np.argwhere(self.hierarchy.label_components[t] > 0)
        self.coords.append(frame_coords)

        frame_component_labels = self.hierarchy.label_components[t][tuple(frame_coords.T)]
        self.component_labels.append(frame_component_labels)

        frame_branch_labels = self.hierarchy.label_branches[t][tuple(frame_coords.T)]
        self.branch_labels.append(frame_branch_labels)

        frame_intensity_vals = self.hierarchy.im_raw[t][tuple(frame_coords.T)]
        self.intensity.append(frame_intensity_vals)

        if not self.hierarchy.im_info.no_z:
            frame_z_vals = frame_coords[:, 0]
            frame_y_vals = frame_coords[:, 1]
            frame_x_vals = frame_coords[:, 2]
        else:
            frame_z_vals = np.full(len(frame_coords), np.nan)
            frame_y_vals = frame_coords[:, 0]
            frame_x_vals = frame_coords[:, 1]
        self.z.append(frame_z_vals)
        self.y.append(frame_y_vals)
        self.x.append(frame_x_vals)

        frame_structure_vals = self.hierarchy.im_struct[t][tuple(frame_coords.T)]
        self.structure.append(frame_structure_vals)

        frame_t = np.ones(frame_coords.shape[0], dtype=int) * t
        self.time.append(frame_t)

        im_name = (
            np.ones(frame_coords.shape[0], dtype=object)
            * self.hierarchy.im_info.file_info.filename_no_ext
        )
        self.image_name.append(im_name)

        if not self.hierarchy.skip_nodes:
            self._get_node_info(t, frame_coords)

        self._get_motility_stats(t, frame_coords)

    def run(self):
        if self.hierarchy.num_t is None:
            self.hierarchy.num_t = 1
        for t in range(self.hierarchy.num_t):
            if self.hierarchy.viewer is not None:
                self.hierarchy.viewer.status = (
                    f"Extracting voxel features. Frame: {t + 1} of {self.hierarchy.num_t}."
                )
            self._run_frame(t)


def aggregate_stats_for_class(child_class, t, list_of_idxs, low_memory: bool = False):
    """
    Aggregate mean/std/min/max/sum over groups of indices for a given class at time t.

    Parameters
    ----------
    child_class : object with stats_to_aggregate and per-stat per-frame arrays
    t : int
    list_of_idxs : list of 1D index arrays (groups)
    low_memory : bool
        If True, use a slower but memory-light implementation.
    """
    aggregate_stats = {
        stat_name: {"mean": [], "std_dev": [], "min": [], "max": [], "sum": []}
        for stat_name in child_class.stats_to_aggregate
        if stat_name != "reassigned_label"
    }

    if low_memory:
        # Loop over groups and stats to avoid building large intermediate arrays.
        for stat_name in child_class.stats_to_aggregate:
            if stat_name == "reassigned_label":
                continue

            stat_array = np.array(getattr(child_class, stat_name)[t])

            # Skip multi-dimensional stats as in original implementation
            if stat_array.ndim > 1:
                continue

            for idxs in list_of_idxs:
                if len(idxs) == 0:
                    aggregate_stats[stat_name]["mean"].append(np.nan)
                    aggregate_stats[stat_name]["std_dev"].append(np.nan)
                    aggregate_stats[stat_name]["min"].append(np.nan)
                    aggregate_stats[stat_name]["max"].append(np.nan)
                    aggregate_stats[stat_name]["sum"].append(np.nan)
                else:
                    vals = stat_array[idxs.astype(int)]
                    aggregate_stats[stat_name]["mean"].append(np.nanmean(vals))
                    aggregate_stats[stat_name]["std_dev"].append(np.nanstd(vals))
                    aggregate_stats[stat_name]["min"].append(np.nanmin(vals))
                    aggregate_stats[stat_name]["max"].append(np.nanmax(vals))
                    aggregate_stats[stat_name]["sum"].append(np.nansum(vals))

        # Convert lists to arrays
        for stat_name in aggregate_stats:
            for key in aggregate_stats[stat_name]:
                arr = np.asarray(aggregate_stats[stat_name][key])
                if arr.ndim == 0:
                    arr = arr.reshape(1, 1)
                elif arr.ndim == 1:
                    arr = arr[None, :]
                aggregate_stats[stat_name][key] = arr
        return aggregate_stats

    # Original vectorized implementation (faster but more memory-intensive)
    aggregate_stats = {
        stat_name: {"mean": [], "std_dev": [], "min": [], "max": [], "sum": []}
        for stat_name in child_class.stats_to_aggregate
        if stat_name != "reassigned_label"
    }

    largest_idx = max((len(idxs) for idxs in list_of_idxs), default=0)

    for stat_name in child_class.stats_to_aggregate:
        if stat_name == "reassigned_label":
            continue

        stat_array = np.array(getattr(child_class, stat_name)[t])

        if stat_array.ndim > 1:
            # skip multi-dimensional stats, as in original code
            continue

        # append a NaN at the end
        stat_array = np.append(stat_array, np.nan)

        # Build big index array
        idxs_array = np.full(
            (len(list_of_idxs), largest_idx), len(stat_array) - 1, dtype=int
        )
        for i, idxs in enumerate(list_of_idxs):
            if len(idxs) > 0:
                idxs_array[i, : len(idxs)] = idxs

        stat_values = stat_array[idxs_array.astype(int)]
        if stat_values.shape[1] == 0:
            stat_values = np.full((stat_values.shape[0], 1), np.nan)

        mean = np.nanmean(stat_values, axis=1)
        std_dev = np.nanstd(stat_values, axis=1)
        min_val = np.nanmin(stat_values, axis=1)
        max_val = np.nanmax(stat_values, axis=1)
        sum_val = np.nansum(stat_values, axis=1)

        aggregate_stats[stat_name]["mean"].append(mean)
        aggregate_stats[stat_name]["std_dev"].append(std_dev)
        aggregate_stats[stat_name]["min"].append(min_val)
        aggregate_stats[stat_name]["max"].append(max_val)
        aggregate_stats[stat_name]["sum"].append(sum_val)

    # Convert lists-of-arrays to arrays-of-arrays
    for stat_name in aggregate_stats:
        for key in aggregate_stats[stat_name]:
            aggregate_stats[stat_name][key] = np.array(aggregate_stats[stat_name][key])

    return aggregate_stats


class Nodes:
    """
    Node-level features.
    """

    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.nodes = []

        self.aggregate_voxel_metrics = []

        self.z = []
        self.x = []
        self.y = []
        self.node_thickness = []
        self.divergence = []
        self.convergence = []
        self.vergere = []

        self.stats_to_aggregate = [
            "divergence",
            "convergence",
            "vergere",
            "node_thickness",
        ]

        self.features_to_save = self.stats_to_aggregate + ["x", "y", "z"]

        self.voxel_idxs = self.hierarchy.voxels.node_voxel_idxs
        self.branch_label = []
        self.component_label = []
        self.image_name = []

        self.node_z_lims = self.hierarchy.voxels.node_dim0_lims
        self.node_y_lims = self.hierarchy.voxels.node_dim1_lims
        self.node_x_lims = self.hierarchy.voxels.node_dim2_lims

    def _get_aggregate_voxel_stats(self, t):
        frame_agg = aggregate_stats_for_class(
            self.hierarchy.voxels,
            t,
            self.hierarchy.voxels.node_voxel_idxs[t],
            low_memory=self.hierarchy.low_memory,
        )
        self.aggregate_voxel_metrics.append(frame_agg)

    def _get_node_stats(self, t):
        radius = distance_check(
            self.hierarchy.im_border_mask[t],
            self.nodes[t],
            self.hierarchy.spacing,
        )
        self.node_thickness.append(radius * 2)

        divergence = []
        convergence = []
        vergere = []
        z = []
        y = []
        x = []
        for i, node in enumerate(self.nodes[t]):
            vox_idxs = self.voxel_idxs[t][i]
            if len(vox_idxs) == 0:
                divergence.append(np.nan)
                convergence.append(np.nan)
                vergere.append(np.nan)
                z.append(np.nan)
                y.append(np.nan)
                x.append(np.nan)
                continue

            coords_vox = self.hierarchy.voxels.coords[t][vox_idxs]
            if not self.hierarchy.im_info.no_z:
                z.append(
                    np.nanmean(coords_vox[:, 0]) * self.hierarchy.spacing[0]
                )
                y.append(
                    np.nanmean(coords_vox[:, 1]) * self.hierarchy.spacing[1]
                )
                x.append(
                    np.nanmean(coords_vox[:, 2]) * self.hierarchy.spacing[2]
                )
            else:
                z.append(np.nan)
                y.append(
                    np.nanmean(coords_vox[:, 0]) * self.hierarchy.spacing[0]
                )
                x.append(
                    np.nanmean(coords_vox[:, 1]) * self.hierarchy.spacing[1]
                )

            dist_vox_node = (coords_vox - self.nodes[t][i]).astype(float, copy=False)
            dist_vox_node_mag = np.linalg.norm(dist_vox_node, axis=1, keepdims=True)
            dir_vox_node = np.divide(
                dist_vox_node,
                dist_vox_node_mag,
                out=np.full_like(dist_vox_node, np.nan),
                where=dist_vox_node_mag != 0,
            )

            vec01 = self.hierarchy.voxels.vec01[t][vox_idxs]
            vec12 = self.hierarchy.voxels.vec12[t][vox_idxs]

            dot_prod_01 = -np.nanmean(np.sum(-vec01 * dir_vox_node, axis=1))
            convergence.append(dot_prod_01)

            dot_prod_12 = np.nanmean(np.sum(vec12 * dir_vox_node, axis=1))
            divergence.append(dot_prod_12)

            vergere.append(dot_prod_01 + dot_prod_12)

        self.divergence.append(divergence)
        self.convergence.append(convergence)
        self.vergere.append(vergere)
        self.z.append(z)
        self.y.append(y)
        self.x.append(x)

    def _run_frame(self, t):
        frame_skel_coords = np.argwhere(self.hierarchy.im_pixel_class[t] > 0)
        self.nodes.append(frame_skel_coords)

        frame_t = np.ones(frame_skel_coords.shape[0], dtype=int) * t
        self.time.append(frame_t)

        frame_component_label = self.hierarchy.label_components[t][
            tuple(frame_skel_coords.T)
        ]
        self.component_label.append(frame_component_label)

        frame_branch_label = self.hierarchy.label_branches[t][
            tuple(frame_skel_coords.T)
        ]
        self.branch_label.append(frame_branch_label)

        im_name = (
            np.ones(frame_skel_coords.shape[0], dtype=object)
            * self.hierarchy.im_info.file_info.filename_no_ext
        )
        self.image_name.append(im_name)

        self._get_aggregate_voxel_stats(t)
        self._get_node_stats(t)

    def run(self):
        if self.hierarchy.skip_nodes:
            return
        for t in range(self.hierarchy.num_t):
            if self.hierarchy.viewer is not None:
                self.hierarchy.viewer.status = (
                    f"Extracting node features. Frame: {t + 1} of {self.hierarchy.num_t}."
                )
            self._run_frame(t)


def distance_check(border_mask, check_coords, spacing):
    """
    Compute distance from points in check_coords to nearest border point.
    """
    border_coords = np.argwhere(border_mask) * spacing
    if border_coords.size == 0:
        return np.full(len(check_coords), np.nan, dtype=float)
    border_tree = spatial.cKDTree(border_coords)
    dist, _ = border_tree.query(check_coords * spacing, k=1)
    return dist


class Branches:
    """
    Branch-level features.
    """

    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.branch_label = []

        self.aggregate_voxel_metrics = []
        self.aggregate_node_metrics = []

        self.z = []
        self.y = []
        self.x = []
        self.branch_length = []
        self.branch_thickness = []
        self.branch_aspect_ratio = []
        self.branch_tortuosity = []
        self.branch_area = []
        self.branch_axis_length_maj = []
        self.branch_axis_length_min = []
        self.branch_extent = []
        self.branch_solidity = []
        self.reassigned_label = []

        self.branch_idxs = []
        self.component_label = []
        self.image_name = []

        self.stats_to_aggregate = [
            "branch_length",
            "branch_thickness",
            "branch_aspect_ratio",
            "branch_tortuosity",
            "branch_area",
            "branch_axis_length_maj",
            "branch_axis_length_min",
            "branch_extent",
            "branch_solidity",
            "reassigned_label",
        ]

        self.features_to_save = self.stats_to_aggregate + ["x", "y", "z"]

    def _get_aggregate_stats(self, t):
        voxel_labels = self.hierarchy.voxels.branch_labels[t]
        grouped_vox_idxs = _group_indices_by_label(voxel_labels)
        vox_agg = aggregate_stats_for_class(
            self.hierarchy.voxels, t, grouped_vox_idxs, low_memory=self.hierarchy.low_memory
        )
        self.aggregate_voxel_metrics.append(vox_agg)

        if not self.hierarchy.skip_nodes:
            node_labels = self.hierarchy.nodes.branch_label[t]
            grouped_node_idxs = _group_indices_by_label(node_labels)
            node_agg = aggregate_stats_for_class(
                self.hierarchy.nodes, t, grouped_node_idxs, low_memory=self.hierarchy.low_memory
            )
            self.aggregate_node_metrics.append(node_agg)

    def _compute_branch_lengths_and_degrees_backend(self, t, xp):
        """
        Compute per-label branch length (centerline length) and neighbor degree per voxel
        using local neighborhood connectivity, with backend xp (np or cp).
        """
        L_cpu = self.hierarchy.im_skel[t]
        L = xp.asarray(L_cpu) if xp is not np else L_cpu
        spacing = self.hierarchy.spacing
        no_z = self.hierarchy.im_info.no_z

        if no_z:
            neighbor_counts = xp.zeros_like(L, dtype=xp.uint8)
            max_label = int(L.max())
            lengths = xp.zeros(max_label + 1, dtype=xp.float32)

            offsets = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    if (dy > 0) or (dy == 0 and dx > 0):
                        offsets.append((dy, dx))

            for dy, dx in offsets:
                y0_start = max(0, dy)
                y0_end = L.shape[0] + min(0, dy)
                x0_start = max(0, dx)
                x0_end = L.shape[1] + min(0, dx)

                y1_start = max(0, -dy)
                y1_end = L.shape[0] - max(0, dy)
                x1_start = max(0, -dx)
                x1_end = L.shape[1] - max(0, dx)

                base = L[y0_start:y0_end, x0_start:x0_end]
                neigh = L[y1_start:y1_end, x1_start:x1_end]

                same = (base > 0) & (base == neigh)
                if not xp.any(same):
                    continue

                same_u8 = same.astype(xp.uint8)
                neighbor_counts[y0_start:y0_end, x0_start:x0_end] += same_u8
                neighbor_counts[y1_start:y1_end, x1_start:x1_end] += same_u8

                labels = base[same]
                if labels.size == 0:
                    continue
                dy_phys = dy * spacing[0]
                dx_phys = dx * spacing[1]
                edge_len = xp.sqrt(dy_phys * dy_phys + dx_phys * dx_phys).astype(xp.float32)
                weights = xp.full(labels.size, edge_len, dtype=xp.float32)
                lengths += xp.bincount(labels.ravel(), weights=weights, minlength=max_label + 1)
        else:
            neighbor_counts = xp.zeros_like(L, dtype=xp.uint8)
            max_label = int(L.max())
            lengths = xp.zeros(max_label + 1, dtype=xp.float32)

            offsets = []
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        if (dz > 0) or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0):
                            offsets.append((dz, dy, dx))

            for dz, dy, dx in offsets:
                z0_start = max(0, dz)
                z0_end = L.shape[0] + min(0, dz)
                y0_start = max(0, dy)
                y0_end = L.shape[1] + min(0, dy)
                x0_start = max(0, dx)
                x0_end = L.shape[2] + min(0, dx)

                z1_start = max(0, -dz)
                z1_end = L.shape[0] - max(0, dz)
                y1_start = max(0, -dy)
                y1_end = L.shape[1] - max(0, dy)
                x1_start = max(0, -dx)
                x1_end = L.shape[2] - max(0, dx)

                base = L[z0_start:z0_end, y0_start:y0_end, x0_start:x0_end]
                neigh = L[z1_start:z1_end, y1_start:y1_end, x1_start:x1_end]

                same = (base > 0) & (base == neigh)
                if not xp.any(same):
                    continue

                same_u8 = same.astype(xp.uint8)
                neighbor_counts[z0_start:z0_end, y0_start:y0_end, x0_start:x0_end] += same_u8
                neighbor_counts[z1_start:z1_end, y1_start:y1_end, x1_start:x1_end] += same_u8

                labels = base[same]
                if labels.size == 0:
                    continue
                dz_phys = dz * spacing[0]
                dy_phys = dy * spacing[1]
                dx_phys = dx * spacing[2]
                edge_len = xp.sqrt(
                    dz_phys * dz_phys + dy_phys * dy_phys + dx_phys * dx_phys
                ).astype(xp.float32)
                weights = xp.full(labels.size, edge_len, dtype=xp.float32)
                lengths += xp.bincount(labels.ravel(), weights=weights, minlength=max_label + 1)

        if hasattr(lengths, 'get'):
            lengths_np = lengths.get()
        else:
            lengths_np = np.asarray(lengths)
        if hasattr(neighbor_counts, 'get'):
            neighbor_counts_np = neighbor_counts.get()
        else:
            neighbor_counts_np = np.asarray(neighbor_counts)
        return lengths_np, neighbor_counts_np

    def _compute_branch_lengths_and_degrees(self, t):
        """
        Wrapper that tries GPU backend if enabled, falls back to CPU otherwise.
        """
        if self.hierarchy.device_type == "cuda":
            try:
                return self._compute_branch_lengths_and_degrees_backend(t, self.hierarchy.xp)
            except Exception as exc:
                if not adaptive_run.is_oom_error(exc):
                    raise
                logger.warning("GPU OOM computing branch lengths; falling back to CPU.")
                adaptive_run.free_gpu_memory(self.hierarchy.xp)
        return self._compute_branch_lengths_and_degrees_backend(t, np)

    def _get_branch_stats(self, t):
        """
        Compute branch length / thickness / aspect ratio / tortuosity and
        regionprops-based morphology, using an O(N) neighborhood-based algorithm
        rather than an O(N^2) distance matrix.
        """
        branch_idxs_arr = np.array(self.branch_idxs[t])
        L = self.hierarchy.im_skel[t]
        spacing = self.hierarchy.spacing

        # Branch lengths and voxel degrees
        label_lengths, neighbor_counts = self._compute_branch_lengths_and_degrees(t)

        unique_labels = np.unique(L[L > 0])
        if unique_labels.size == 0:
            self.branch_tortuosity.append([])
            self.branch_aspect_ratio.append([])
            self.branch_thickness.append([])
            self.branch_length.append([])
            self.branch_area.append([])
            self.branch_axis_length_maj.append([])
            self.branch_axis_length_min.append([])
            self.branch_extent.append([])
            self.branch_solidity.append([])
            self.reassigned_label.append([])
            self.z.append([])
            self.y.append([])
            self.x.append([])
            return

        # Map neighbor_counts to branch voxel indices
        neighbor_counts_branch = neighbor_counts[tuple(branch_idxs_arr.T)]
        tips = np.where(neighbor_counts_branch == 1)[0]
        lone_tips = np.where(neighbor_counts_branch == 0)[0]

        lone_tip_coords = branch_idxs_arr[lone_tips]
        tip_coords = branch_idxs_arr[tips]

        lone_tip_labels = L[tuple(lone_tip_coords.T)] if len(lone_tip_coords) else np.array([], dtype=int)
        tip_labels = L[tuple(tip_coords.T)] if len(tip_coords) else np.array([], dtype=int)

        # Radii from border
        radii = distance_check(
            self.hierarchy.im_border_mask[t],
            branch_idxs_arr,
            spacing,
        )
        lone_tip_radii = radii[lone_tips] if len(lone_tips) else np.array([], dtype=float)
        tip_radii = radii[tips] if len(tips) else np.array([], dtype=float)

        # Base length per label from adjacency.
        # `unique_labels` is ascending (from `np.unique`); single gather
        # + bounds-mask replaces the per-label Python loop.
        unique_labels_int = unique_labels.astype(int, copy=False)
        in_range = unique_labels_int < len(label_lengths)
        # Accumulate in float64 so the tip-radius adjustments below
        # round once (at the final cast back to float32), matching the
        # legacy loop's `base_lengths[i] += 2.0 * radius` semantics
        # where numpy promotes float32 + float64 → float64 before the
        # assignment-time cast. Pure-float32 accumulation would
        # double-round (cast addend, then cast result) and drift by
        # 1 ULP on labels with multiple tips.
        base_lengths_64 = np.zeros(len(unique_labels), dtype=np.float64)
        base_lengths_64[in_range] = label_lengths[unique_labels_int[in_range]].astype(
            np.float64, copy=False
        )

        # Adjust lengths with tip radii. Multiple tips per label must
        # accumulate, so use `np.add.at` (bare `+=` indexed assignment
        # would only apply the last update for duplicate indices).
        # `unique_labels` is sorted ascending; `np.searchsorted` gives
        # the row for each tip label. The `==` check filters out labels
        # that aren't in `unique_labels` (defensive — by construction
        # tip-labels come from `L[tip_coords]` so should always match).
        if lone_tip_labels.size:
            idx = np.searchsorted(unique_labels, lone_tip_labels)
            valid = (idx < len(unique_labels)) & (
                unique_labels[np.clip(idx, 0, len(unique_labels) - 1)] == lone_tip_labels
            )
            np.add.at(base_lengths_64, idx[valid], 2.0 * lone_tip_radii[valid])
        if tip_labels.size:
            idx = np.searchsorted(unique_labels, tip_labels)
            valid = (idx < len(unique_labels)) & (
                unique_labels[np.clip(idx, 0, len(unique_labels) - 1)] == tip_labels
            )
            np.add.at(base_lengths_64, idx[valid], tip_radii[valid])
        base_lengths = base_lengths_64.astype(np.float32, copy=False)

        # Median thickness per label. `scipy.ndimage.median` does the
        # per-label sort+middle in C; bit-identical to per-group
        # `np.median` for the float64 inputs here (verified on the
        # synthetic data the test pins, plus yeast 2D/3D fixtures).
        # Returns 0 for absent labels — defensively NaN-fill them. By
        # construction every label in `unique_labels = np.unique(L[L>0])`
        # has at least one matching voxel in `labels_branch_vox`, so
        # the NaN-fill is a no-op in practice.
        labels_branch_vox = L[tuple(branch_idxs_arr.T)]
        thicknesses = radii * 2.0
        median_thickness = ndi_cpu.median(
            thicknesses, labels=labels_branch_vox, index=unique_labels
        ).astype(np.float32, copy=False)
        if labels_branch_vox.size:
            counts = np.bincount(
                labels_branch_vox.astype(np.intp, copy=False),
                minlength=int(unique_labels_int.max()) + 1,
            )
            absent = counts[unique_labels_int] == 0
            if absent.any():
                median_thickness[absent] = np.nan

        # If thickness > length, swap (preserves "long axis" semantics
        # for blobby segments — gotcha pinned in
        # wiki/feature-extraction.md). Mirror the legacy `not np.isnan`
        # guard with `~np.isnan`, NOT `np.isfinite` (the latter would
        # also exclude +/-inf, semantic drift).
        swap = ~np.isnan(median_thickness) & (median_thickness > base_lengths)
        if np.any(swap):
            tmp = median_thickness[swap].copy()
            median_thickness[swap] = base_lengths[swap]
            base_lengths[swap] = tmp

        aspect_ratios = np.divide(
            base_lengths,
            median_thickness,
            out=np.full_like(base_lengths, np.nan),
            where=median_thickness != 0,
        )

        # Tortuosity: use first two tips per label if available; otherwise 1.
        # Vectorize via stable argsort — per-label position order matches
        # the legacy `tip_coords[tip_labels == lbl]` (boolean-masked view
        # preserves source order, and `tip_coords` is itself in
        # ascending-position order from `np.where`).
        tortuosity = np.ones(len(unique_labels), dtype=np.float32)
        if tip_labels.size:
            order = np.argsort(tip_labels, kind="stable")
            sorted_tip_labels = tip_labels[order]
            sorted_tip_coords = tip_coords[order]
            uniq_tip, starts, counts_tips = np.unique(
                sorted_tip_labels, return_index=True, return_counts=True
            )
            has_two = counts_tips >= 2
            if np.any(has_two):
                qualifying = uniq_tip[has_two]
                first_idx = starts[has_two]
                p0 = sorted_tip_coords[first_idx].astype(np.float64, copy=False)
                p1 = sorted_tip_coords[first_idx + 1].astype(np.float64, copy=False)
                spacing_arr = np.asarray(spacing, dtype=np.float64)
                # `(delta**2).sum(axis=1)` adds along axis 1 in ascending
                # index order — same per-axis sum-of-squares ordering
                # (Z+Y+X for 3D; Y+X for 2D) as the explicit
                # `dz*dz + dy*dy + dx*dx` in the legacy loop. The 2D vs
                # 3D split falls out of `tip_coords` shape, no `no_z`
                # branch needed.
                delta = (p0 - p1) * spacing_arr
                tip_dist = np.sqrt((delta * delta).sum(axis=1))
                target_idx = np.searchsorted(unique_labels, qualifying)
                safe = tip_dist > 0
                if np.any(safe):
                    tortuosity[target_idx[safe]] = (
                        base_lengths[target_idx[safe]] / tip_dist[safe]
                    ).astype(np.float32, copy=False)
                # Unsafe (tip_dist == 0): default 1.0 already in place.

        self.branch_tortuosity.append(tortuosity)
        self.branch_aspect_ratio.append(aspect_ratios)
        self.branch_thickness.append(median_thickness)
        self.branch_length.append(base_lengths)

        # regionprops-based morphology and centroids
        regions = regionprops(self.hierarchy.label_branches[t], spacing=self.hierarchy.spacing)
        areas = []
        axis_length_maj = []
        axis_length_min = []
        extent = []
        solidity = []
        reassigned_label = []
        z = []
        y = []
        x = []
        for region in regions:
            reassigned_label_region = np.nan
            if not self.hierarchy.im_info.no_t and self.hierarchy.im_branch_reassigned is not None:
                region_reassigned_labels = self.hierarchy.im_branch_reassigned[t][tuple(region.coords.T)]
                if region_reassigned_labels.size > 0:
                    reassigned_label_region = np.argmax(np.bincount(region_reassigned_labels))
            reassigned_label.append(reassigned_label_region)

            areas.append(region.area)
            try:
                maj_axis = region.major_axis_length
                min_axis = region.minor_axis_length
            except ValueError:
                maj_axis = np.nan
                min_axis = np.nan
            axis_length_maj.append(maj_axis)
            axis_length_min.append(min_axis)
            extent.append(region.extent)
            solidity.append(region.solidity)
            if not self.hierarchy.im_info.no_z:
                z.append(region.centroid[0])
                y.append(region.centroid[1])
                x.append(region.centroid[2])
            else:
                z.append(np.nan)
                y.append(region.centroid[0])
                x.append(region.centroid[1])

        self.branch_area.append(areas)
        self.branch_axis_length_maj.append(axis_length_maj)
        self.branch_axis_length_min.append(axis_length_min)
        self.branch_extent.append(extent)
        self.branch_solidity.append(solidity)
        self.reassigned_label.append(reassigned_label)
        self.z.append(z)
        self.y.append(y)
        self.x.append(x)

    def _run_frame(self, t):
        frame_branch_idxs = np.argwhere(self.hierarchy.im_skel[t] > 0)
        self.branch_idxs.append(frame_branch_idxs)

        frame_skel_branch_labels = self.hierarchy.im_skel[t][tuple(frame_branch_idxs.T)]

        if frame_skel_branch_labels.size == 0:
            # nothing to do
            self.time.append(np.array([], dtype=int))
            self.component_label.append(np.array([], dtype=int))
            self.branch_label.append(np.array([], dtype=int))
            self.image_name.append(np.array([], dtype=object))
            self.aggregate_voxel_metrics.append({})
            if not self.hierarchy.skip_nodes:
                self.aggregate_node_metrics.append({})
            self.branch_length.append([])
            self.branch_thickness.append([])
            self.branch_aspect_ratio.append([])
            self.branch_tortuosity.append([])
            self.branch_area.append([])
            self.branch_axis_length_maj.append([])
            self.branch_axis_length_min.append([])
            self.branch_extent.append([])
            self.branch_solidity.append([])
            self.reassigned_label.append([])
            self.z.append([])
            self.y.append([])
            self.x.append([])
            return

        unique_branch_labels = np.unique(frame_skel_branch_labels)
        unique_branch_labels = unique_branch_labels[unique_branch_labels > 0]
        num_branches = len(unique_branch_labels)

        frame_t = np.ones(num_branches, dtype=int) * t
        self.time.append(frame_t)

        if self.hierarchy.im_info.no_z:
            frame_branch_coords = np.zeros((num_branches, 2), dtype=int)
        else:
            frame_branch_coords = np.zeros((num_branches, 3), dtype=int)

        for idx, lbl in enumerate(unique_branch_labels):
            branch_voxels = frame_branch_idxs[frame_skel_branch_labels == lbl]
            # there should always be at least one voxel for each label here
            if len(branch_voxels) == 0:
                continue
            frame_branch_coords[idx] = branch_voxels[0]

        # Component labels for each branch
        frame_component_label = self.hierarchy.label_components[t][tuple(frame_branch_coords.T)]
        self.component_label.append(frame_component_label)

        # Store the actual branch labels (sorted, unique)
        self.branch_label.append(unique_branch_labels.astype(int))

        im_name = (
            np.ones(num_branches, dtype=object)
            * self.hierarchy.im_info.file_info.filename_no_ext
        )
        self.image_name.append(im_name)

        self._get_aggregate_stats(t)
        self._get_branch_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            if self.hierarchy.viewer is not None:
                self.hierarchy.viewer.status = (
                    f"Extracting branch features. Frame: {t + 1} of {self.hierarchy.num_t}."
                )
            self._run_frame(t)


class Components:
    """
    Component (organelle)-level features.
    """

    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.component_label = []
        self.aggregate_voxel_metrics = []
        self.aggregate_node_metrics = []
        self.aggregate_branch_metrics = []

        self.z = []
        self.y = []
        self.x = []
        self.organelle_area = []
        self.organelle_axis_length_maj = []
        self.organelle_axis_length_min = []
        self.organelle_extent = []
        self.organelle_solidity = []
        self.reassigned_label = []

        self.image_name = []

        self.stats_to_aggregate = [
            "organelle_area",
            "organelle_axis_length_maj",
            "organelle_axis_length_min",
            "organelle_extent",
            "organelle_solidity",
            "reassigned_label",
        ]

        self.features_to_save = self.stats_to_aggregate + ["x", "y", "z"]

    def _get_aggregate_stats(self, t):
        voxel_labels = self.hierarchy.voxels.component_labels[t]
        # All three groupings key off `np.unique(voxel_labels)` — kept
        # this way so the agg lists are 1:1 across vox/node/branch even
        # when a component has no nodes or branches at this t.
        all_unique, all_vox_groups = _sorted_label_groups(voxel_labels)
        keep_mask = all_unique != 0
        unique_keys = all_unique[keep_mask]
        grouped_vox_idxs = [g for u, g in zip(all_unique, all_vox_groups) if u != 0]
        vox_agg = aggregate_stats_for_class(
            self.hierarchy.voxels, t, grouped_vox_idxs, low_memory=self.hierarchy.low_memory
        )
        self.aggregate_voxel_metrics.append(vox_agg)

        if not self.hierarchy.skip_nodes:
            node_labels = self.hierarchy.nodes.component_label[t]
            grouped_node_idxs = _group_indices_for_keys(node_labels, unique_keys)
            node_agg = aggregate_stats_for_class(
                self.hierarchy.nodes, t, grouped_node_idxs, low_memory=self.hierarchy.low_memory
            )
            self.aggregate_node_metrics.append(node_agg)

        branch_labels = self.hierarchy.branches.component_label[t]
        grouped_branch_idxs = _group_indices_for_keys(branch_labels, unique_keys)
        branch_agg = aggregate_stats_for_class(
            self.hierarchy.branches, t, grouped_branch_idxs, low_memory=self.hierarchy.low_memory
        )
        self.aggregate_branch_metrics.append(branch_agg)

    def _get_component_stats(self, t):
        regions = regionprops(self.hierarchy.label_components[t], spacing=self.hierarchy.spacing)
        areas = []
        axis_length_maj = []
        axis_length_min = []
        extent = []
        solidity = []
        reassigned_label = []
        z = []
        y = []
        x = []
        for region in regions:
            reassigned_label_region = np.nan
            if not self.hierarchy.im_info.no_t and self.hierarchy.im_obj_reassigned is not None:
                region_reassigned_labels = self.hierarchy.im_obj_reassigned[t][tuple(region.coords.T)]
                if region_reassigned_labels.size > 0:
                    reassigned_label_region = np.argmax(np.bincount(region_reassigned_labels))
            reassigned_label.append(reassigned_label_region)
            areas.append(region.area)
            try:
                maj_axis = region.major_axis_length
                min_axis = region.minor_axis_length
            except ValueError:
                maj_axis = np.nan
                min_axis = np.nan
            axis_length_maj.append(maj_axis)
            axis_length_min.append(min_axis)
            extent.append(region.extent)
            solidity.append(region.solidity)
            if not self.hierarchy.im_info.no_z:
                z.append(region.centroid[0])
                y.append(region.centroid[1])
                x.append(region.centroid[2])
            else:
                z.append(np.nan)
                y.append(region.centroid[0])
                x.append(region.centroid[1])
        self.organelle_area.append(areas)
        self.organelle_axis_length_maj.append(axis_length_maj)
        self.organelle_axis_length_min.append(axis_length_min)
        self.organelle_extent.append(extent)
        self.organelle_solidity.append(solidity)
        self.reassigned_label.append(reassigned_label)
        self.z.append(z)
        self.y.append(y)
        self.x.append(x)

    def _run_frame(self, t):
        component_labels_t = self.hierarchy.label_components[t]
        mask = component_labels_t > 0
        if not np.any(mask):
            self.component_label.append(np.array([], dtype=int))
            self.time.append(np.array([], dtype=int))
            self.image_name.append(np.array([], dtype=object))
            self.aggregate_voxel_metrics.append({})
            if not self.hierarchy.skip_nodes:
                self.aggregate_node_metrics.append({})
            self.aggregate_branch_metrics.append({})
            self.organelle_area.append([])
            self.organelle_axis_length_maj.append([])
            self.organelle_axis_length_min.append([])
            self.organelle_extent.append([])
            self.organelle_solidity.append([])
            self.reassigned_label.append([])
            self.z.append([])
            self.y.append([])
            self.x.append([])
            return

        frame_component_labels = np.unique(component_labels_t[mask])
        self.component_label.append(frame_component_labels)
        num_components = len(frame_component_labels)

        frame_t = np.ones(num_components, dtype=int) * t
        self.time.append(frame_t)

        im_name = (
            np.ones(num_components, dtype=object)
            * self.hierarchy.im_info.file_info.filename_no_ext
        )
        self.image_name.append(im_name)

        self._get_aggregate_stats(t)
        self._get_component_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            if self.hierarchy.viewer is not None:
                self.hierarchy.viewer.status = (
                    f"Extracting organelle features. Frame: {t + 1} of {self.hierarchy.num_t}."
                )
            self._run_frame(t)


class Image:
    """
    Image-level aggregated features.
    """

    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.image_name = []
        self.aggregate_voxel_metrics = []
        self.aggregate_node_metrics = []
        self.aggregate_branch_metrics = []
        self.aggregate_component_metrics = []
        self.stats_to_aggregate = []
        self.features_to_save = []

    def _get_aggregate_stats(self, t):
        # Voxels: one group containing all voxels in frame t
        voxel_agg = aggregate_stats_for_class(
            self.hierarchy.voxels,
            t,
            [np.arange(len(self.hierarchy.voxels.coords[t]), dtype=int)],
            low_memory=self.hierarchy.low_memory,
        )
        self.aggregate_voxel_metrics.append(voxel_agg)

        # Nodes: one group containing all nodes (if any)
        if not self.hierarchy.skip_nodes:
            node_agg = aggregate_stats_for_class(
                self.hierarchy.nodes,
                t,
                [np.arange(len(self.hierarchy.nodes.nodes[t]), dtype=int)],
                low_memory=self.hierarchy.low_memory,
            )
            self.aggregate_node_metrics.append(node_agg)

        # Branches: one group containing all branches
        n_branches = len(self.hierarchy.branches.branch_length[t])
        branch_idxs = np.arange(n_branches, dtype=int)
        branch_agg = aggregate_stats_for_class(
            self.hierarchy.branches,
            t,
            [branch_idxs],
            low_memory=self.hierarchy.low_memory,
        )
        self.aggregate_branch_metrics.append(branch_agg)

        # Components: one group containing all components
        n_components = len(self.hierarchy.components.organelle_area[t])
        comp_idxs = np.arange(n_components, dtype=int)
        component_agg = aggregate_stats_for_class(
            self.hierarchy.components,
            t,
            [comp_idxs],
            low_memory=self.hierarchy.low_memory,
        )
        self.aggregate_component_metrics.append(component_agg)

    def _run_frame(self, t):
        self.time.append(t)
        self.image_name.append(self.hierarchy.im_info.file_info.filename_no_ext)
        self._get_aggregate_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            if self.hierarchy.viewer is not None:
                self.hierarchy.viewer.status = (
                    f"Extracting image features. Frame: {t + 1} of {self.hierarchy.num_t}."
                )
            self._run_frame(t)
