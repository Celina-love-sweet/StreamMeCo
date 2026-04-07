
import os
import math
import pickle
import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from mmagent import *
import sys
import mmagent.videograph as vg
sys.modules["videograph"] = vg  


TEXT_NODE_TYPES = {"episodic", "semantic"}
MEDIA_NODE_TYPES = {"img", "voice"}
FIRST_CLASS_CLUSTER_RATIO = 0.05
FIRST_CLASS_KEEP_RATIO = 0.7
SECOND_CLASS_KEEP_RATIO = 0.7


def _node_embedding(node) -> Optional[np.ndarray]:
    """Return a single embedding vector for a node."""
    embeddings = getattr(node, "embeddings", None)
    if not embeddings:
        return None
    emb_array = np.asarray(embeddings, dtype=np.float32)
    if emb_array.ndim == 1:
        return emb_array
    return emb_array.mean(axis=0)


def _ensure_directory(path: str) -> None:
    """Create directory tree if it does not exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def _collect_graph_files(src_root: str, dst_root: str) -> List[Tuple[str, str]]:
    """Collect (source, destination) pairs for every pickle file that needs compression."""
    pairs: List[Tuple[str, str]] = []
    if os.path.isfile(src_root):
        dst_path = os.path.join(dst_root, os.path.basename(src_root))
        pairs.append((src_root, dst_path))
    elif os.path.isdir(src_root):
        for root, _, files in os.walk(src_root):
            for filename in files:
                if not filename.endswith(".pkl"):
                    continue
                src_path = os.path.join(root, filename)
                rel_path = os.path.relpath(src_path, src_root)
                dst_path = os.path.join(dst_root, rel_path)
                pairs.append((src_path, dst_path))
    else:
        raise FileNotFoundError(f"Cannot find memory graph path: {src_root}")
    pairs.sort(key=lambda item: item[0])
    return pairs


def _assign_rank_weights(scores: Dict[int, float], invert: bool = False) -> Dict[int, float]:
    """Normalize scores to [0, 1] and optionally invert for dissimilarity."""
    if not scores:
        return {}
    values = list(scores.values())
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        normalized = {k: 1.0 for k in scores}
    else:
        normalized = {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
    if invert:
        return {k: 1.0 - v for k, v in normalized.items()}
    return normalized


def _is_equivalence_text_node(node) -> bool:
    if not node or getattr(node, "type", None) not in TEXT_NODE_TYPES:
        return False
    metadata = getattr(node, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    contents = metadata.get("contents", [])
    if not isinstance(contents, list):
        contents = [contents]
    for item in contents:
        text = str(item).strip().lower()
        if text.startswith("equivalence"):
            return True
    return False


def classify_text_nodes(graph) -> Tuple[List[int], List[int], np.ndarray]:
    """Split text nodes into (no-media, with-media) groups and build the weight matrix."""
    media_nodes = [node_id for node_id, node in graph.nodes.items() if node.type in MEDIA_NODE_TYPES]
    media_index = {node_id: idx for idx, node_id in enumerate(media_nodes)}

    ordered_text_ids: List[int] = []
    seen: Set[int] = set()
    for node_id in getattr(graph, "text_nodes", []):
        node = graph.nodes.get(node_id)
        if not node or node.type not in TEXT_NODE_TYPES:
            continue
        ordered_text_ids.append(node_id)
        seen.add(node_id)
    for node_id, node in graph.nodes.items():
        if node.type in TEXT_NODE_TYPES and node_id not in seen:
            ordered_text_ids.append(node_id)

    text_to_media_weights: Dict[int, Dict[int, float]] = defaultdict(dict)
    processed_pairs: Set[Tuple[int, int]] = set()
    for (src, dst), weight in graph.edges.items():
        pair = tuple(sorted((src, dst)))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)
        src_node = graph.nodes.get(src)
        dst_node = graph.nodes.get(dst)
        if not src_node or not dst_node:
            continue
        if src_node.type in MEDIA_NODE_TYPES and dst_node.type in TEXT_NODE_TYPES:
            text_to_media_weights[dst][src] = weight
        elif dst_node.type in MEDIA_NODE_TYPES and src_node.type in TEXT_NODE_TYPES:
            text_to_media_weights[src][dst] = weight

    text_without_media: List[int] = []
    text_with_media: List[int] = []
    for node_id in ordered_text_ids:
        if _is_equivalence_text_node(graph.nodes.get(node_id)):
            continue
        if node_id in text_to_media_weights:
            text_with_media.append(node_id)
        else:
            text_without_media.append(node_id)

    matrix = np.zeros((len(media_nodes), len(text_with_media)), dtype=np.float32)
    if media_nodes and text_with_media:
        text_index = {node_id: idx for idx, node_id in enumerate(text_with_media)}
        for text_id, media_weights in text_to_media_weights.items():
            text_idx = text_index.get(text_id)
            if text_idx is None:
                continue
            for media_id, weight in media_weights.items():
                media_idx = media_index.get(media_id)
                if media_idx is None:
                    continue
                matrix[media_idx, text_idx] = weight

    return text_without_media, text_with_media, matrix


def _group_first_class_nodes(graph, node_ids: Sequence[int]) -> Dict[object, List[int]]:
    """Group first-class text nodes by clip id."""
    groups: Dict[object, List[int]] = defaultdict(list)
    first_class_set = set(node_ids)
    assigned: Set[int] = set()
    for clip_id, ids in getattr(graph, "text_nodes_by_clip", {}).items():
        clip_nodes = [node_id for node_id in ids if node_id in first_class_set]
        if clip_nodes:
            groups[clip_id].extend(clip_nodes)
            assigned.update(clip_nodes)
    for node_id in node_ids:
        if node_id in assigned:
            continue
        node = graph.nodes.get(node_id)
        clip_id = None
        if node and isinstance(getattr(node, "metadata", None), dict):
            clip_id = node.metadata.get("timestamp")
        groups[clip_id if clip_id is not None else "unknown"].append(node_id)
    return groups


def _adjust_keep_counts(
    cluster_members: Dict[int, List[int]],
    keep_ratio: float,
    target_total: int,
) -> Dict[int, int]:
    """Distribute keep counts per cluster so totals hit the desired ratio."""
    keep_counts: Dict[int, int] = {}
    raw_keeps: Dict[int, float] = {}
    base_total = 0
    for label, members in cluster_members.items():
        size = len(members)
        raw_keep = size * keep_ratio
        base_keep = max(1, min(size, int(math.ceil(raw_keep))))
        keep_counts[label] = base_keep
        raw_keeps[label] = raw_keep
        base_total += base_keep

    diff = target_total - base_total
    if diff > 0:
        # Allocate remaining quota to clusters with the largest fractional part, respecting capacity.
        order = sorted(
            cluster_members.keys(),
            key=lambda lbl: (raw_keeps[lbl] - math.floor(raw_keeps[lbl]), len(cluster_members[lbl])),
            reverse=True,
        )
        while diff > 0:
            progressed = False
            for label in order:
                capacity = len(cluster_members[label])
                if keep_counts[label] < capacity:
                    keep_counts[label] += 1
                    diff -= 1
                    progressed = True
                    if diff == 0:
                        break
            if not progressed:
                break
    elif diff < 0:
        # Trim extras from clusters with the smallest fractional part while keeping at least one element.
        order = sorted(
            cluster_members.keys(),
            key=lambda lbl: (raw_keeps[lbl] - math.floor(raw_keeps[lbl]), len(cluster_members[lbl])),
        )
        while diff < 0:
            progressed = False
            for label in order:
                if keep_counts[label] > 1:
                    keep_counts[label] -= 1
                    diff += 1
                    progressed = True
                    if diff == 0:
                        break
            if not progressed:
                break
    return keep_counts

def _greedy_farthest_selection(
    member_indices: List[int],
    data: np.ndarray,
    center: np.ndarray,
    keep_count: int,
) -> Set[int]:
    """Pick center-nearest, then iteratively farthest-from-selected within a cluster."""
    if keep_count >= len(member_indices):
        return set(member_indices)
    # First pick: closest to the cluster center.
    first_idx = max(member_indices, key=lambda ridx: float(np.dot(data[ridx], center)))
    selected: List[int] = [first_idx]
    while len(selected) < keep_count:
        best_candidate = None
        best_distance = -1.0
        for idx in member_indices:
            if idx in selected:
                continue
            # Cosine distance because data is normalized.
            nearest = min(1.0 - float(np.dot(data[idx], data[sidx])) for sidx in selected)
            if nearest > best_distance:
                best_distance = nearest
                best_candidate = idx
        if best_candidate is None:
            break
        selected.append(best_candidate)
    return set(selected)

 
def _select_via_spherical_kmeans(
    node_ids: Sequence[int],
    embeddings: Sequence[np.ndarray],
    cluster_ratio: float,
    keep_ratio: float,
) -> Set[int]:
    """Run spherical K-Means, keep 70% overall, distribute by cluster size with farthest-first inside each cluster."""
    if not node_ids:
        return set()
    if len(node_ids) == 1:
        return {node_ids[0]}
    data = normalize(np.stack(embeddings), axis=1)
    n_clusters = max(1, math.ceil(len(node_ids) * cluster_ratio))
    n_clusters = min(n_clusters, len(node_ids))
    if n_clusters == 1:
        target_total = max(1, min(len(node_ids), int(math.ceil(len(node_ids) * keep_ratio))))
        center = normalize(data.mean(axis=0).reshape(1, -1))[0]
        selected_indices = _greedy_farthest_selection(list(range(len(node_ids))), data, center, target_total)
        return {node_ids[idx] for idx in selected_indices}
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(data)
    centers = normalize(model.cluster_centers_, axis=1)
    cluster_members: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_members[label].append(idx)

    target_total = max(1, min(len(node_ids), int(math.ceil(len(node_ids) * keep_ratio))))
    keep_counts = _adjust_keep_counts(cluster_members, keep_ratio, target_total)

    kept_indices: Set[int] = set()
    for label, member_indices in cluster_members.items():
        count = keep_counts.get(label, 0)
        if count <= 0:
            continue
        center = centers[label]
        selected = _greedy_farthest_selection(member_indices, data, center, count)
        kept_indices.update(selected)
    # Ensure we do not exceed total target due to rounding in rare cases.
    if len(kept_indices) > target_total:
        kept_indices = set(sorted(kept_indices)[: target_total])
    return {node_ids[idx] for idx in kept_indices}


def compress_first_class_nodes(
    graph,
    node_ids: Sequence[int],
    cluster_ratio: float = FIRST_CLASS_CLUSTER_RATIO,
    keep_ratio: float = FIRST_CLASS_KEEP_RATIO,
) -> Tuple[List[int], List[int]]:
    """Compress first-class nodes clip-by-clip via spherical K-Means."""
    if not node_ids:
        return [], []
    kept: Set[int] = set()
    groups = _group_first_class_nodes(graph, node_ids)
    for clip_id, clip_nodes in groups.items():
        candidate_ids: List[int] = []
        candidate_embeddings: List[np.ndarray] = []
        for node_id in clip_nodes:
            embedding = _node_embedding(graph.nodes.get(node_id))
            if embedding is None:
                kept.add(node_id)
                continue
            candidate_ids.append(node_id)
            candidate_embeddings.append(np.asarray(embedding, dtype=np.float32))
        kept.update(
            _select_via_spherical_kmeans(
                candidate_ids,
                candidate_embeddings,
                cluster_ratio=cluster_ratio,
                keep_ratio=keep_ratio,
            )
        )
    first_class_set = set(node_ids)
    removed = first_class_set - kept
    kept_ordered = [node_id for node_id in node_ids if node_id in kept]
    removed_ordered = [node_id for node_id in node_ids if node_id in removed]
    return kept_ordered, removed_ordered


def compress_second_class_nodes(
    graph,
    node_ids: Sequence[int],
    weight_matrix: Optional[np.ndarray],
    alpha: float = 0.1,
    keep_ratio: float = SECOND_CLASS_KEEP_RATIO,
) -> Tuple[List[int], List[int]]:
    """Compress second-class nodes by combining importance and dissimilarity rankings."""
    if not node_ids:
        return [], []
    alpha = max(0.0, min(1.0, alpha))
    n = len(node_ids)
    original_order = {node_id: idx for idx, node_id in enumerate(node_ids)}
    matrix = weight_matrix if weight_matrix is not None else np.zeros((0, n), dtype=np.float32)
    if matrix.size:
        importance_scores = matrix.sum(axis=0)
    else:
        importance_scores = np.zeros(n, dtype=np.float32)   
    text_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    importance_scores_map: Dict[int, float] = {
        node_id: float(importance_scores[text_index[node_id]]) for node_id in node_ids
    }
    importance_weights = _assign_rank_weights(importance_scores_map, invert=False)

    raw_embeddings: List[Optional[np.ndarray]] = []
    embedding_dim: Optional[int] = None
    for node_id in node_ids:
        embedding = _node_embedding(graph.nodes.get(node_id))
        raw_embeddings.append(embedding)
        if embedding is not None and embedding_dim is None:
            embedding_dim = embedding.shape[0]
    if embedding_dim is None:
        similarity_scores = np.zeros(len(node_ids), dtype=np.float32)
    else:
        processed_vectors: List[np.ndarray] = []
        for embedding in raw_embeddings:
            if embedding is None:
                processed_vectors.append(np.zeros(embedding_dim, dtype=np.float32))
                continue
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.shape[0] > embedding_dim:
                vector = vector[:embedding_dim]
            elif vector.shape[0] < embedding_dim:
                padded = np.zeros(embedding_dim, dtype=np.float32)
                padded[: vector.shape[0]] = vector
                vector = padded
            processed_vectors.append(vector)
        data = normalize(np.stack(processed_vectors), axis=1)
        similarity_matrix = data @ data.T
        similarity_scores = similarity_matrix.sum(axis=1)
    similarity_scores_map: Dict[int, float] = {
        node_ids[idx]: float(similarity_scores[idx]) for idx in range(len(node_ids))
    }
    similarity_weights = _assign_rank_weights(similarity_scores_map, invert=True)

    combined_scores: Dict[int, float] = {}
    for node_id in node_ids:
        score_a = importance_weights.get(node_id, 0)
        score_b = similarity_weights.get(node_id, 0)
        combined_scores[node_id] = alpha * score_a + (1 - alpha) * score_b
    ordered_by_score = sorted(node_ids, key=lambda node_id: (-combined_scores[node_id], original_order[node_id]))
    keep_count = max(1, math.ceil(len(node_ids) * keep_ratio))
    kept_set = set(ordered_by_score[:keep_count])
    kept = [node_id for node_id in node_ids if node_id in kept_set]
    removed = [node_id for node_id in node_ids if node_id not in kept_set]
    return kept, removed


def remove_text_nodes_from_graph(graph, node_ids: Iterable[int]):
    """Remove nodes and all related references from the graph."""
    removal_set = set(node_ids)
    if not removal_set:
        return graph
    for node_id in removal_set:
        graph.nodes.pop(node_id, None)
    if hasattr(graph, "text_nodes"):
        graph.text_nodes = [node_id for node_id in graph.text_nodes if node_id not in removal_set]
    if hasattr(graph, "edges"):
        for edge in list(graph.edges.keys()):
            if edge[0] in removal_set or edge[1] in removal_set:
                graph.edges.pop(edge, None)
    if hasattr(graph, "text_nodes_by_clip"):
        for clip_id in list(graph.text_nodes_by_clip.keys()):
            filtered = [node_id for node_id in graph.text_nodes_by_clip[clip_id] if node_id not in removal_set]
            if filtered:
                graph.text_nodes_by_clip[clip_id] = filtered
            else:
                del graph.text_nodes_by_clip[clip_id]
    if hasattr(graph, "event_sequence_by_clip"):
        for clip_id in list(graph.event_sequence_by_clip.keys()):
            filtered = [node_id for node_id in graph.event_sequence_by_clip[clip_id] if node_id not in removal_set]
            if filtered:
                graph.event_sequence_by_clip[clip_id] = filtered
            else:
                del graph.event_sequence_by_clip[clip_id]
    return graph


def compress_graph(graph, alpha: float = 0.1):
    """Full compression pipeline for a single VideoGraph object."""
    text_without_media, text_with_media, weight_matrix = classify_text_nodes(graph)
    first_kept, first_removed = compress_first_class_nodes(graph, text_without_media)
    second_kept, second_removed = compress_second_class_nodes(
        graph,
        text_with_media,
        weight_matrix,
        alpha=alpha,
    )
    nodes_to_remove = first_removed + second_removed
    updated_graph = remove_text_nodes_from_graph(graph, nodes_to_remove)
    if hasattr(updated_graph, "refresh_equivalences"):
        updated_graph.refresh_equivalences()
    summary = {
        "first_total": len(text_without_media),
        "first_kept": len(first_kept),
        "first_removed": len(first_removed),
        "second_total": len(text_with_media),
        "second_kept": len(second_kept),
        "second_removed": len(second_removed),
        "total_removed": len(nodes_to_remove),
    }
    return updated_graph, summary


def process_graph_file(input_path: str, output_path: str, alpha: float) -> Dict[str, int]:
    """Load, compress, and store one graph file."""
    with open(input_path, "rb") as f:
        graph = pickle.load(f)
    if hasattr(graph, "refresh_equivalences"):
        graph.refresh_equivalences()
    graph, summary = compress_graph(graph, alpha=alpha)
    _ensure_directory(os.path.dirname(output_path))
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, default="data/memory_graphs/robot")
    parser.add_argument("--compressed_mem_path", type=str, default="data/memory_graphs/streammeco_robot70")
    parser.add_argument("--alpha", type=float, default=0.1, help="Fusion coefficient for importance vs. dissimilarity rankings.")
    return parser.parse_args()


def main():
    args = parse_args()

    file_pairs = _collect_graph_files(args.mem_path, args.compressed_mem_path)
    if not file_pairs:
        print("No memory graph files found to compress.")
        return
    mem_is_dir = os.path.isdir(args.mem_path)
    compressed_is_dir = os.path.isdir(args.compressed_mem_path)
    for src_path, dst_path in file_pairs:
        summary = process_graph_file(src_path, dst_path, alpha=args.alpha)
        src_label = (
            os.path.relpath(src_path, args.mem_path)
            if mem_is_dir
            else os.path.basename(src_path)
        )
        dst_label = (
            os.path.relpath(dst_path, args.compressed_mem_path)
            if compressed_is_dir
            else os.path.basename(dst_path)
        )
        print(f"Compressed {src_label} -> {dst_label} | ")


if __name__ == "__main__":
    main()
