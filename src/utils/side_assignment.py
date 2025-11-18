"""
Side Assignment Utilities for Bi-Directional Detection
========================================================

Assigns vehicle tracks to LEFT or RIGHT carriageway based on:
1. Centroid X threshold (simple)
2. K-means clustering (robust)
"""

import numpy as np
from sklearn.cluster import KMeans


def assign_side_by_threshold(centroid_x, threshold):
    """
    Assign track to LEFT or RIGHT based on centroid X coordinate.
    
    Args:
        centroid_x: X coordinate of track centroid
        threshold: Boundary value (typically frame_width / 2)
    
    Returns:
        "LEFT" if centroid_x < threshold, else "RIGHT"
    """
    return "LEFT" if centroid_x < threshold else "RIGHT"


def assign_sides_by_kmeans(track_centroids, n_clusters=2):
    """
    Assign tracks to LEFT or RIGHT using K-means clustering.
    
    More robust than threshold method when:
    - Road has perspective distortion
    - Divider line not at exact center
    - Camera angle is oblique
    
    Args:
        track_centroids: List of (x, y) centroids for all tracks
        n_clusters: Number of sides (default=2 for LEFT/RIGHT)
    
    Returns:
        dict: {track_idx: "LEFT" or "RIGHT"}
    """
    if len(track_centroids) < n_clusters:
        # Not enough tracks - fall back to simple midpoint
        midpoint = np.mean([c[0] for c in track_centroids])
        return {i: "LEFT" if c[0] < midpoint else "RIGHT" 
                for i, c in enumerate(track_centroids)}
    
    # Extract X coordinates only (clustering based on lateral position)
    X = np.array([[c[0]] for c in track_centroids])
    
    # Cluster into 2 groups
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Determine which cluster is LEFT (lower X) vs RIGHT (higher X)
    cluster_centers = kmeans.cluster_centers_.flatten()
    left_cluster_id = np.argmin(cluster_centers)
    right_cluster_id = np.argmax(cluster_centers)
    
    # Assign sides
    assignments = {}
    for i, label in enumerate(labels):
        if label == left_cluster_id:
            assignments[i] = "LEFT"
        elif label == right_cluster_id:
            assignments[i] = "RIGHT"
        else:
            # Shouldn't happen with n_clusters=2, but handle gracefully
            assignments[i] = "LEFT"
    
    return assignments


def update_track_sides(tracks, side_assignment_config):
    """
    Update all tracks with side assignments.
    
    Args:
        tracks: List of track dictionaries with 'centroid' field
        side_assignment_config: Config dict with 'method' and parameters
    
    Returns:
        Updated tracks with 'side' field added
    """
    method = side_assignment_config.get('method', 'centroid_x_threshold')
    
    if method == 'centroid_x_threshold':
        threshold = side_assignment_config['threshold']
        for track in tracks:
            cx = track['centroid'][0]
            track['side'] = assign_side_by_threshold(cx, threshold)
    
    elif method == 'kmeans':
        # Collect all centroids
        centroids = [track['centroid'] for track in tracks]
        
        # Get assignments
        assignments = assign_sides_by_kmeans(centroids)
        
        # Update tracks
        for i, track in enumerate(tracks):
            track['side'] = assignments[i]
    
    else:
        raise ValueError(f"Unknown side assignment method: {method}")
    
    return tracks


def get_side_statistics(tracks):
    """
    Get statistics about LEFT/RIGHT distribution.
    
    Args:
        tracks: List of tracks with 'side' field
    
    Returns:
        dict: Statistics about side distribution
    """
    sides = [t.get('side', 'UNKNOWN') for t in tracks]
    
    left_count = sides.count('LEFT')
    right_count = sides.count('RIGHT')
    unknown_count = sides.count('UNKNOWN')
    
    return {
        'left_count': left_count,
        'right_count': right_count,
        'unknown_count': unknown_count,
        'total': len(tracks),
        'left_pct': (left_count / len(tracks) * 100) if tracks else 0,
        'right_pct': (right_count / len(tracks) * 100) if tracks else 0
    }


def visualize_side_assignment(frame, tracks, show_labels=True):
    """
    Visualize track side assignments on frame.
    
    Args:
        frame: Image frame (numpy array)
        tracks: List of tracks with 'side', 'centroid', 'bbox'
        show_labels: Whether to show LEFT/RIGHT labels
    
    Returns:
        Annotated frame
    """
    import cv2
    
    annotated = frame.copy()
    
    for track in tracks:
        side = track.get('side', 'UNKNOWN')
        centroid = track['centroid']
        
        # Color based on side
        if side == 'LEFT':
            color = (0, 255, 0)  # Green
        elif side == 'RIGHT':
            color = (255, 0, 0)  # Blue
        else:
            color = (128, 128, 128)  # Gray
        
        # Draw centroid
        cv2.circle(annotated, (int(centroid[0]), int(centroid[1])), 5, color, -1)
        
        # Draw bounding box if available
        if 'bbox' in track:
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label
        if show_labels:
            label = f"{track.get('id', '?')}-{side}"
            cv2.putText(annotated, label, 
                       (int(centroid[0]) - 20, int(centroid[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated


if __name__ == "__main__":
    # Test side assignment
    print("Testing side assignment utilities...")
    
    # Test threshold method
    print("\n1. Threshold Method (threshold=480)")
    test_centroids = [(200, 300), (600, 300), (100, 400), (800, 400)]
    for cx, cy in test_centroids:
        side = assign_side_by_threshold(cx, 480)
        print(f"   Centroid ({cx}, {cy}) -> {side}")
    
    # Test K-means method
    print("\n2. K-means Method")
    assignments = assign_sides_by_kmeans(test_centroids)
    for i, (cx, cy) in enumerate(test_centroids):
        print(f"   Centroid ({cx}, {cy}) -> {assignments[i]}")
    
    print("\n✅ Side assignment utilities ready!")
