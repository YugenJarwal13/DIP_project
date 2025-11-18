"""
Dynamic Direction Vector Computation for Bi-Directional Detection
===================================================================

Computes expected movement direction vectors from actual track data,
eliminating the need for manual direction specification.

Key Idea:
---------
Normal-flow tracks exhibit consistent movement direction.
Average their displacement vectors to get expected direction.
"""

import numpy as np
from collections import defaultdict


def compute_displacement_vectors(track_history, window=6):
    """
    Compute displacement vectors for a track over a time window.
    
    Args:
        track_history: List of (frame_num, centroid) tuples
        window: Number of frames to compute displacement over
    
    Returns:
        List of (frame_num, displacement_vector) tuples
    """
    if len(track_history) < window:
        return []
    
    displacements = []
    for i in range(window, len(track_history)):
        start_centroid = np.array(track_history[i - window][1])
        end_centroid = np.array(track_history[i][1])
        
        displacement = end_centroid - start_centroid
        magnitude = np.linalg.norm(displacement)
        
        if magnitude > 0:
            displacements.append((track_history[i][0], displacement))
    
    return displacements


def identify_normal_flow_tracks(tracks_data, zones, flow_mapping, side=None):
    """
    Identify tracks that follow normal flow pattern.
    
    Args:
        tracks_data: Dict of track_id -> list of frame data
        zones: Dict of zone_name -> Polygon
        flow_mapping: Dict of "sequence" -> "normal" or "wrong_way"
        side: "LEFT" or "RIGHT" (for bidirectional), or None (unidirectional)
    
    Returns:
        List of track_ids that follow normal flow
    """
    from shapely.geometry import Point
    
    normal_tracks = []
    
    # Determine normal sequence
    normal_sequence = None
    for seq, label in flow_mapping.items():
        if label == "normal":
            normal_sequence = seq.replace('-', '')  # "A-B-C" -> "ABC"
            break
    
    if not normal_sequence:
        return []
    
    # Check each track
    for track_id, frames in tracks_data.items():
        # Filter by side if specified
        if side:
            track_side = frames[0].get('side')
            if track_side != side:
                continue
        
        # Build zone sequence
        zone_sequence = []
        for frame_data in frames:
            centroid = frame_data['centroid']
            
            # Check which zone centroid is in
            for zone_name, zone_poly in zones.items():
                if zone_poly is None:
                    continue
                
                # For bidirectional, only check zones matching the side
                if side and not zone_name.endswith(f"_{side[0]}"):
                    continue
                
                try:
                    if zone_poly.contains(Point(centroid)):
                        # Extract base zone letter (A, B, C)
                        base_zone = zone_name[0] if '_' not in zone_name else zone_name.split('_')[0]
                        if not zone_sequence or zone_sequence[-1] != base_zone:
                            zone_sequence.append(base_zone)
                        break
                except:
                    pass
        
        # Check if matches normal sequence
        sequence_str = ''.join(zone_sequence)
        if normal_sequence in sequence_str:
            normal_tracks.append(track_id)
    
    return normal_tracks


def compute_direction_vector_from_tracks(tracks_data, normal_track_ids, window=6, min_displacement=1.0):
    """
    Compute average direction vector from normal-flow tracks.
    
    Args:
        tracks_data: Dict of track_id -> list of frame data
        normal_track_ids: List of track IDs following normal flow
        window: Window size for displacement calculation
        min_displacement: Minimum displacement magnitude to include
    
    Returns:
        Normalized direction vector [dx, dy] or None if insufficient data
    """
    all_displacements = []
    
    for track_id in normal_track_ids:
        if track_id not in tracks_data:
            continue
        
        frames = tracks_data[track_id]
        
        # Build centroid history
        centroids = [(f['frame_num'] if 'frame_num' in f else i, f['centroid']) 
                     for i, f in enumerate(frames)]
        
        # Compute displacements
        displacements = compute_displacement_vectors(centroids, window)
        
        # Filter by magnitude
        for frame_num, disp in displacements:
            mag = np.linalg.norm(disp)
            if mag >= min_displacement:
                all_displacements.append(disp)
    
    if len(all_displacements) < 5:
        print(f"⚠️  Warning: Only {len(all_displacements)} valid displacements for direction computation")
        return None
    
    # Average all displacement vectors
    mean_displacement = np.mean(all_displacements, axis=0)
    
    # Normalize
    magnitude = np.linalg.norm(mean_displacement)
    if magnitude < 0.1:
        print(f"⚠️  Warning: Mean displacement magnitude too small: {magnitude}")
        return None
    
    normalized = mean_displacement / magnitude
    
    return normalized.tolist()


def compute_bidirectional_direction_vectors(tracks_data, zones, flow_mappings, 
                                           side_assignment_config, 
                                           window=6, min_displacement=1.0):
    """
    Compute direction vectors for both LEFT and RIGHT carriageways.
    
    Args:
        tracks_data: Dict of track_id -> list of frame data
        zones: Dict of zone_name -> Polygon (with A_L, B_L, C_L, A_R, B_R, C_R)
        flow_mappings: Dict with 'LEFT' and 'RIGHT' flow mappings
        side_assignment_config: Config for side assignment
        window: Window for displacement calculation
        min_displacement: Minimum displacement magnitude
    
    Returns:
        dict: {
            'LEFT': {'normal': [dx, dy], 'wrong': [dx, dy]},
            'RIGHT': {'normal': [dx, dy], 'wrong': [dx, dy]}
        }
    """
    from src.utils.side_assignment import update_track_sides
    
    result = {
        'LEFT': {'normal': None, 'wrong': None},
        'RIGHT': {'normal': None, 'wrong': None}
    }
    
    # Ensure tracks have side assignments
    for track_id in tracks_data:
        for frame_data in tracks_data[track_id]:
            if 'side' not in frame_data:
                # Assign side if missing
                all_frames = [f for frames in tracks_data.values() for f in frames]
                update_track_sides(all_frames, side_assignment_config)
                break
    
    # Process each side
    for side in ['LEFT', 'RIGHT']:
        print(f"\n  Computing direction vectors for {side} side...")
        
        # Get normal flow tracks for this side
        normal_track_ids = identify_normal_flow_tracks(
            tracks_data, zones, flow_mappings[side], side=side
        )
        
        print(f"    Found {len(normal_track_ids)} normal-flow tracks on {side}")
        
        if not normal_track_ids:
            print(f"    ⚠️  No normal-flow tracks on {side}, cannot compute direction")
            continue
        
        # Compute normal direction
        normal_dir = compute_direction_vector_from_tracks(
            tracks_data, normal_track_ids, window, min_displacement
        )
        
        if normal_dir:
            result[side]['normal'] = normal_dir
            # Wrong direction is opposite of normal
            result[side]['wrong'] = [-normal_dir[0], -normal_dir[1]]
            
            print(f"    ✅ {side} normal direction: [{normal_dir[0]:.3f}, {normal_dir[1]:.3f}]")
            print(f"    ✅ {side} wrong direction: [{-normal_dir[0]:.3f}, {-normal_dir[1]:.3f}]")
        else:
            print(f"    ❌ Failed to compute direction for {side}")
    
    return result


def compute_unidirectional_direction_vector(tracks_data, zones, expected_mapping,
                                            window=6, min_displacement=1.0):
    """
    Compute direction vector for unidirectional (legacy) configs.
    
    Args:
        tracks_data: Dict of track_id -> list of frame data
        zones: Dict of zone_name -> Polygon (A, B, C)
        expected_mapping: Dict of "A->C" -> "normal", etc.
        window: Window for displacement calculation
        min_displacement: Minimum displacement magnitude
    
    Returns:
        dict: {'normal': [dx, dy], 'wrong': [dx, dy]} or None
    """
    print("\n  Computing direction vector for unidirectional flow...")
    
    # Identify normal flow tracks
    normal_track_ids = identify_normal_flow_tracks(
        tracks_data, zones, expected_mapping, side=None
    )
    
    print(f"    Found {len(normal_track_ids)} normal-flow tracks")
    
    if not normal_track_ids:
        print(f"    ⚠️  No normal-flow tracks found, cannot compute direction")
        return None
    
    # Compute normal direction
    normal_dir = compute_direction_vector_from_tracks(
        tracks_data, normal_track_ids, window, min_displacement
    )
    
    if not normal_dir:
        print(f"    ❌ Failed to compute direction vector")
        return None
    
    result = {
        'normal': normal_dir,
        'wrong': [-normal_dir[0], -normal_dir[1]]
    }
    
    print(f"    ✅ Normal direction: [{normal_dir[0]:.3f}, {normal_dir[1]:.3f}]")
    print(f"    ✅ Wrong direction: [{-normal_dir[0]:.3f}, {-normal_dir[1]:.3f}]")
    
    return result


if __name__ == "__main__":
    # Test displacement computation
    print("Testing dynamic direction vector computation...")
    
    # Simulate track moving rightward
    track_history = [
        (0, (100, 200)),
        (1, (105, 201)),
        (2, (110, 202)),
        (3, (115, 203)),
        (4, (120, 204)),
        (5, (125, 205)),
        (6, (130, 206)),
        (7, (135, 207)),
        (8, (140, 208)),
        (9, (145, 209))
    ]
    
    displacements = compute_displacement_vectors(track_history, window=6)
    print(f"\nDisplacements: {len(displacements)}")
    
    # Compute average direction
    vectors = [d[1] for d in displacements]
    mean_vector = np.mean(vectors, axis=0)
    normalized = mean_vector / np.linalg.norm(mean_vector)
    
    print(f"Mean displacement: {mean_vector}")
    print(f"Normalized direction: {normalized}")
    print(f"Expected: ~[1.0, 0.0] (rightward movement)")
    
    print("\n✅ Direction vector computation utilities ready!")
