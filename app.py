import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from collections import defaultdict
import time

# Load YOLO model
model = YOLO("yolov8x.pt")

# Define ByteTrack arguments
class TrackerArgs:
    track_thresh = 0.5
    match_thresh = 0.8
    track_buffer = 30
    mot20 = False

# Initialize ByteTrack tracker
tracker = BYTETracker(TrackerArgs())

# Video source
VIDEO_SOURCE = "class1.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Read first frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Couldn't read video frame.")
    cap.release()
    exit()

height, width, _ = frame.shape

# Define custom grid coordinates
custom_grid = [
    ((0, 0), (260, 150)),   # Block 1
    ((260, 0), (760, 150)), # Block 2
    ((0, 150), (270, 220)), # Block 3
    ((270, 150), (760, 220)), # Block 4
    ((0, 220), (440, 480)), # Block 5
    ((440, 220), (760, 480)) # Block 6
]

# Tracking data structures
person_blocks = {}
block_status = {i: set() for i in range(1, len(custom_grid) + 1)}
assigned_ids = {}
movement_history = defaultdict(list)  # Stores all movements as (from_block, to_block)
original_blocks = {}
unauthorized_ids = set()
lost_tracks = {}  # {track_id: (timestamp, block, position)}
block_objects = defaultdict(lambda: defaultdict(list))
person_positions = {}
track_loss_times = {}
unauthorized_flags = defaultdict(list)  # Now stores a list of flag descriptions

# Helper function to get the block number based on coordinates
def get_block(x, y):
    for i, ((x1, y1), (x2, y2)) in enumerate(custom_grid, 1):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None

# Function to check if positions are close (for re-identification)
def is_same_position(pos1, pos2, threshold=50):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < threshold

# Function to get the full movement path for unauthorized movement message
def get_movement_path(track_id):
    path = [original_blocks[track_id]]
    for from_block, to_block in movement_history[assigned_ids[track_id]]:
        if path[-1] == from_block:
            path.append(to_block)
    return path

fps = cap.get(cv2.CAP_PROP_FPS) or 30
start_time = time.time()
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    current_time = time.time() - start_time
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    
    # Filter detections by class (0 = person)
    person_detections = np.array([d[:5] for d in detections if int(d[5]) == 0], dtype=np.float64)
    if person_detections.shape[0] == 0:
        person_detections = np.empty((0, 6))

    tracked_objects = tracker.update(person_detections, (height, width), (height, width))

    # Clear previous block status
    prev_block_status = {k: v.copy() for k, v in block_status.items()}
    for key in block_status:
        block_status[key].clear()
        block_objects[key].clear()

    # Draw grid lines and block numbers with blue color
    for i, ((x1, y1), (x2, y2)) in enumerate(custom_grid, 1):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, str(i), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Process tracked persons
    current_positions = {}
    flag_in_current_frame = False  # Track if a flag occurs in this frame

    print(f"\nFrame-{frame_number}")
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.tlbr
        track_id = obj.track_id
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        block = get_block(center_x, center_y)

        if track_id not in assigned_ids:
            assigned_ids[track_id] = f"{block}.{len([v for v in assigned_ids.values() if v.startswith(f'{block}.')]) + 1}"
            original_blocks[track_id] = block

        assigned_id = assigned_ids[track_id]
        color = (0, 255, 0)

        # Check for re-appearance
        if track_id in lost_tracks:
            last_time, last_block, last_pos = lost_tracks[track_id]
            if block == last_block and is_same_position((center_x, center_y), last_pos):
                time_lost = current_time - last_time
                print(f"Person ID {assigned_id} re-appeared in Block {block} after {time_lost:.1f} seconds")
                del lost_tracks[track_id]
                if track_id in track_loss_times:
                    del track_loss_times[track_id]

        # Update position
        current_positions[track_id] = (center_x, center_y)
        person_positions[track_id] = (center_x, center_y)

        # Check unauthorized movement
        prev_block = person_blocks.get(track_id, None)
        if track_id in unauthorized_ids:
            color = (0, 0, 255)
        elif prev_block is not None and prev_block != block and block is not None:
            print(f"Movement in Block {block}: ID {assigned_id} entered from Block {prev_block}")
            movement_history[assigned_id].append((prev_block, block))
            if original_blocks[track_id] != block:
                path = get_movement_path(track_id)
                path_str = " through Block " + " through Block ".join(map(str, path[1:-1])) if len(path) > 2 else ""
                flag_msg = f"Unauthorized movement: Person ID {assigned_id} moved to Block {block}{path_str} from Block {original_blocks[track_id]}"
                print(flag_msg)
                unauthorized_flags[track_id].append(f"Frame-{frame_number}: {flag_msg}")
                color = (0, 0, 255)
                unauthorized_ids.add(track_id)
                flag_in_current_frame = True

        if block is not None:
            person_blocks[track_id] = block
            block_status[block].add(assigned_id)
            if track_id in lost_tracks:
                del lost_tracks[track_id]
                if track_id in track_loss_times:
                    del track_loss_times[track_id]
        else:
            if track_id in person_blocks and track_id not in lost_tracks:
                lost_tracks[track_id] = (current_time, person_blocks[track_id], person_positions[track_id])
                print(f"Person ID {assigned_id} track lost in Block {person_blocks[track_id]}")
                unauthorized_flags[track_id].append(f"Frame-{frame_number}: Person ID {assigned_id} disappeared from Block {person_blocks[track_id]}")
                flag_in_current_frame = True

        # Draw bounding box and ID
        cv2.putText(frame, f"{assigned_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Check for lost tracks
    for track_id in list(person_blocks.keys()):
        if track_id not in current_positions and track_id not in lost_tracks:
            lost_tracks[track_id] = (current_time, person_blocks[track_id], person_positions[track_id])
            print(f"Person ID {assigned_ids[track_id]} track lost in Block {person_blocks[track_id]}")
            unauthorized_flags[track_id].append(f"Frame-{frame_number}: Person ID {assigned_ids[track_id]} disappeared from Block {person_blocks[track_id]}")
            flag_in_current_frame = True

    # Process other objects
    object_counts = defaultdict(lambda: defaultdict(int))
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        if int(class_id) == 0:
            continue
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        block = get_block(center_x, center_y)
        if block is not None:
            class_name = model.names[int(class_id)]
            object_counts[block][class_name] += 1
            obj_id = f"{class_name} {block}.{object_counts[block][class_name]}"
            block_objects[block][class_name].append(obj_id)

    # Display block status
    print("Block Status:")
    for block in block_status:
        persons = block_status[block]
        persons_str = (
            f"{len(persons)} persons ("
            f"{', '.join([f'{pid} at ({person_positions[tid][0]:.1f}, {person_positions[tid][1]:.1f})' for tid, pid in assigned_ids.items() if pid in persons])}"
            f")" if persons else "0 persons"
        )
        objects_str = ', '.join([f"{len(ids)} {cls} ({', '.join(sorted(ids))})" 
                                for cls, ids in block_objects[block].items() if ids])
        print(f"Block {block}: {persons_str}{', ' + objects_str if objects_str else ''}")
    
    # Track block change updates
    print("\nBlock Change Updates:")
    for block, persons in block_status.items():
        prev_persons = prev_block_status[block]
        print(f"{'No changes' if persons == prev_persons else f'Changes'} in Block {block}: ({', '.join(persons) if persons else 'empty'})")
        
    total_movements = sum(len(moves) for moves in movement_history.values())
    total_persons = len(current_positions)
    print(f"\nTotal movements recorded: {total_movements}")
    print(f"Total persons tracked: {total_persons}")

    # Print flags summary only if a flag occurred in this frame
    if flag_in_current_frame and unauthorized_flags:
        print("Unauthorized Flags Summary:")
        for track_id, flags in unauthorized_flags.items():
            if any(f"Frame-{frame_number}" in flag for flag in flags):
                print(f"Person ID {assigned_ids[track_id]}: {len(flags)} flags (Latest: {flags[-1]})")

    cv2.imshow("Classroom Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Print final summary on exit
        print("\nFinal Summary of Unauthorized Actions:")
        for track_id, flags in unauthorized_flags.items():
            print(f"Person ID {assigned_ids[track_id]}:")
            print(f"  Total Flags: {len(flags)}")
            print(f"  Movements: {', '.join([f'Block {from_b} to Block {to_b}' for from_b, to_b in movement_history[assigned_ids[track_id]]])}")
            print("  Flags:")
            for flag in flags:
                print(f"    {flag}")
        break

cap.release()
cv2.destroyAllWindows()

#2
import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from collections import defaultdict
import time

# Load YOLO model
model = YOLO("yolov8x.pt")

# Define ByteTrack arguments
class TrackerArgs:
    track_thresh = 0.5
    match_thresh = 0.8
    track_buffer = 30
    mot20 = False

# Initialize ByteTrack tracker
tracker = BYTETracker(TrackerArgs())

# Video source
VIDEO_SOURCE = "class1.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Read first frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Couldn't read video frame.")
    cap.release()
    exit()

height, width, _ = frame.shape

# Define custom grid coordinates
custom_grid = [
    ((0, 0), (260, 150)),   # Block 1
    ((260, 0), (760, 150)), # Block 2
    ((0, 150), (270, 220)), # Block 3
    ((270, 150), (760, 220)), # Block 4
    ((0, 220), (440, 480)), # Block 5
    ((440, 220), (760, 480)) # Block 6
]

# Tracking data structures
person_blocks = {}
block_status = {i: set() for i in range(1, len(custom_grid) + 1)}
assigned_ids = {}
movement_history = defaultdict(list)  # Stores all movements as (from_block, to_block)
original_blocks = {}
unauthorized_ids = set()
lost_tracks = {}  # {track_id: (timestamp, block, position)}
block_objects = defaultdict(lambda: defaultdict(list))
person_positions = {}
track_loss_times = {}
unauthorized_flags = defaultdict(list)  # Now stores a list of flag descriptions

# Helper function to get the block number based on coordinates
def get_block(x, y):
    for i, ((x1, y1), (x2, y2)) in enumerate(custom_grid, 1):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None

# Function to check if positions are close (for re-identification)
def is_same_position(pos1, pos2, threshold=50):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < threshold

# Function to get the full movement path for unauthorized movement message
def get_movement_path(track_id):
    path = [original_blocks[track_id]]
    for from_block, to_block in movement_history[assigned_ids[track_id]]:
        if path[-1] == from_block:
            path.append(to_block)
    return path

fps = cap.get(cv2.CAP_PROP_FPS) or 30
start_time = time.time()
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    current_time = time.time() - start_time
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    
    # Filter detections by class (0 = person)
    person_detections = np.array([d[:5] for d in detections if int(d[5]) == 0], dtype=np.float64)
    if person_detections.shape[0] == 0:
        person_detections = np.empty((0, 6))

    tracked_objects = tracker.update(person_detections, (height, width), (height, width))

    # Clear previous block status
    prev_block_status = {k: v.copy() for k, v in block_status.items()}
    for key in block_status:
        block_status[key].clear()
        block_objects[key].clear()

    # Draw grid lines and block numbers with blue color
    for i, ((x1, y1), (x2, y2)) in enumerate(custom_grid, 1):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, str(i), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Process tracked persons
    current_positions = {}
    flag_in_current_frame = False  # Track if a flag occurs in this frame

    print(f"\nFrame-{frame_number}")
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.tlbr
        track_id = obj.track_id
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        block = get_block(center_x, center_y)

        if track_id not in assigned_ids:
            assigned_ids[track_id] = f"{block}.{len([v for v in assigned_ids.values() if v.startswith(f'{block}.')]) + 1}"
            original_blocks[track_id] = block

        assigned_id = assigned_ids[track_id]
        color = (0, 255, 0)

        # Check for re-appearance
        if track_id in lost_tracks:
            last_time, last_block, last_pos = lost_tracks[track_id]
            if block == last_block and is_same_position((center_x, center_y), last_pos):
                time_lost = current_time - last_time
                print(f"Person ID {assigned_id} re-appeared in Block {block} after {time_lost:.1f} seconds")
                del lost_tracks[track_id]
                if track_id in track_loss_times:
                    del track_loss_times[track_id]

        # Update position
        current_positions[track_id] = (center_x, center_y)
        person_positions[track_id] = (center_x, center_y)

        # Check unauthorized movement
        prev_block = person_blocks.get(track_id, None)
        if track_id in unauthorized_ids:
            color = (0, 0, 255)
        elif prev_block is not None and prev_block != block and block is not None:
            print(f"Movement in Block {block}: ID {assigned_id} entered from Block {prev_block}")
            movement_history[assigned_id].append((prev_block, block))
            if original_blocks[track_id] != block:
                path = get_movement_path(track_id)
                path_str = " through Block " + " through Block ".join(map(str, path[1:-1])) if len(path) > 2 else ""
                flag_msg = f"Unauthorized movement: Person ID {assigned_id} moved to Block {block}{path_str} from Block {original_blocks[track_id]}"
                print(flag_msg)
                unauthorized_flags[track_id].append(f"Frame-{frame_number}: {flag_msg}")
                color = (0, 0, 255)
                unauthorized_ids.add(track_id)
                flag_in_current_frame = True

        if block is not None:
            person_blocks[track_id] = block
            block_status[block].add(assigned_id)
            if track_id in lost_tracks:
                del lost_tracks[track_id]
                if track_id in track_loss_times:
                    del track_loss_times[track_id]
        else:
            if track_id in person_blocks and track_id not in lost_tracks:
                lost_tracks[track_id] = (current_time, person_blocks[track_id], person_positions[track_id])
                print(f"Person ID {assigned_id} track lost in Block {person_blocks[track_id]}")
                unauthorized_flags[track_id].append(f"Frame-{frame_number}: Person ID {assigned_id} disappeared from Block {person_blocks[track_id]}")
                flag_in_current_frame = True

        # Draw bounding box and ID
        cv2.putText(frame, f"{assigned_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Check for lost tracks
    for track_id in list(person_blocks.keys()):
        if track_id not in current_positions and track_id not in lost_tracks:
            lost_tracks[track_id] = (current_time, person_blocks[track_id], person_positions[track_id])
            print(f"Person ID {assigned_ids[track_id]} track lost in Block {person_blocks[track_id]}")
            unauthorized_flags[track_id].append(f"Frame-{frame_number}: Person ID {assigned_ids[track_id]} disappeared from Block {person_blocks[track_id]}")
            flag_in_current_frame = True

    # Process other objects
    object_counts = defaultdict(lambda: defaultdict(int))
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        if int(class_id) == 0:
            continue
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        block = get_block(center_x, center_y)
        if block is not None:
            class_name = model.names[int(class_id)]
            object_counts[block][class_name] += 1
            obj_id = f"{class_name} {block}.{object_counts[block][class_name]}"
            block_objects[block][class_name].append(obj_id)

    # Display block status
    print("Block Status:")
    for block in block_status:
        persons = block_status[block]
        persons_str = (
            f"{len(persons)} persons ("
            f"{', '.join([f'{pid} at ({person_positions[tid][0]:.1f}, {person_positions[tid][1]:.1f})' for tid, pid in assigned_ids.items() if pid in persons])}"
            f")" if persons else "0 persons"
        )
        objects_str = ', '.join([f"{len(ids)} {cls} ({', '.join(sorted(ids))})" 
                                for cls, ids in block_objects[block].items() if ids])
        print(f"Block {block}: {persons_str}{', ' + objects_str if objects_str else ''}")
    
    # Track block change updates
    print("\nBlock Change Updates:")
    for block, persons in block_status.items():
        prev_persons = prev_block_status[block]
        print(f"{'No changes' if persons == prev_persons else f'Changes'} in Block {block}: ({', '.join(persons) if persons else 'empty'})")
        
    total_movements = sum(len(moves) for moves in movement_history.values())
    total_persons = len(current_positions)
    print(f"\nTotal movements recorded: {total_movements}")
    print(f"Total persons tracked: {total_persons}")

    # Print flags summary only if a flag occurred in this frame
    if flag_in_current_frame and unauthorized_flags:
        print("Unauthorized Flags Summary:")
        for track_id, flags in unauthorized_flags.items():
            if any(f"Frame-{frame_number}" in flag for flag in flags):
                print(f"Person ID {assigned_ids[track_id]}: {len(flags)} flags (Latest: {flags[-1]})")

    cv2.imshow("Classroom Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Print final summary on exit
        print("\nFinal Summary of Unauthorized Actions:")
        for track_id, flags in unauthorized_flags.items():
            print(f"Person ID {assigned_ids[track_id]}:")
            print(f"  Total Flags: {len(flags)}")
            print(f"  Movements: {', '.join([f'Block {from_b} to Block {to_b}' for from_b, to_b in movement_history[assigned_ids[track_id]]])}")
            print("  Flags:")
            for flag in flags:
                print(f"    {flag}")
        break

cap.release()
cv2.destroyAllWindows()
