import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from collections import defaultdict
import time
import streamlit as st
from PIL import Image
import io
import base64
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

# Initialize Qwen model with memory optimization
@st.cache_resource
def load_qwen_model():
    try:
        model_name = "Qwen/Qwen1.5-0.5B"  # Using a smaller 0.5B parameter model
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load Qwen model: {str(e)}")
        return None, None

def generate_description(track_id, block, position, action):
    model, tokenizer = load_qwen_model()
    if model is None or tokenizer is None:
        return f"Person {track_id} in Block {block} is {action}"
    
    prompt = (
        f"Describe classroom behavior in one sentence: "
        f"Person {track_id} in Block {block} at position {position} is {action}."
    )
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return description.replace(prompt, "").strip()
    except Exception as e:
        return f"Person {track_id} in Block {block} is {action}"

# Function to detect available cameras
def get_available_cameras():
    available_cameras = []
    for i in range(5):  # Check up to 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Streamlit UI Setup
st.set_page_config(page_title="Classroom Tracking System", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f4f7fa;
        padding: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background-color: #2c3e50;
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    .tracking-log {
        background-color: #1e2a44;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        height: 60vh;
        overflow: auto;
        resize: both;
        white-space: pre-wrap;
        font-family: 'Courier New', Courier, monospace;
        font-size: 16px;
        border: 1px solid #444;
        margin-top: 10px;
    }
    .video-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin-top: 10px;
    }
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #000000;
        padding: 8px 15px;
        border-radius: 5px;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.3s;
        margin: 0 10px 0 0;
        display: inline-block;
    }
    .status-bar {
        background-color: #ecf0f1;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
        font-size: 14px;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Header
with st.container():
    st.markdown('<div class="header"><h1>Classroom Tracking System</h1></div>', unsafe_allow_html=True)

# Initialize session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
    st.session_state.tracking_data = ""
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'selected_camera' not in st.session_state:
    st.session_state.selected_camera = 0
if 'prev_block_status' not in st.session_state:
    st.session_state.prev_block_status = {}

# Initialize UI placeholders
status_placeholder = st.empty()
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("<h3>Tracking Log</h3>", unsafe_allow_html=True)
    log_placeholder = st.empty()
with col_right:
    st.markdown("<h3>Live View</h3>", unsafe_allow_html=True)
    video_placeholder = st.empty()

# Input Source Selection
input_source = st.selectbox("Select Input Source", ["Video File", "Webcam"])
if input_source == "Video File":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
else:
    uploaded_file = None
    available_cameras = get_available_cameras()
    if available_cameras:
        camera_options = {f"Camera {i} (Index {i})": i for i in available_cameras}
        selected_camera_label = st.selectbox(
            "Select Camera",
            options=list(camera_options.keys()),
            index=0,
            key="camera_select"
        )
        st.session_state.selected_camera = camera_options[selected_camera_label]
    else:
        st.error("No cameras detected.")

# Start/Stop Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start") and not st.session_state.is_running:
        st.session_state.is_running = True
        st.session_state.tracking_data = ""  # Clear previous data
        if input_source == "Video File" and uploaded_file:
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.read())
            st.session_state.cap = cv2.VideoCapture("temp_video.mp4")
        else:
            st.session_state.cap = cv2.VideoCapture(st.session_state.selected_camera)
            
        if st.session_state.cap.isOpened():
            width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.session_state.video_dims = (width, height)
        else:
            st.error("Failed to open video source")
            st.session_state.is_running = False
            
with col2:
    if st.button("Stop") and st.session_state.is_running:
        st.session_state.is_running = False
        if st.session_state.cap:
            st.session_state.cap.release()

# Define custom grid coordinates
def get_block(x, y, width, height):
    if x < width/2:
        return 1  # Left half (Block 1)
    else:
        return 2  # Right half (Block 2)

# Tracking Logic
if st.session_state.is_running and st.session_state.cap and 'video_dims' in st.session_state:
    width, height = st.session_state.video_dims
    
    # Tracking data structures
    track_data = {
        'person_blocks': {},
        'assigned_ids': {},
        'movement_history': defaultdict(list),
        'unauthorized_flags': defaultdict(list),
        'person_posture': {},
        'person_head_orientation': {},
        'previous_center_x': {}
    }
    
    def detect_posture(y1, y2):
        bbox_height = y2 - y1
        standing_threshold = height * 0.3
        return "sitting" if bbox_height < standing_threshold else "standing"

    def detect_head_orientation(track_id, center_x):
        if track_id not in track_data['previous_center_x']:
            track_data['previous_center_x'][track_id] = center_x
            return "none"
        
        prev_center_x = track_data['previous_center_x'][track_id]
        delta_x = center_x - prev_center_x
        head_turn_threshold = 20
        orientation = "none"
        if abs(delta_x) > head_turn_threshold:
            orientation = "left" if delta_x < 0 else "right"
        track_data['previous_center_x'][track_id] = center_x
        return orientation
    
    while st.session_state.is_running:
        ret, frame = st.session_state.cap.read()
        if not ret:
            break
            
        # Run detection
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()
        
        # Process detections
        person_detections = [d for d in detections if int(d[5]) == 0]
        tracked_objects = []
        if person_detections:
            person_detections = np.array(person_detections, dtype=np.float32)
            tracked_objects = tracker.update(torch.from_numpy(person_detections), (height, width), (height, width))
        
        # Update tracking data
        frame_log = ""
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj.tlbr)
            track_id = int(obj.track_id)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get current block
            current_block = get_block(center_x, center_y, width, height)
                    
            # Update tracking data
            if track_id not in track_data['assigned_ids']:
                track_data['assigned_ids'][track_id] = f"P-{track_id}"
                
            # Detect posture and head orientation
            posture = detect_posture(y1, y2)
            head_orientation = detect_head_orientation(track_id, center_x)
            
            track_data['person_posture'][track_id] = posture
            if head_orientation != "none":
                track_data['person_head_orientation'][track_id] = head_orientation
            
            # Generate description
            action = posture
            if head_orientation != "none":
                action += f", head turned {head_orientation}"
                
            description = generate_description(
                track_data['assigned_ids'][track_id], 
                current_block, 
                (center_x, center_y), 
                action
            )
            frame_log += f"{description}\n"
            
            # Draw on frame
            color = (0, 255, 0)  # Green for normal, red for flagged
            if track_id in track_data['unauthorized_flags']:
                color = (0, 0, 255)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, track_data['assigned_ids'][track_id], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update UI with tracking data
        st.session_state.tracking_data = frame_log + st.session_state.tracking_data
        log_placeholder.markdown(f'<div class="tracking-log">{st.session_state.tracking_data}</div>', unsafe_allow_html=True)
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        status_placeholder.markdown(
            f'<div class="status-bar">Status: Processing {input_source}</div>',
            unsafe_allow_html=True
        )
        
        # Control processing speed
        time.sleep(0.1)
        
    # Cleanup
    if st.session_state.cap:
        st.session_state.cap.release()

# Update final status
if not st.session_state.is_running:
    status_placeholder.markdown(
        '<div class="status-bar">Status: Stopped</div>',
        unsafe_allow_html=True
    )

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from yolox.tracker.byte_tracker import BYTETracker
# from collections import defaultdict
# import time
# import streamlit as st
# from PIL import Image
# import io
# import base64
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# # Load YOLO model
# model = YOLO("yolov8x.pt")

# # Define ByteTrack arguments
# class TrackerArgs:
#     track_thresh = 0.5
#     match_thresh = 0.8
#     track_buffer = 30
#     mot20 = False

# # Initialize ByteTrack tracker
# tracker = BYTETracker(TrackerArgs())

# # Initialize Qwen model with memory optimization
# @st.cache_resource
# def load_qwen_model():
#     try:
#         model_name = "Qwen/Qwen1.5-0.5B"
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#         )
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             trust_remote_code=True,
#             quantization_config=quantization_config,
#             torch_dtype=torch.float16
#         )
#         return model, tokenizer
#     except Exception as e:
#         st.error(f"Failed to load Qwen model: {str(e)}")
#         return None, None

# def generate_description(track_id, block, position, action):
#     model, tokenizer = load_qwen_model()
#     if model is None or tokenizer is None:
#         return f"Person {track_id} in Block {block} is {action}"
    
#     prompt = (
#         f"Describe classroom behavior in one sentence: "
#         f"Person {track_id} in Block {block} at position {position} is {action}."
#     )
    
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=50,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             pad_token_id=tokenizer.eos_token_id
#         )
#         description = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return description.replace(prompt, "").strip()
#     except Exception as e:
#         return f"Person {track_id} in Block {block} is {action}"

# def get_available_cameras():
#     available_cameras = []
#     for i in range(5):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             available_cameras.append(i)
#             cap.release()
#     return available_cameras

# # Streamlit UI Setup
# st.set_page_config(page_title="Classroom Tracking System", layout="wide")

# # Custom CSS
# st.markdown("""
#     <style>
#     .tracking-log {
#         background-color: #1e2a44;
#         color: #ffffff;
#         padding: 15px;
#         border-radius: 10px;
#         height: 60vh;
#         overflow-y: auto;
#         font-family: monospace;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Header
# st.markdown('<div class="header"><h1>Classroom Tracking System</h1></div>', unsafe_allow_html=True)

# # Initialize session state
# if 'tracking_data' not in st.session_state:
#     st.session_state.tracking_data = []
# if 'cap' not in st.session_state:
#     st.session_state.cap = None
# if 'is_running' not in st.session_state:
#     st.session_state.is_running = False

# # UI Layout
# status_placeholder = st.empty()
# col_left, col_right = st.columns(2)

# with col_left:
#     st.markdown("<h3>Tracking Log</h3>", unsafe_allow_html=True)
#     log_container = st.empty()

# with col_right:
#     st.markdown("<h3>Live View</h3>", unsafe_allow_html=True)
#     video_placeholder = st.empty()

# # Input Source Selection
# input_source = st.selectbox("Select Input Source", ["Video File", "Webcam"])
# if input_source == "Video File":
#     uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
# else:
#     uploaded_file = None
#     available_cameras = get_available_cameras()
#     if available_cameras:
#         selected_camera = st.selectbox("Select Camera", available_cameras)
#     else:
#         st.error("No cameras detected.")

# # Start/Stop Buttons
# if st.button("Start") and not st.session_state.is_running:
#     st.session_state.is_running = True
#     st.session_state.tracking_data = []
    
#     if input_source == "Video File" and uploaded_file:
#         with open("temp_video.mp4", "wb") as f:
#             f.write(uploaded_file.read())
#         st.session_state.cap = cv2.VideoCapture("temp_video.mp4")
#     else:
#         st.session_state.cap = cv2.VideoCapture(selected_camera if input_source == "Webcam" else 0)
    
#     if not st.session_state.cap.isOpened():
#         st.error("Failed to open video source")
#         st.session_state.is_running = False

# if st.button("Stop") and st.session_state.is_running:
#     st.session_state.is_running = False
#     if st.session_state.cap:
#         st.session_state.cap.release()

# # Tracking Functions
# def get_block(x, y, width, height):
#     return 1 if x < width/2 else 2

# def detect_posture(y1, y2, height):
#     return "sitting" if (y2 - y1) < height * 0.3 else "standing"

# def detect_head_orientation(track_id, center_x, prev_centers):
#     if track_id not in prev_centers:
#         prev_centers[track_id] = center_x
#         return "none"
    
#     delta_x = center_x - prev_centers[track_id]
#     prev_centers[track_id] = center_x
#     return "left" if delta_x < -20 else "right" if delta_x > 20 else "none"

# # Main Tracking Loop
# if st.session_state.is_running and st.session_state.cap:
#     width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     track_data = {
#         'assigned_ids': {},
#         'prev_centers': {},
#         'unauthorized_flags': set()
#     }
    
#     while st.session_state.is_running:
#         ret, frame = st.session_state.cap.read()
#         if not ret:
#             break
            
#         results = model(frame)
#         detections = results[0].boxes.data.cpu().numpy()
#         person_detections = [d for d in detections if int(d[5]) == 0]
        
#         if person_detections:
#             tracked_objects = tracker.update(
#                 torch.from_numpy(np.array(person_detections, dtype=np.float32)),
#                 (height, width), (height, width)
#             )
            
#             frame_log = []
#             for obj in tracked_objects:
#                 x1, y1, x2, y2 = map(int, obj.tlbr)
#                 track_id = int(obj.track_id)
#                 center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
#                 if track_id not in track_data['assigned_ids']:
#                     track_data['assigned_ids'][track_id] = f"P-{track_id}"
                
#                 posture = detect_posture(y1, y2, height)
#                 head_orientation = detect_head_orientation(
#                     track_id, center_x, track_data['prev_centers']
#                 )
                
#                 block = get_block(center_x, center_y, width, height)
#                 action = posture
#                 if head_orientation != "none":
#                     action += f", head turned {head_orientation}"
                
#                 description = generate_description(
#                     track_data['assigned_ids'][track_id],
#                     block,
#                     (center_x, center_y),
#                     action
#                 )
#                 frame_log.append(description)
                
#                 color = (0, 0, 255) if track_id in track_data['unauthorized_flags'] else (0, 255, 0)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, track_data['assigned_ids'][track_id], 
#                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#             if frame_log:
#                 st.session_state.tracking_data = frame_log + st.session_state.tracking_data
#                 if len(st.session_state.tracking_data) > 100:
#                     st.session_state.tracking_data = st.session_state.tracking_data[:100]
                
#                 with log_container:
#                     st.markdown(
#                         f'<div class="tracking-log">{"<br>".join(st.session_state.tracking_data)}</div>',
#                         unsafe_allow_html=True
#                     )
        
#         video_placeholder.image(frame, channels="BGR", use_container_width=True)
#         status_placeholder.markdown(
#             f'<div class="status-bar">Status: Processing {input_source}</div>',
#             unsafe_allow_html=True
#         )
        
#         time.sleep(0.1)
    
#     if st.session_state.cap:
#         st.session_state.cap.release()

# if not st.session_state.is_running:
#     status_placeholder.markdown(
#         '<div class="status-bar">Status: Stopped</div>',
#         unsafe_allow_html=True
#     )