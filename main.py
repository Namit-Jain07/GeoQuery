import streamlit as st
import torch
import numpy as np
import cv2 
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

st.set_page_config(page_title="GeoQuery Workspace", layout="wide")

@st.cache_resource(show_spinner=False)
def load_models():
    yolo = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='30epochs.pt',
        confidence_threshold=0.25,
        device="mps" 
    )
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    return yolo, tokenizer, llm

if "messages" not in st.session_state:
    st.session_state.messages = []

yolo_model, tokenizer, llm_model = load_models()

st.title("GeoQuery")
st.markdown("Upload satellite or aerial imagery for automated mapping and geospatial Q&A.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"], label_visibility="collapsed")

if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.messages = []
        st.session_state.last_uploaded = uploaded_file.name
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    image = Image.open(uploaded_file)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width > 1024 or image.height > 1024:
        image.thumbnail((1024, 1024))
        
    img_width, img_height = image.size
    total_area = img_width * img_height
    
    left_col, right_col = st.columns([1, 1], gap="large")
    
    with left_col:
        if "processed_image_name" not in st.session_state or st.session_state.processed_image_name != uploaded_file.name:
            
            with st.spinner("Processing the uploaded image..."):
                
                image_np = np.array(image)

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                results = get_sliced_prediction(
                    image_np,
                    yolo_model,
                    slice_height=256, 
                    slice_width=256,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    postprocess_type="GREEDYNMM",          
                    postprocess_match_threshold=0.10  
                )
                
                detailed_objects = []
                counts = {}
                annotated_image = image_np.copy()
                
                # Define a palette of distinct BGR colors
                colors_palette = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 128, 0), (128, 0, 255)]
                class_colors = {}
                
                for prediction in results.object_prediction_list:
                    cls_name = prediction.category.name
                    conf = float(prediction.score.value)
                    
                    if conf < 0.55: 
                        continue
                        
                    # Dynamically assign a color to the class if it doesn't have one yet
                    if cls_name not in class_colors:
                        class_colors[cls_name] = colors_palette[len(class_colors) % len(colors_palette)]
                    color = class_colors[cls_name]
                        
                    bbox = prediction.bbox
                    x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_image, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    box_area = box_width * box_height
                    area_percentage = (box_area / total_area) * 100
                    
                    if area_percentage < 2:
                        size_attr = "Small footprint"
                    elif area_percentage < 10:
                        size_attr = "Medium footprint"
                    else:
                        size_attr = "Large footprint"
                        
                    detailed_objects.append(f"{cls_name.title()} (Confidence: {conf:.2f}, Attribute: {size_attr})")
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                    
                st.session_state.annotated_image = annotated_image
                st.session_state.metrics = counts
                st.session_state.detailed_objects = detailed_objects
                st.session_state.processed_image_name = uploaded_file.name

        st.image(st.session_state.annotated_image, use_container_width=True, caption="Target Area with Bounding Box Overlay")

    with right_col:
        saved_counts = st.session_state.get("metrics", {})
        saved_details = st.session_state.get("detailed_objects", [])
        
        if saved_counts:
            if "auto_caption" not in st.session_state or st.session_state.get("last_uploaded_caption") != uploaded_file.name:
                with st.spinner("Processing view description based on the extracted data..."):
                    caption_context = f"DETECTED_OBJECTS: {saved_counts}\nDETAILED_ATTRIBUTES: {saved_details}"
                    
                    caption_messages = [
                        {
                            "role": "system", 
                            "content": "You are a professional geospatial AI. Write a single, fluid, natural sentence summarizing the scene based on the detected objects. Do not list them like a robot; weave them into a descriptive summary and don't use the word footprint in your description."
                        },
                        {"role": "user", "content": f"{caption_context}\n\nWrite a short caption:"}
                    ]
                    
                    cap_text = tokenizer.apply_chat_template(caption_messages, tokenize=False, add_generation_prompt=True)
                    cap_inputs = tokenizer([cap_text], return_tensors="pt").to(llm_model.device)
                    
                    with torch.no_grad():
                        cap_outputs = llm_model.generate(**cap_inputs, max_new_tokens=75, temperature=0.5)
                    cap_response_ids = cap_outputs[0][len(cap_inputs.input_ids[0]):]
                    
                    st.session_state.auto_caption = tokenizer.decode(cap_response_ids, skip_special_tokens=True).strip()
                    st.session_state.last_uploaded_caption = uploaded_file.name
                    
            st.success(f"{st.session_state.auto_caption}")
            
            with st.expander("View Detected Object Metrics", expanded=False):
                for obj in saved_details:
                    st.markdown(f"- {obj}")
        else:
            st.info("No target infrastructure identified in this frame.")
            
        st.subheader("Query Chat")
        
        chat_container = st.container(border=False)
        
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                
        if prompt := st.chat_input("Ask about the extracted spatial data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing spatial logic..."):
                        
                        context = f"DETECTED_OBJECTS: {st.session_state.get('metrics', 'None')}\nDETAILED_ATTRIBUTES: {st.session_state.get('detailed_objects', [])}"
                        
                        messages = [
                            {
                                "role": "system", 
                                "content": (
                                    "You are an expert Geospatial Intelligence Analyst. Answer user queries based EXCLUSIVELY on the provided DETECTED_OBJECTS data.\n\n"
                                    "RULES:\n"
                                    "1. OUT-OF-SCOPE REJECTION: If the user asks a general knowledge question unrelated to the image or geospatial data, explicitly reply with: 'Error: Query out of scope. I can only analyze the provided aerial structural data.'\n"
                                    "2. MISSING OBJECTS: If asked about an object (e.g., trucks, pools) not in the DETECTED_OBJECTS list, do not throw an error. State clearly that 0 were detected.\n"
                                    "3. ACCURATE COUNTING: Never group different categories together unless explicitly asked for a total count.\n"
                                    "4. DIRECTNESS: Be concise and just deliver the answer.\n"
                                    "5. Don't mention structure footprints in your responses ever unless explicitly asked to do so. They are only for your reference."
                                )
                            },
                            {"role": "user", "content": f"{context}\n\nUSER_QUERY: {prompt}"}
                        ]
                        
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
                        
                        with torch.no_grad():
                            outputs = llm_model.generate(**inputs, max_new_tokens=150, temperature=0.3)
                        response_ids = outputs[0][len(inputs.input_ids[0]):]
                        response = tokenizer.decode(response_ids, skip_special_tokens=True)
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
