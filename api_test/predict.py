import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
import time
import itertools
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import io
import os




# config params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
seqlen = 10
duration = 4
img_size = 224
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])
transform1 = transforms.Compose([  transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])



# deepfake model
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes, lstm_hidden_dim=2612, bidirectional=True):
        super(DeepfakeDetector, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # AdaptiveAvgPool2d to replace avgpool
        self.lstm = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.dense1 = nn.Linear(lstm_hidden_dim * (2 if bidirectional else 1), 512)
        self.dense2=nn.Linear(512, 2)
        self.dp1 = nn.Dropout(0.4)
        self.dp2 = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.facenet = InceptionResnetV1(pretrained='vggface2')

    def forward(self, x, y, p1):
        batch_size, seq_length, c, h, w = x.shape
        batch_size, seq_length, c, h1, w1 = p1.shape


        x = x.view(batch_size * seq_length, c, h, w)
        p1 = p1.view(batch_size * seq_length, c, h1, w1)


        p1 = self.facenet(p1)


        x = self.model(x)
        x = self.avgpool(x).flatten(1)

        x = x.view(batch_size, seq_length, -1)  # Flatten x
        y = y.view(batch_size, seq_length, -1)  # Flatten x
        p1 = p1.view(batch_size, seq_length, -1)  # Flatten x

        
        x = torch.cat((x, y, p1), dim=2)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(batch_size * seq_length, -1)
        lstm_out = lstm_out.view(batch_size, seq_length, -1) 
        lstm_out = lstm_out[:, -1, :]  
        lstm_out = self.dp1(lstm_out) #dropout layer 1
        output1 = self.relu(self.dense1(lstm_out))
        output1 = self.dp2(output1) #dropout layer 2
        output=self.sigmoid(self.dense2(output1))
        return 0, output




# initialise detector
mp_face_mesh = mp.solutions.face_mesh
base_options = python.BaseOptions(model_asset_path='weights/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=False,
                                    num_faces=1, min_face_detection_confidence = 0.01, min_face_presence_confidence = 0.01, min_tracking_confidence = 0.01)
detector = vision.FaceLandmarker.create_from_options(options)

# Load checkpoint
checkpoint = torch.load("model/checkpoint.pt", map_location=device)

# Initialize deepfake detection model
deepfake_model = DeepfakeDetector(num_classes=2)

# Load model's state dict
deepfake_model.load_state_dict(checkpoint['model_state_dict'])

# Set model to evaluation mode
deepfake_model.eval()

# Move model to CUDA if available
deepfake_model = deepfake_model.to(device)
del checkpoint




# frames extraction
def extract_frames(video_path, duration=4):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    
    # Get the total number of frames and calculate the duration of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    
    frames_per_second = []
    
    # Iterate through each second of the video
    for sec in range(int(duration)):
        frames_in_sec = []
        # Set the frame position to the start of the current second
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        
        # Extract frames for the current second
        for _ in range(fps):
            ret, frame = cap.read()
            if not ret:
                break
            frames_in_sec.append(frame)
        
        frames_per_second.append(frames_in_sec)
    
    # Release the video capture object
    cap.release()
    
    return frames_per_second




# face detection module
def detect_and_align_faces(frames_list, img_size=224, seqlen=10):
    processed_frames_per_second = []
    for frames in frames_list:
        processed_frames_in_sec = []
        for frame in frames:
            try:
                faces = RetinaFace.extract_faces(img_path = frame, align = True, align_first=True)
                processed_frames_in_sec.append(cv2.resize(faces[0], (img_size,img_size)))
                if len(faces) > 1:
                     return None
                
            except Exception as e:
                pass
            if (len(processed_frames_in_sec) == seqlen):
                break
        if (len(processed_frames_in_sec) == seqlen):
            processed_frames_per_second.append(processed_frames_in_sec)
    return processed_frames_per_second




# blendshape extraction module
def extract_blendshapes(detection_result):
        face_blendshapes =  detection_result.face_blendshapes[0]
        face_blendshapes = sorted(face_blendshapes, key=lambda x: x.category_name)
        face_blendshapes_coeff = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        return face_blendshapes_coeff




# facial parts extraction module

'''
 'FACEMESH_CONTOURS',
 'FACEMESH_FACE_OVAL',
 'FACEMESH_IRISES',
 'FACEMESH_LEFT_EYE',
 'FACEMESH_LEFT_EYEBROW',
 'FACEMESH_LEFT_IRIS',
 'FACEMESH_LIPS',
 'FACEMESH_NOSE',
 'FACEMESH_RIGHT_EYE',
 'FACEMESH_RIGHT_EYEBROW',
 'FACEMESH_RIGHT_IRIS',
 'FACEMESH_TESSELATION',
'''


def extract_facial_parts(detection_result, img_size=224):

        FACEMESH_CONTOURS_LIST = []
        FACEMESH_FACE_OVAL_LIST =[]
        FACEMESH_IRISES_LIST =[]
        FACEMESH_LEFT_EYE_LIST =[]
        FACEMESH_LEFT_EYEBROW_LIST =[]
        FACEMESH_LEFT_IRIS_LIST =[]
        FACEMESH_LIPS_LIST =[]
        FACEMESH_NOSE_LIST =[]
        FACEMESH_RIGHT_EYE_LIST =[]
        FACEMESH_RIGHT_EYEBROW_LIST =[]
        FACEMESH_RIGHT_IRIS_LIST =[]
        FACEMESH_TESSELATION_LIST =[]

        FACEMESH_CONTOURS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))
        FACEMESH_FACE_OVAL_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL)))
        FACEMESH_IRISES_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_IRISES)))
        FACEMESH_LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
        FACEMESH_LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
        FACEMESH_LEFT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_IRIS)))
        FACEMESH_LIPS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
        FACEMESH_NOSE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_NOSE)))
        FACEMESH_RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
        FACEMESH_RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
        FACEMESH_RIGHT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_IRIS)))
        FACEMESH_TESSELATION_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_TESSELATION)))



        face_landmarks =  detection_result.face_landmarks[0]

        for FACEMESH_CONTOURS_INDEX in FACEMESH_CONTOURS_INDEXES:
                FACEMESH_CONTOURS_LIST.append([int(face_landmarks[FACEMESH_CONTOURS_INDEX].x * img_size),int(face_landmarks[FACEMESH_CONTOURS_INDEX].y * img_size)])

        for FACEMESH_FACE_OVAL_INDEX in FACEMESH_FACE_OVAL_INDEXES:
                FACEMESH_FACE_OVAL_LIST.append([int(face_landmarks[FACEMESH_FACE_OVAL_INDEX].x * img_size),int(face_landmarks[FACEMESH_FACE_OVAL_INDEX].y * img_size)])
                
        for FACEMESH_IRISES_INDEX in FACEMESH_IRISES_INDEXES:
                FACEMESH_IRISES_LIST.append([int(face_landmarks[FACEMESH_IRISES_INDEX].x * img_size),int(face_landmarks[FACEMESH_IRISES_INDEX].y * img_size)])

        for FACEMESH_LEFT_EYE_INDEX in FACEMESH_LEFT_EYE_INDEXES:
                FACEMESH_LEFT_EYE_LIST.append([int(face_landmarks[FACEMESH_LEFT_EYE_INDEX].x * img_size),int(face_landmarks[FACEMESH_LEFT_EYE_INDEX].y * img_size)])

        for FACEMESH_LEFT_EYEBROW_INDEX in FACEMESH_LEFT_EYEBROW_INDEXES:
                FACEMESH_LEFT_EYEBROW_LIST.append([int(face_landmarks[FACEMESH_LEFT_EYEBROW_INDEX].x * img_size),int(face_landmarks[FACEMESH_LEFT_EYEBROW_INDEX].y * img_size)])

        for FACEMESH_LEFT_IRIS_INDEX in FACEMESH_LEFT_IRIS_INDEXES:
                FACEMESH_LEFT_IRIS_LIST.append([int(face_landmarks[FACEMESH_LEFT_IRIS_INDEX].x * img_size),int(face_landmarks[FACEMESH_LEFT_IRIS_INDEX].y * img_size)])

        for FACEMESH_LIPS_INDEX in FACEMESH_LIPS_INDEXES:
                FACEMESH_LIPS_LIST.append([int(face_landmarks[FACEMESH_LIPS_INDEX].x * img_size),int(face_landmarks[FACEMESH_LIPS_INDEX].y * img_size)])

        for FACEMESH_NOSE_INDEX in FACEMESH_NOSE_INDEXES:
                FACEMESH_NOSE_LIST.append([int(face_landmarks[FACEMESH_NOSE_INDEX].x * img_size),int(face_landmarks[FACEMESH_NOSE_INDEX].y * img_size)])

        for FACEMESH_RIGHT_EYE_INDEX in FACEMESH_RIGHT_EYE_INDEXES:
                FACEMESH_RIGHT_EYE_LIST.append([int(face_landmarks[FACEMESH_RIGHT_EYE_INDEX].x * img_size),int(face_landmarks[FACEMESH_RIGHT_EYE_INDEX].y * img_size)])

        for FACEMESH_RIGHT_EYEBROW_INDEX in FACEMESH_RIGHT_EYEBROW_INDEXES:
                FACEMESH_RIGHT_EYEBROW_LIST.append([int(face_landmarks[FACEMESH_RIGHT_EYEBROW_INDEX].x * img_size),int(face_landmarks[FACEMESH_RIGHT_EYEBROW_INDEX].y * img_size)])
        for FACEMESH_RIGHT_IRIS_INDEX in FACEMESH_RIGHT_IRIS_INDEXES:
                FACEMESH_RIGHT_IRIS_LIST.append([int(face_landmarks[FACEMESH_RIGHT_IRIS_INDEX].x * img_size),int(face_landmarks[FACEMESH_RIGHT_IRIS_INDEX].y * img_size)])

        for FACEMESH_TESSELATION_INDEX in FACEMESH_TESSELATION_INDEXES:
                FACEMESH_TESSELATION_LIST.append([int(face_landmarks[FACEMESH_TESSELATION_INDEX].x * img_size),int(face_landmarks[FACEMESH_TESSELATION_INDEX].y * img_size)])

        FACEMESH_CONTOURS_LIST =  (FACEMESH_CONTOURS_LIST)  #128
        FACEMESH_FACE_OVAL_LIST = (FACEMESH_FACE_OVAL_LIST) #36
        FACEMESH_IRISES_LIST = (FACEMESH_IRISES_LIST) #8
        FACEMESH_LEFT_EYE_LIST = (FACEMESH_LEFT_EYE_LIST) #16
        FACEMESH_LEFT_EYEBROW_LIST = (FACEMESH_LEFT_EYEBROW_LIST) #10
        FACEMESH_LEFT_IRIS_LIST = (FACEMESH_LEFT_IRIS_LIST) #4
        FACEMESH_LIPS_LIST = (FACEMESH_LIPS_LIST) #40
        FACEMESH_NOSE_LIST = (FACEMESH_NOSE_LIST) #24
        FACEMESH_RIGHT_EYE_LIST = (FACEMESH_RIGHT_EYE_LIST) #16
        FACEMESH_RIGHT_EYEBROW_LIST = (FACEMESH_RIGHT_EYEBROW_LIST) #10
        FACEMESH_RIGHT_IRIS_LIST = (FACEMESH_RIGHT_IRIS_LIST) #4
        FACEMESH_TESSELATION_LIST = (FACEMESH_TESSELATION_LIST) #468

        concatenated_list = (FACEMESH_CONTOURS_LIST + FACEMESH_FACE_OVAL_LIST + FACEMESH_IRISES_LIST + FACEMESH_LEFT_EYE_LIST + FACEMESH_LEFT_EYEBROW_LIST + FACEMESH_LEFT_IRIS_LIST + FACEMESH_LIPS_LIST + FACEMESH_NOSE_LIST + FACEMESH_RIGHT_EYE_LIST + FACEMESH_RIGHT_EYEBROW_LIST + FACEMESH_RIGHT_IRIS_LIST + FACEMESH_TESSELATION_LIST)
        
        return concatenated_list



# blendshape extraction module
def extract_blendshapes_facial_parts(processed_frames, img_size=224):
    processed_frames_blendshapes = []
    processed_frames_facial_parts = []
   
    for frames_in_sec in processed_frames:
        processed_frames_in_sec_blendshapes = []
        processed_frames_in_sec_facial_parts = []

        for frame in frames_in_sec: 
            detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            try:
                blendshapes_temp = extract_blendshapes(detection_result)
                processed_frames_in_sec_blendshapes.append(torch.tensor(blendshapes_temp))
                
            except Exception as e:
                pass

            try:
                facial_parts_temp = extract_facial_parts(detection_result, img_size=img_size)
                processed_frames_in_sec_facial_parts.append(facial_parts_temp)
 
            except Exception as e:
                pass
            
        processed_frames_blendshapes.append(processed_frames_in_sec_blendshapes)
        processed_frames_facial_parts.append(processed_frames_in_sec_facial_parts)
        

    return processed_frames_blendshapes, processed_frames_facial_parts




# function to pad frames with black pixels
def pad_frame(frame, target_size=(100, 100)):

    height, width, _ = frame.shape

    if width < target_size[1]:
        left_padding = (target_size[0] - width) // 2
        right_padding = target_size[0] - width - left_padding
        frame = cv2.copyMakeBorder(frame, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    if height < target_size[0]:
        top_padding = (target_size[1] - height) // 2
        bottom_padding = target_size[1] - height - top_padding
        frame = cv2.copyMakeBorder(frame, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return cv2.resize(frame,target_size)




# extract individual facial parts
index_list = {
    "top" : [[164,172],[188,198],[282,292]],
    "nose_lips" : [[202,242],[242,266]]

    }

def extract_individual_facial_parts(frame, processed_frame_facial_parts, img_size ):

    facial_parts_dict = {}

    for idx in index_list:
        concat = []
        for i in index_list[idx]:
            temp = processed_frame_facial_parts[i[0]:i[1]]
            for val in temp:
                concat.append(val)

        f_set = concat
        min_x, min_y = max_x, max_y = next(iter(f_set))
        for point in f_set:
            x, y = point
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        # Extreme diagonal vertices
        min_x = int(min_x * 1)
        min_x = max(0,min_x)
        max_x = int(max_x * 1)
        max_x = min(img_size,max_x)
        min_y = int(min_y * 1)
        min_y = max(0,min_y)
        max_y = int(max_y * 1)
        max_y = min(img_size,max_y)
        x = (min_x, max_x)
        y = (min_y, max_y)

        temp = frame[y[0]:y[1], x[0]:x[1]]

        if idx == "top":
            facial_parts_dict.update({"top" : temp})
        elif idx == "nose_lips":
            facial_parts_dict.update({"nose_lips" : temp})

    return facial_parts_dict





# combine the facial parts into single image
def combined_canvas(img1, img2):
    
    # Define the gap size
    gap_size = 10  # Adjust as needed
    
    # Create a blank canvas to accommodate both images with the gap
    canvas_height = img1.shape[0] + img2.shape[0] + gap_size
    canvas_width = max(img1.shape[1], img2.shape[1])
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Calculate the vertical offset to align the second image in the middle
    offset_y = (canvas_height - img1.shape[0] - img2.shape[0] - gap_size) // 2

    # Copy img1 to the top portion of the canvas
    canvas[:img1.shape[0], :img1.shape[1]] = img1

    # Calculate the horizontal offset to align the second image in the middle
    offset_x = (canvas_width - img2.shape[1]) // 2

    # Copy img2 to the middle portion of the canvas
    canvas[offset_y + img1.shape[0] + gap_size:offset_y + img1.shape[0] + img2.shape[0] + gap_size, offset_x:offset_x + img2.shape[1]] = img2


    return pad_frame(canvas,[124,124])





# creating facial parts crops
def crop_facial_parts(processed_frames,processed_frames_facial_parts, img_size=img_size):
    processed_frames_cropped_facial_parts = []
    for i in range(len(processed_frames)):
        processed_frames_cropped_facial_parts_temp = []
        for j in range(len(processed_frames[i])):
        
           facial_parts_temp = extract_individual_facial_parts(processed_frames[i][j], processed_frames_facial_parts[i][j], img_size=img_size)
           processed_frames_cropped_facial_parts_temp.append(facial_parts_temp)
        processed_frames_cropped_facial_parts.append(processed_frames_cropped_facial_parts_temp)

    return processed_frames_cropped_facial_parts




# applying transformation on ip
def prepare_ip(processed_frames, transform):
    ip = []        
    for frames_in_sec in processed_frames:
        temp_ip = []
        for frame in frames_in_sec:  
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_tensor = transform(frame)
            temp_ip.append(frame_tensor)
        ip.append(temp_ip)
    return ip




# applying transformation on facial crops
def prepare_facial_crops(cropped_facial_parts, transform):
    facial_crops_combined_ip = []     
    for frames_in_sec in cropped_facial_parts:
        temp_facial_crops_combined_ip = []
        for frame in frames_in_sec:  
            frame = Image.fromarray(cv2.cvtColor(combined_canvas(frame["top"],frame["nose_lips"]), cv2.COLOR_BGR2RGB))
            frame_tensor = transform(frame)
            temp_facial_crops_combined_ip.append(frame_tensor)

        facial_crops_combined_ip.append(temp_facial_crops_combined_ip)

    return facial_crops_combined_ip



# unit inferencing
def unit_inference(deepfake_model, ip, blendshapes_ip, p1):
    ip = torch.stack(ip)
    ip = ip.unsqueeze(0) 

    blendshapes_ip = torch.stack(blendshapes_ip)
    blendshapes_ip = blendshapes_ip.unsqueeze(0) 
          
    p1 = torch.stack(p1)
    p1 = p1.unsqueeze(0)



    with torch.no_grad():
          _, output = deepfake_model(ip.to(device), blendshapes_ip.to(device), p1.to(device))
          _,prediction = torch.max(output,1)
          prediction = int(prediction.item())
          confidence = output[:,prediction].item()*100
    return prediction, confidence





def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  return plt




# tensor into image
def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    r,g,b = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    image = image.astype(int)
    return image


# prediction function
def predict(vid_path, temp_path=None, seqlen=10, img_size=224):
   try:  
    print("\n")
    start_time = time.time()

    vidname = vid_path.split("/")[-1]
    print(f"Video Name: {vidname}")
    frames_per_second = extract_frames(vid_path, duration=duration)

    st = time.time()
    processed_frames = detect_and_align_faces(frames_per_second, img_size=img_size, seqlen=seqlen)
    if processed_frames is None:
        print("Multiple Faces Detected")
        result = {"Video Name": vidname,
            "Predicted Output": "MULTIPLE FACES DETECTED",
            "Prediction Confidence": f"0 %",
            "Inference Time": "0 sec"
            }
        return result
    

         
    print(f"Valid Duration: {len(processed_frames)}")
    et = time.time()
    print(f"Face Detection: {(et-st):.2f} sec")

    if not len(processed_frames):
      print("No Face Detected")
      result = {"Video Name": vidname,
              "Predicted Output": "NO FACE DETECTED",
              "Prediction Confidence": f"0 %",
              "Inference Time": "0 sec"
              }
      return result

    st = time.time()
    processed_frames_blendshapes, processed_frames_facial_parts = extract_blendshapes_facial_parts(processed_frames, img_size=img_size)
    et = time.time()
    print(f"Blendshapes and Facial Parts Extraction: {(et-st):.2f} sec")
      
    cropped_facial_parts = crop_facial_parts(processed_frames, processed_frames_facial_parts, img_size=img_size)
    p1 = prepare_facial_crops(cropped_facial_parts, transform=transform1)
    ip = prepare_ip(processed_frames, transform=transform)

    if temp_path is not None:
        cropped_face = processed_frames[0][0]
        temp_detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_face))
        cropped_face_landmarks = draw_landmarks_on_image(cropped_face, temp_detection_result)
        cropped_face_blendshapes = plot_face_blendshapes_bar_graph(temp_detection_result.face_blendshapes[0])
        cropped_facial_part = im_plot(p1[0][0])

        image_path = os.path.join(temp_path, 'image_cropped_face.png')
        Image.fromarray(cropped_face).save(image_path)

        image_path = os.path.join(temp_path, 'image_cropped_face_landmarks.png')
        Image.fromarray(cropped_face_landmarks).save(image_path)

        image_path = os.path.join(temp_path, 'image_cropped_face_blendshapes.png')
        cropped_face_blendshapes.savefig(image_path, format='png', bbox_inches='tight', pad_inches=0)

        image_path = os.path.join(temp_path, 'image_cropped_facial_part.png')
        cv2.imwrite(image_path, cropped_facial_part)

   
    preds = []
    confs = []

    st = time.time()
    try:
        for i in range(len(ip)):
            pred, conf = unit_inference(deepfake_model=deepfake_model, ip=ip[i], blendshapes_ip=processed_frames_blendshapes[i], p1=p1[i])
            preds.append(pred)
            confs.append(conf)
    except Exception as e:
         print(e)
    et = time.time()

    print(f"Predictions and their Confidence: {preds, confs}")
    print(f"Model Inferencing for each Iteration: {((et-st)/len(ip)):.2f} sec")
    
    end_time = time.time() 
    total_inference_time = end_time - start_time

    pred_op = ""

    if preds.count(1) > preds.count(0):
      confidence = 0.0
      for idx in  [i for i, x in enumerate(preds) if x == 1]:
        confidence += confs[idx]
      confidence /= preds.count(1)
      confidence -= 2
      pred_op = "REAL"

      result = {"Video Name": vidname,
              "Predicted Output": pred_op,
              "Prediction Confidence": f"{confidence:.2f} %",
              "Inference Time": f"{total_inference_time:.2f} sec"
              }

    elif preds.count(1) < preds.count(0):
      confidence = 0.0
      for idx in  [i for i, x in enumerate(preds) if x == 0]:
        confidence += confs[idx]
      confidence /= preds.count(0)
      confidence -= 2
      pred_op = "DEEPFAKE"

      result = {"Video Name": vidname,
              "Predicted Output": pred_op,
              "Prediction Confidence": f"{confidence:.2f} %",
              "Inference Time": f"{total_inference_time:.2f} sec"
              }

    else:
      confidence = 50.0
      pred_op = "SUSPECTED"

      result = {"Video Name": vidname,
              "Predicted Output": pred_op,
              "Prediction Confidence": f"{confidence:.2f} %",
              "Inference Time": f"{total_inference_time:.2f} sec"
              }
    print(f"Predicted Output: {pred_op}")
    print(f"Prediction Confidence: {confidence:.2f} %")
    print(f"Inference Time: {total_inference_time:.2f} sec")
    print("\n")

    del ip, p1
    
    return result
   
   except Exception as e:
       print(e)



       