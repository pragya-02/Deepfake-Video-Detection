import os
import shutil
from tqdm import tqdm
import glob
import cv2
import numpy as np
import itertools
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_face_mesh = mp.solutions.face_mesh
base_options = python.BaseOptions(model_asset_path='weights/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=False,
                                    num_faces=1, min_face_detection_confidence = 0.01, min_face_presence_confidence = 0.01, min_tracking_confidence = 0.01)
detector = vision.FaceLandmarker.create_from_options(options)



def extract_blendshapes(vid_list, blendshapes_root, blendshapes_root_checkpoints):

    for i,vid in enumerate(tqdm(vid_list)):
        if i % 500 == 0 and i != 0:
          shutil.move(blendshapes_root, f"{blendshapes_root_checkpoints}/checkpoint_{i/500}" )
          os.makedirs(blendshapes_root, exist_ok=True)
        ori_name = vid.split("/")[-1]
        ori_name = ori_name.split(".")[0]
        ori_dir = os.path.join(blendshapes_root)
        blendshapes_dir = os.path.join(ori_dir,ori_name)
        os.makedirs(blendshapes_dir, exist_ok=True)
        capture = cv2.VideoCapture(vid)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(50):
            capture.grab()
            success, frame = capture.retrieve()
            if not success :
                continue
            
            detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            try:
                face_blendshapes =  detection_result.face_blendshapes[0]
                face_blendshapes = sorted(face_blendshapes, key=lambda x: x.category_name)
                face_blendshapes_coeff = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
                frame_blendshapes = np.array(face_blendshapes_coeff)
                blendshapes_path = os.path.join(blendshapes_dir, f"{i+1}")
                np.save(blendshapes_path, frame_blendshapes)
    
            except Exception as e:
                pass


        


def extract_landmarks(vid_list, landmarks_root, landmarks_root_checkpoint ):

    for i,vid in enumerate(tqdm(vid_list)):
        if i % 500 == 0 and i != 0:
          shutil.move(landmarks_root, f"{landmarks_root_checkpoint}/checkpoint_{i/500}" )
          os.makedirs(landmarks_root, exist_ok=True)
        ori_name = vid.split("/")[-1]
        ori_name = ori_name.split(".")[0]
        ori_dir = os.path.join(landmarks_root)
        landmark_dir = os.path.join(ori_dir,ori_name)
        os.makedirs(landmark_dir, exist_ok=True)
        capture = cv2.VideoCapture(vid)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(50):
            capture.grab()
            success, frame = capture.retrieve()
            if not success :
                continue
            
            detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            try:
                face_landmarks =  detection_result.face_landmarks[0]
                
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
                        ])
            
                frame_landmarks = np.array([])
            
                for landmark in face_landmarks_proto.landmark:
                    
                    landmark_x = np.around(landmark.x, decimals=8)
                    landmark_y = np.around(landmark.y, decimals=8)
                    landmark_z = np.around(landmark.z, decimals=8)
                    arr = np.array([landmark_x, landmark_y, landmark_z])
                    frame_landmarks = np.append(frame_landmarks, arr, axis=0)
                landmark_path = os.path.join(landmark_dir, f"{i+1}")
                np.save(landmark_path, frame_landmarks)
    
            except Exception as e:
                pass

        

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


def extract_facial_parts_frame(facial_parts_dir,i,frame, img_size):

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

    detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
    
    try:

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

        FACEMESH_CONTOURS_LIST =  np.array(FACEMESH_CONTOURS_LIST)  #128
        FACEMESH_FACE_OVAL_LIST = np.array(FACEMESH_FACE_OVAL_LIST) #36
        FACEMESH_IRISES_LIST = np.array(FACEMESH_IRISES_LIST) #8
        FACEMESH_LEFT_EYE_LIST = np.array(FACEMESH_LEFT_EYE_LIST) #16
        FACEMESH_LEFT_EYEBROW_LIST = np.array(FACEMESH_LEFT_EYEBROW_LIST) #10
        FACEMESH_LEFT_IRIS_LIST = np.array(FACEMESH_LEFT_IRIS_LIST) #4
        FACEMESH_LIPS_LIST = np.array(FACEMESH_LIPS_LIST) #40
        FACEMESH_NOSE_LIST = np.array(FACEMESH_NOSE_LIST) #24
        FACEMESH_RIGHT_EYE_LIST = np.array(FACEMESH_RIGHT_EYE_LIST) #16
        FACEMESH_RIGHT_EYEBROW_LIST = np.array(FACEMESH_RIGHT_EYEBROW_LIST) #10
        FACEMESH_RIGHT_IRIS_LIST = np.array(FACEMESH_RIGHT_IRIS_LIST) #4
        FACEMESH_TESSELATION_LIST = np.array(FACEMESH_TESSELATION_LIST) #468

        concatenated_list = np.concatenate((FACEMESH_CONTOURS_LIST, FACEMESH_FACE_OVAL_LIST, FACEMESH_IRISES_LIST, FACEMESH_LEFT_EYE_LIST, FACEMESH_LEFT_EYEBROW_LIST, FACEMESH_LEFT_IRIS_LIST, FACEMESH_LIPS_LIST, FACEMESH_NOSE_LIST, FACEMESH_RIGHT_EYE_LIST, FACEMESH_RIGHT_EYEBROW_LIST, FACEMESH_RIGHT_IRIS_LIST, FACEMESH_TESSELATION_LIST))
        # Save the list of arrays as an .npy file
        facial_parts_path = os.path.join(facial_parts_dir, f"{i+1}")
        np.save(facial_parts_path, concatenated_list)
        
    except Exception as e:
      pass









def extract_facial_parts(vid_list, facial_parts_root, facial_parts_root_checkpoints):

    for i,vid in enumerate(tqdm(vid_list)):
        if i % 500 == 0 and i != 0:
          shutil.move(facial_parts_root, f"{facial_parts_root_checkpoints}/checkpoint_{i/500}" )
          os.makedirs(facial_parts_root, exist_ok=True)
        ori_name = vid.split("/")[-1]
        ori_name = ori_name.split(".")[0]
        ori_dir = os.path.join(facial_parts_root)
        facial_parts_dir = os.path.join(ori_dir,ori_name)
        os.makedirs(facial_parts_dir, exist_ok=True)
        capture = cv2.VideoCapture(vid)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(50):
                capture.grab()
                success, frame = capture.retrieve()
                if not success :
                        continue
        
                extract_facial_parts_frame(i=i, frame=frame, img_size=224, facial_parts_dir=facial_parts_dir)

            










def main():

    data_root = "data_root"
    landmarks_root_dir = "landmarks"
    blendshapes_root_dir = "blendshapes"
    facial_parts_root_dir = "facial_parts"

    os.makedirs(landmarks_root_dir, exist_ok=True)
    os.makedirs(blendshapes_root_dir, exist_ok=True)
    os.makedirs(facial_parts_root_dir, exist_ok=True)
    
    vid_list = glob.glob(data_root +"/*.mp4")
    vid_list.sort()

    extract_landmarks(vid_list=vid_list, landmarks_root=landmarks_root_dir, landmarks_root_checkpoint="landmarks_checkpoints")
    extract_blendshapes(vid_list=vid_list, blendshapes_root=blendshapes_root_dir, blendshapes_root_checkpoints="blendshapes_checkpoints")
    extract_facial_parts(vid_list=vid_list, facial_parts_root=facial_parts_root_dir, facial_parts_root_checkpoints="facial_parts_checkpoints")



if __name__ == '__main__':
    main()
