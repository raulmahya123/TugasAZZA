import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(0, 255, 0))
class FaceMesh:
    
    def __init__(self, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.faces = mp_face_mesh.FaceMesh(max_num_faces=max_num_faces, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)
    
    
    def findFaceLandMarks(self, image, faceNumber=0, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #
        results = self.faces.process(image)
        landMarkList = []

        if results.multi_face_landmarks:  
            face = results.multi_face_landmarks[faceNumber] 
            for id, landMark in enumerate(face.landmark):
                imgH, imgW, imgC = originalImage.shape  
                xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                landMarkList.append([id, xPos, yPos])
            if draw:
                
                mp_drawing.draw_landmarks(
                    image=originalImage,
                    landmark_list=face,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        return landMarkList