import os, cv2, math, numpy as np, zipfile, urllib.request
from google.colab import files
from IPython.display import display, Image

# TTS fallback
try:
    from elevenlabslib import *
    ELEVENLABS_AVAILABLE = True
except:
    ELEVENLABS_AVAILABLE = False
    from gtts import gTTS

import mediapipe as mp

# Cleanup
os.chdir('/content')
!rm -rf /content/*.mp4 /content/*.jpg /content/animation_frames 2>/dev/null

# Load / Upload Photo
image_files = [f for f in os.listdir('/content') if f.lower().endswith(('.jpg','.jpeg','.png'))]
if image_files:
    photo_path = f"/content/{image_files[0]}"
else:
    print("📤 Upload a clear face photo...")
    uploaded = files.upload()
    for filename in uploaded.keys():
        if filename.lower().endswith(('.jpg','.jpeg','.png')):
            photo_path = f"/content/{filename}"
            break
    else:
        urllib.request.urlretrieve(
            "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512&h=512&fit=crop",
            "/content/sample_face.jpg"
        )
        photo_path = "/content/sample_face.jpg"

img = cv2.imread(photo_path)
img = cv2.resize(img,(512,512))
cv2.imwrite('/content/face.jpg',img)
_, buffer = cv2.imencode('.jpg', img)
display(Image(data=buffer.tobytes(), width=300))

expression_text = input("Enter text to speak: ").strip()
print(f"🎭 Generating talking head for: {expression_text}")

# ------------------ FACE ANIMATOR ------------------
class BinderFaceAnimator:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        )

    def detect_landmarks(self, image):
        rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results=self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h,w=image.shape[:2]
            return [(int(lm.x*w), int(lm.y*h)) for lm in landmarks.landmark]
        return None

    def deform_face(self, img, landmarks, frame_idx, total_frames):
        result = img.copy()
        h,w=img.shape[:2]
        if not landmarks:
            return result

        # Lips
        mouth_pts = landmarks[61:88]
        xs=[p[0] for p in mouth_pts]; ys=[p[1] for p in mouth_pts]
        x1,x2=min(xs),max(xs); y1,y2=min(ys),max(ys)
        mouth_roi = result[y1:y2, x1:x2].copy()
        lip_open = 0.3 + 0.7*math.sin(frame_idx*2*math.pi/total_frames)
        new_h = max(1,int((y2-y1)*(1+lip_open*0.5)))
        mouth_roi=cv2.resize(mouth_roi,(x2-x1,new_h))
        y_off=(y2-y1-new_h)//2
        result[y1+y_off:y1+y_off+new_h,x1:x2]=mouth_roi[:new_h,:]

        # Eyes
        for eye_idxs in [(33,133),(362,263)]:
            x_start,y_start=landmarks[eye_idxs[0]]
            x_end,y_end=landmarks[eye_idxs[1]]
            eye_roi=result[y_start:y_end,x_start:x_end].copy()
            blink=max(0,math.sin(frame_idx*4*math.pi/total_frames))
            new_h=max(1,int((y_end-y_start)*(1-blink*0.5)))
            if eye_roi.shape[0]>0 and new_h>0:
                eye_roi=cv2.resize(eye_roi,(eye_roi.shape[1],new_h))
                yoff=(y_end-y_start-new_h)//2
                result[y_start+yoff:y_start+yoff+new_h,x_start:x_end]=eye_roi[:new_h,:]

        return result

    def create_frames(self,img_path, keyframes=60):
        base_img=cv2.imread(img_path)
        base_img=cv2.resize(base_img,(512,512))
        landmarks=self.detect_landmarks(base_img)
        os.makedirs('/content/animation_frames',exist_ok=True)
        frames=[]
        for i in range(keyframes):
            frame=self.deform_face(base_img,landmarks,i,keyframes)
            frame_path=f'/content/animation_frames/frame_{i:04d}.jpg'
            cv2.imwrite(frame_path,frame)
            frames.append(frame_path)
        return frames

# ---------------- GENERATE KEYFRAMES ----------------
animator=BinderFaceAnimator()
keyframes = animator.create_frames('/content/face.jpg', keyframes=60)

# ---------------- INTERPOLATE FRAMES ----------------
# Use RIFE interpolation if available
try:
    from rife_ncnn_vulkan import RIFE
    rife = RIFE()
    print("⚡ Using RIFE for AI frame interpolation...")
    interp_frames = rife.interpolate(keyframes, fps_out=30)  # 30 FPS final
except:
    print("⚡ Falling back to OpenCV linear interpolation...")
    frames=[]
    for i in range(len(keyframes)-1):
        img1=cv2.imread(keyframes[i])
        img2=cv2.imread(keyframes[i+1])
        for alpha in np.linspace(0,1,5):  # 5 interpolated frames
            inter=cv2.addWeighted(img1,1-alpha,img2,alpha,0)
            frame_path=f"/content/animation_frames/frame_interp_{i:04d}_{int(alpha*10):02d}.jpg"
            cv2.imwrite(frame_path,inter)
            frames.append(frame_path)

# ---------------- GENERATE VIDEO ----------------
video_path='/content/binder_face_1min.mp4'
!ffmpeg -y -framerate 30 -pattern_type glob -i '/content/animation_frames/*.jpg' \
                                               -c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast "{video_path}" 2>/dev/null
print(f"✅ Video created: {video_path}")

# ---------------- GENERATE VOICE ----------------
voice_file="voice.wav"
if ELEVENLABS_AVAILABLE:
    api_key = os.environ.get("ELEVENLABS_API_KEY","")
    if api_key:
        user = ElevenLabsUser(api_key)
        voice = user.get_voices()[0]
        voice.generate_and_save(expression_text,voice_file)
    else:
        tts=gTTS(text=expression_text,lang='en',slow=False)
        tts.save(voice_file)
else:
    tts=gTTS(text=expression_text,lang='en',slow=False)
    tts.save(voice_file)

# ---------------- CREATE ZIP ----------------
zip_path="/content/binder_1min_package.zip"
with zipfile.ZipFile(zip_path,"w") as zf:
    zf.write(video_path,"binder_face_1min.mp4")
    zf.write(voice_file,"voice.wav")
    zf.write('/content/face.jpg','face.jpg')
print(f"✅ Binder 1-min package ready: {zip_path}")
files.download(zip_path)
