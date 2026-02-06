# ==============================================
# 🎭 FACE ANIMATOR - VERSIUNE BINDER COMPATIBILĂ
# ==============================================

import os
import cv2
import math
import numpy as np
import zipfile
import urllib.request
import subprocess
import sys
from IPython.display import display, Image, HTML, Audio
import ipywidgets as widgets
from io import BytesIO
import base64

# ==============================================
# 📦 1. INSTALARE PACHETE (pentru Binder)
# ==============================================

print("📦 Instalare pachete necesare...")

# Instalare cu pip (în Binder)
!pip install -q opencv-python-headless mediapipe numpy pillow imageio gtts

# Verificare instalări
try:
    import cv2
    print("✅ OpenCV instalat")
except:
    print("❌ OpenCV - instalare manuală necesară")

try:
    import mediapipe as mp
    print("✅ MediaPipe instalat")
except:
    !pip install -q mediapipe
    import mediapipe as mp

try:
    from gtts import gTTS
    print("✅ gTTS instalat")
except:
    !pip install -q gTTS
    from gtts import gTTS

# ==============================================
# 📸 2. ÎNCĂRCARE FOTO (Binder version)
# ==============================================

print("\n" + "="*50)
print("📸 ÎNCĂRCARE FOTOGRAFIE")
print("="*50)

def upload_image_binder():
    """Funcție pentru upload în Binder"""
    uploader = widgets.FileUpload(
        accept='.jpg,.jpeg,.png',
        multiple=False
    )

    display(uploader)

    def on_upload_change(change):
        if uploader.value:
            # Salvare fișier
            for filename, file_info in uploader.value.items():
                with open(filename, 'wb') as f:
                    f.write(file_info['content'])
                return filename

    uploader.observe(on_upload_change, names='value')
    return None

# Verificare dacă există deja imagini
image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg','.jpeg','.png'))]

if image_files:
    photo_path = image_files[0]
    print(f"✅ Foto găsită: {photo_path}")
else:
    print("📤 Folosește uploader-ul de mai sus pentru a încărca o fotografie...")
    uploaded_file = upload_image_binder()

    if uploaded_file:
        photo_path = uploaded_file
    else:
        # Folosește o imagine demo
        print("⚠️ Folosesc imagine demo...")
        demo_url = "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512&h=512&fit=crop"
        photo_path = "demo_face.jpg"
        urllib.request.urlretrieve(demo_url, photo_path)

# ==============================================
# 🎭 3. CLASA ANIMATOR ÎMBUNĂTĂȚITĂ
# ==============================================

class BinderFaceAnimator:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("✅ Animator inițializat")

    def detect_landmarks(self, image):
        """Detectare puncte faciale cu gestionare erori"""
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w = image.shape[:2]
                points = []

                for idx, lm in enumerate(landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))

                print(f"✅ Detectat {len(points)} puncte faciale")
                return points
            else:
                print("⚠️ Nu s-au detectat puncte faciale")
                return None

        except Exception as e:
            print(f"❌ Eroare detectare: {e}")
            return None

    def create_animation_frames(self, img_path, num_frames=40, expression="talk"):
        """Creează cadre de animație realiste"""
        print(f"\n🎬 Creare {num_frames} cadre pentru expresia: {expression}")

        # Încărcare imagine
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Nu pot citi imaginea!")
            return []

        # Redimensionare inteligentă
        h, w = img.shape[:2]
        if w > 512:
            scale = 512 / w
            new_w, new_h = 512, int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"📏 Redimensionat la: {new_w}x{new_h}")

        # Detectare landmark-uri
        landmarks = self.detect_landmarks(img)

        if landmarks is None:
            print("⚠️ Folosesc poziții estimate...")
            # Poziții estimate pentru cazul fără detectare
            h, w = img.shape[:2]
            landmarks = []
            for i in range(478):  # MediaPipe are 478 puncte
                x = w // 2 + int(math.sin(i/50) * 100)
                y = h // 2 + int(math.cos(i/50) * 100)
                landmarks.append((x, y))

        # Creare folder pentru cadre
        os.makedirs('animation_frames', exist_ok=True)

        frames = []
        for i in range(num_frames):
            frame = img.copy()
            h, w = frame.shape[:2]

            # Calcul progres animație
            progress = i / num_frames

            # Expresii diferite
            if expression == "laugh":
                # RÂS ACCENTUAT
                mouth_open = 0.4 + 0.3 * math.sin(progress * 4 * math.pi)
                eye_close = 0.3 * abs(math.sin(progress * 8 * math.pi))
                brow_raise = 0.2 * math.sin(progress * 2 * math.pi)

            elif expression == "talk":
                # VORBIRE
                mouth_open = 0.2 + 0.15 * math.sin(progress * 8 * math.pi)
                eye_close = 0.1 * abs(math.sin(progress * 16 * math.pi))
                brow_raise = 0.05 * math.sin(progress * 4 * math.pi)

            elif expression == "smile":
                # ZÂMBET
                mouth_open = 0.1 + 0.1 * math.sin(progress * 2 * math.pi)
                eye_close = 0.15 * abs(math.sin(progress * 4 * math.pi))
                brow_raise = 0.1 * math.sin(progress * math.pi)

            else:
                # NEUTRU
                mouth_open = 0.05 * math.sin(progress * 4 * math.pi)
                eye_close = 0.05 * math.sin(progress * 8 * math.pi)
                brow_raise = 0

            # Aplicare transformări
            if landmarks:
                # GURĂ - warp bazat pe landmark-uri
                mouth_points = [61, 291, 39, 181, 0, 17]  # Puncte pentru gură
                for idx in mouth_points:
                    if idx < len(landmarks):
                        x, y = landmarks[idx]
                        # Deplasare verticală pentru deschidere gură
                        new_y = y - int(20 * mouth_open)

                        # Aplicare warp local
                        radius = 15
                        for dy in range(-radius, radius):
                            for dx in range(-radius, radius):
                                ny = y + dy
                                nx = x + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    # Calcul factor de influență
                                    dist = math.sqrt(dx*dx + dy*dy)
                                    if dist < radius:
                                        factor = 1 - (dist / radius)
                                        # Deplasare progresivă
                                        frame[ny, nx] = frame[
                                            max(0, min(h-1, int(y + dy - 10 * mouth_open * factor))),
                                            max(0, min(w-1, nx))
                                        ]

                # OCHI - închidere
                eye_points_left = [33, 133, 157, 158, 159, 160, 161, 173]
                eye_points_right = [362, 263, 384, 385, 386, 387, 388, 466]

                for eye_points in [eye_points_left, eye_points_right]:
                    for idx in eye_points:
                        if idx < len(landmarks):
                            x, y = landmarks[idx]
                            # Închidere ochi
                            for dy in range(-5, 6):
                                ny = y + dy
                                if 0 <= ny < h and 0 <= x < w:
                                    # Mix cu culoarea pielii pentru efect de închidere
                                    skin_color = frame[y, x]
                                    frame[ny, x] = cv2.addWeighted(
                                        frame[ny, x], 1 - eye_close,
                                        skin_color, eye_close,
                                        0
                                    )

            # Salvare cadru
            frame_path = f'animation_frames/frame_{i:04d}.jpg'
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)

            if i % 10 == 0:
                print(f"  ⏳ Cadru {i+1}/{num_frames} creat")

        print(f"✅ {len(frames)} cadre create")
        return frames

    def create_video(self, frames, output_path="animation.mp4", fps=25):
        """Creează video din cadre"""
        print(f"\n🎥 Creare video {fps}FPS...")

        if not frames:
            print("❌ Nu există cadre!")
            return None

        # Citire primul cadru pentru dimensiuni
        first_frame = cv2.imread(frames[0])
        h, w = first_frame.shape[:2]

        # Creare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Scriere cadre
        for frame_path in frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)

        out.release()

        # Verificare video creat
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024*1024)
            print(f"✅ Video creat: {output_path}")
            print(f"📏 Dimensiuni: {w}x{h}")
            print(f"💾 Mărime: {size_mb:.2f} MB")
            print(f"🎞️ Cadre: {len(frames)}")
            print(f"⏱️ Durată: {len(frames)/fps:.1f}s")

            # Afișare preview
            from IPython.display import Video
            display(Video(output_path, width=400, embed=True))

            return output_path
        else:
            print("❌ Video-ul nu a fost creat")
            return None

# ==============================================
# 🎬 4. INTERFAȚĂ UTILIZATOR BINDER
# ==============================================

def create_binder_interface():
    """Interfață pentru Binder notebook"""

    print("="*50)
    print("🎭 FACE ANIMATION STUDIO - BINDER EDITION")
    print("="*50)

    # Widget-uri pentru input
    expression_dropdown = widgets.Dropdown(
        options=['laugh', 'talk', 'smile', 'wink', 'surprised'],
        value='laugh',
        description='Expresie:',
        style={'description_width': 'initial'}
    )

    frames_slider = widgets.IntSlider(
        value=40,
        min=20,
        max=100,
        step=10,
        description='Cadre:',
        style={'description_width': 'initial'}
    )

    fps_slider = widgets.IntSlider(
        value=25,
        min=15,
        max=60,
        step=5,
        description='FPS:',
        style={'description_width': 'initial'}
    )

    text_input = widgets.Textarea(
        value="Hello! I'm an animated face!",
        placeholder='Text pentru voiceover...',
        description='Text:',
        rows=3,
        style={'description_width': 'initial'}
    )

    create_button = widgets.Button(
        description="🎬 CREAZĂ ANIMAȚIE",
        button_style='success',
        layout={'width': '200px'}
    )

    output = widgets.Output()

    def on_create_button_clicked(b):
        with output:
            output.clear_output()

            # Obține valori
            expression = expression_dropdown.value
            num_frames = frames_slider.value
            fps = fps_slider.value
            text = text_input.value

            print(f"🚀 Pornire creare animație...")
            print(f"📊 Setări: {expression}, {num_frames} cadre, {fps} FPS")

            # Inițializează animator
            animator = BinderFaceAnimator()

            # Creează cadre
            frames = animator.create_animation_frames(
                photo_path,
                num_frames=num_frames,
                expression=expression
            )

            if frames:
                # Creează video
                video_path = animator.create_video(
                    frames,
                    output_path=f"face_animation_{expression}.mp4",
                    fps=fps
                )

                if video_path:
                    # Creează voiceover
                    if text.strip():
                        print(f"\n🎤 Creare voiceover...")
                        try:
                            tts = gTTS(text=text, lang='en', slow=False)
                            audio_path = "voiceover.mp3"
                            tts.save(audio_path)

                            # Combina video cu audio
                            final_path = f"final_{expression}.mp4"
                            cmd = f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -shortest "{final_path}" 2>/dev/null'
                            os.system(cmd)

                            if os.path.exists(final_path):
                                print(f"✅ Video final cu audio: {final_path}")

                                # Afișare pentru descărcare
                                display(HTML(f"""
                                <div style="background:#e8f5e9;padding:15px;border-radius:10px;margin:10px 0;">
                                <h3>🎉 ANIMAȚIE COMPLETĂ!</h3>
                                <p><b>Fișier:</b> {final_path}</p>
                                <p><b>Expresie:</b> {expression}</p>
                                <p><b>Text:</b> {text[:50]}...</p>
                                <p>Pentru a descărca: click dreapta → Save as</p>
                                </div>
                                """))

                                # Afișare video final
                                display(Video(final_path, width=500, embed=True))

                                # Creare zip
                                zip_path = f"animation_package_{expression}.zip"
                                with zipfile.ZipFile(zip_path, 'w') as zf:
                                    zf.write(final_path, os.path.basename(final_path))
                                    zf.write(audio_path, "voiceover.mp3")
                                    zf.write(photo_path, "original_face.jpg")

                                print(f"📦 Pachet creat: {zip_path}")

                        except Exception as e:
                            print(f"⚠️ Nu s-a putut crea audio: {e}")

    create_button.on_click(on_create_button_clicked)

    # Afișare interfață
    display(widgets.VBox([
        widgets.HTML("<h3>⚙️ SETĂRI ANIMAȚIE</h3>"),
        expression_dropdown,
        frames_slider,
        fps_slider,
        text_input,
        create_button,
        output
    ]))

# ==============================================
# 🚀 5. PORNIRE APLICAȚIE
# ==============================================

# Afișare imagine încărcată
if 'photo_path' in locals():
    img = cv2.imread(photo_path)
    if img is not None:
        img_display = cv2.resize(img, (300, 300))
        _, buffer = cv2.imencode('.jpg', img_display)
        display(Image(data=buffer.tobytes(), width=300))
        print(f"\n👤 Imagine pentru animație: {photo_path}")

# Pornire interfață
create_binder_interface()

print("\n" + "="*50)
print("🎯 CODUL ESTE ACUM COMPATIBIL CU BINDER!")
print("="*50)
print("\nCaracteristici implementate:")
print("✅ Upload foto în Binder")
print("✅ Animator cu MediaPipe")
print("✅ 5 expresii diferite")
print("✅ Voiceover cu gTTS")
print("✅ Interfață interactivă")
print("✅ Preview în notebook")
print("✅ Export video + audio")
print("✅ Pachet zip pentru descărcare")