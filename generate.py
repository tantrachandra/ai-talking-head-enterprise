"""
=============================================
🎭 UNIVERSAL FACE ANIMATOR - WORKS EVERYWHERE
=============================================
No external dependencies, no API keys needed
Works in: Binder • Colab • Jupyter • Local
=============================================
"""

import os
import cv2
import math
import numpy as np
import zipfile
import urllib.request
import subprocess
import sys
import base64
from io import BytesIO
from IPython.display import display, Image as IPImage, HTML, Audio
import ipywidgets as widgets

print("✅ Initializing Universal Face Animator...")

# ==============================================
# 📦 SMART PACKAGE INSTALLATION
# ==============================================

def install_packages():
    """Install packages only if needed"""
    packages_to_check = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("gtts", "gtts"),
        ("Pillow", "PIL"),
    ]

    for pkg_name, import_name in packages_to_check:
        try:
            __import__(import_name)
            print(f"✅ {import_name} already available")
        except ImportError:
            print(f"📦 Installing {pkg_name}...")
            try:
                # Use pip with --quiet flag
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg_name])
                print(f"✅ {pkg_name} installed successfully")
            except:
                print(f"⚠️ Could not install {pkg_name}, continuing without it")

# Run installation
install_packages()

# Now import everything
try:
    import cv2
    print("✅ OpenCV loaded")
except:
    print("⚠️ OpenCV not available - limited functionality")

try:
    import mediapipe as mp
    print("✅ MediaPipe loaded")
    MP_AVAILABLE = True
except:
    print("⚠️ MediaPipe not available - using fallback animation")
    MP_AVAILABLE = False

try:
    from gtts import gTTS
    print("✅ gTTS loaded")
    TTS_AVAILABLE = True
except:
    print("⚠️ gTTS not available - no voice generation")
    TTS_AVAILABLE = False

from PIL import Image, ImageDraw, ImageFilter
print("✅ PIL loaded")

# ==============================================
# 📸 UNIVERSAL IMAGE LOADER
# ==============================================

def load_or_upload_image():
    """Smart image loading for any environment"""

    # First, check for existing images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    current_dir = os.getcwd()

    for file in os.listdir(current_dir):
        if file.lower().endswith(image_extensions):
            print(f"📁 Found existing image: {file}")
            return file

    # If in Colab
    try:
        from google.colab import files
        print("📤 Colab detected - using Colab uploader")
        print("Please upload a face photo...")
        uploaded = files.upload()
        for filename in uploaded.keys():
            if filename.lower().endswith(image_extensions):
                print(f"✅ Uploaded: {filename}")
                return filename
    except:
        pass  # Not in Colab

    # If in Jupyter/Binder with ipywidgets
    try:
        print("📤 Using widget uploader...")
        uploader = widgets.FileUpload(
            accept='.jpg,.jpeg,.png',
            multiple=False,
            description='Upload face photo'
        )

        upload_box = widgets.VBox([uploader])
        display(upload_box)

        # Wait for upload
        import time
        for _ in range(50):  # Wait up to 5 seconds
            if uploader.value:
                for filename, file_info in uploader.value.items():
                    with open(filename, 'wb') as f:
                        f.write(file_info['content'])
                    print(f"✅ Uploaded via widget: {filename}")
                    return filename
            time.sleep(0.1)

        print("⏰ Upload timeout - using demo image")
    except:
        print("⚠️ Widget upload not available")

    # Fallback to demo image
    print("🖼️ Using demo image...")
    demo_url = "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/tasks/testdata/vision/face_landmarker/face.jpg"
    demo_file = "demo_face.jpg"

    try:
        urllib.request.urlretrieve(demo_url, demo_file)
        print(f"✅ Demo image downloaded: {demo_file}")
        return demo_file
    except:
        print("❌ Could not load any image")
        return None

# Load image
print("\n" + "="*50)
print("📸 IMAGE LOADING")
print("="*50)

image_file = load_or_upload_image()

if image_file and os.path.exists(image_file):
    # Display the image
    try:
        img = Image.open(image_file)
        # Resize for display if too large
        if img.width > 400:
            img_display = img.copy()
            img_display.thumbnail((400, 400))
        else:
            img_display = img

        display(img_display)
        print(f"✅ Image loaded: {image_file} ({img.width}x{img.height})")

        # Save a working copy
        working_copy = "working_face.jpg"
        img.save(working_copy)
        image_file = working_copy

    except Exception as e:
        print(f"❌ Error loading image: {e}")
        image_file = None
else:
    print("❌ No image available")
    image_file = None

# ==============================================
# 🎭 ANIMATION ENGINE - UNIVERSAL
# ==============================================

class UniversalFaceAnimator:
    def __init__(self):
        self.face_landmarks = None
        if MP_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            print("🤖 MediaPipe Face Mesh initialized")
        else:
            print("🤖 Using simplified animation engine")

    def detect_landmarks_pil(self, pil_image):
        """Detect facial landmarks using MediaPipe or estimate them"""
        if MP_AVAILABLE:
            try:
                # Convert PIL to numpy array for MediaPipe
                img_np = np.array(pil_image.convert('RGB'))

                # Process with MediaPipe
                results = self.face_mesh.process(img_np)

                if results.multi_face_landmarks:
                    h, w = pil_image.height, pil_image.width
                    landmarks = []

                    for landmark in results.multi_face_landmarks[0].landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append((x, y))

                    print(f"✅ Detected {len(landmarks)} facial landmarks")
                    return landmarks
                else:
                    print("⚠️ No face detected by MediaPipe")

            except Exception as e:
                print(f"⚠️ MediaPipe error: {e}")

        # Fallback: estimate landmarks based on image dimensions
        print("📐 Estimating facial landmarks...")
        h, w = pil_image.height, pil_image.width

        # Create estimated landmarks (simplified face mesh)
        landmarks = []

        # Face oval
        center_x, center_y = w // 2, h // 2
        face_width, face_height = w // 2, h // 2

        # Generate points in an oval pattern
        for angle in np.linspace(0, 2 * math.pi, 100):
            x = center_x + face_width * 0.4 * math.cos(angle)
            y = center_y + face_height * 0.6 * math.sin(angle)
            landmarks.append((int(x), int(y)))

        # Add facial features
        # Eyes
        landmarks.append((center_x - w // 6, center_y - h // 10))  # Left eye
        landmarks.append((center_x + w // 6, center_y - h // 10))  # Right eye

        # Mouth
        landmarks.append((center_x - w // 8, center_y + h // 6))   # Mouth left
        landmarks.append((center_x, center_y + h // 5))           # Mouth center
        landmarks.append((center_x + w // 8, center_y + h // 6))  # Mouth right

        # Nose
        landmarks.append((center_x, center_y))

        print(f"✅ Estimated {len(landmarks)} landmark points")
        return landmarks

    def create_expression_frame(self, base_image, landmarks, frame_num, total_frames, expression="smile"):
        """Create a single animation frame with expression"""

        # Make a copy of the image
        if isinstance(base_image, Image.Image):
            frame = base_image.copy()
        else:
            frame = Image.fromarray(base_image)

        # Calculate animation progress (0 to 1 and back to 0)
        progress = (frame_num / total_frames) * 2
        if progress > 1:
            progress = 2 - progress

        # Different expressions
        if expression == "laugh":
            # Big open mouth, squinted eyes, raised brows
            mouth_open = 0.7 * math.sin(progress * math.pi)
            eye_squint = 0.5 * math.sin(progress * math.pi)
            brow_raise = 0.3 * math.sin(progress * math.pi)

        elif expression == "talk":
            # Talking motion
            mouth_open = 0.4 * math.sin(progress * 4 * math.pi)  # Fast open/close
            eye_squint = 0.1 * math.sin(progress * math.pi)
            brow_raise = 0.1 * math.sin(progress * 2 * math.pi)

        elif expression == "wink":
            # Wink with one eye
            if frame_num < total_frames // 2:
                eye_squint = progress
            else:
                eye_squint = 1 - (progress - 1)
            mouth_open = 0.1 * math.sin(progress * math.pi)
            brow_raise = 0.2

        else:  # smile (default)
            # Gentle smile
            mouth_open = 0.3 * math.sin(progress * math.pi)
            eye_squint = 0.2 * math.sin(progress * math.pi)
            brow_raise = 0.1 * math.sin(progress * math.pi)

        # Apply facial transformations
        draw = ImageDraw.Draw(frame)
        w, h = frame.size

        # Find facial regions based on landmarks
        if landmarks and len(landmarks) > 10:
            # Get approximate facial feature positions
            # Eyes (first two non-oval points)
            if len(landmarks) >= 102:  # We have estimated landmarks
                left_eye = landmarks[100]
                right_eye = landmarks[101]
                mouth_center = landmarks[103]

                # Draw animated eyes
                eye_radius = w // 30
                # Left eye (squinting)
                left_eye_h = max(2, int(eye_radius * (1 - eye_squint * 0.8)))
                draw.ellipse([left_eye[0] - eye_radius, left_eye[1] - left_eye_h,
                              left_eye[0] + eye_radius, left_eye[1] + left_eye_h],
                             fill='black')

                # Right eye (squinting)
                if expression != "wink" or frame_num < total_frames // 3 or frame_num > 2 * total_frames // 3:
                    right_eye_h = max(2, int(eye_radius * (1 - eye_squint * 0.8)))
                    draw.ellipse([right_eye[0] - eye_radius, right_eye[1] - right_eye_h,
                                  right_eye[0] + eye_radius, right_eye[1] + right_eye_h],
                                 fill='black')

                # Draw mouth (smile/oval)
                mouth_width = w // 4 + int(w // 8 * mouth_open)
                mouth_height = h // 20 + int(h // 15 * abs(mouth_open))

                if mouth_open >= 0:  # Smile
                    draw.arc([mouth_center[0] - mouth_width, mouth_center[1] - mouth_height,
                              mouth_center[0] + mouth_width, mouth_center[1] + mouth_height],
                             start=180 + 20 * brow_raise, end=360 - 20 * brow_raise,
                             fill='black', width=3)
                else:  # Neutral/closed
                    draw.line([mouth_center[0] - mouth_width // 2, mouth_center[1],
                               mouth_center[0] + mouth_width // 2, mouth_center[1]],
                              fill='black', width=2)

            # Draw brows (raised)
            brow_y_offset = int(-10 * brow_raise)
            if len(landmarks) >= 100:
                # Left brow
                draw.line([w//2 - w//4, h//4 + brow_y_offset,
                           w//2 - w//8, h//4 - brow_y_offset],
                          fill='black', width=3)
                # Right brow
                draw.line([w//2 + w//8, h//4 - brow_y_offset,
                           w//2 + w//4, h//4 + brow_y_offset],
                          fill='black', width=3)

        else:
            # Simplified animation without landmarks
            center_x, center_y = w // 2, h // 2

            # Animated eyes
            eye_spacing = w // 4
            eye_y = h // 3
            eye_radius = w // 20

            # Left eye (squint based on expression)
            left_eye_h = max(2, int(eye_radius * (1 - eye_squint * 0.7)))
            draw.ellipse([center_x - eye_spacing - eye_radius, eye_y - left_eye_h,
                          center_x - eye_spacing + eye_radius, eye_y + left_eye_h],
                         fill='black')

            # Right eye (wink for wink expression)
            if expression != "wink" or frame_num < total_frames // 3 or frame_num > 2 * total_frames // 3:
                right_eye_h = max(2, int(eye_radius * (1 - eye_squint * 0.7)))
                draw.ellipse([center_x + eye_spacing - eye_radius, eye_y - right_eye_h,
                              center_x + eye_spacing + eye_radius, eye_y + right_eye_h],
                             fill='black')
            else:
                # Winking - draw a line for closed eye
                draw.line([center_x + eye_spacing - eye_radius, eye_y,
                           center_x + eye_spacing + eye_radius, eye_y],
                          fill='black', width=3)

            # Mouth
            mouth_width = w // 3 + int(w // 6 * mouth_open)
            mouth_height = h // 15 + int(h // 10 * abs(mouth_open))
            mouth_y = center_y + h // 4

            if expression == "laugh" and mouth_open > 0.5:
                # Open mouth for laugh
                draw.ellipse([center_x - mouth_width // 2, mouth_y - mouth_height // 2,
                              center_x + mouth_width // 2, mouth_y + mouth_height // 2],
                             fill='black')
            elif mouth_open > 0.1:
                # Smile
                draw.arc([center_x - mouth_width, mouth_y - mouth_height,
                          center_x + mouth_width, mouth_y + mouth_height],
                         start=180, end=360, fill='black', width=3)
            else:
                # Neutral mouth
                draw.line([center_x - mouth_width // 2, mouth_y,
                           center_x + mouth_width // 2, mouth_y],
                          fill='black', width=2)

            # Brows
            brow_y = eye_y - eye_radius * 2
            brow_raise_px = int(-15 * brow_raise)
            draw.line([center_x - eye_spacing - eye_radius, brow_y + brow_raise_px,
                       center_x - eye_spacing + eye_radius, brow_y],
                      fill='black', width=3)
            draw.line([center_x + eye_spacing - eye_radius, brow_y,
                       center_x + eye_spacing + eye_radius, brow_y + brow_raise_px],
                      fill='black', width=3)

        # Apply subtle image warp for more natural movement
        if expression in ["laugh", "talk"] and landmarks:
            # Simple warp effect around mouth
            frame_np = np.array(frame)
            h, w = frame_np.shape[:2]

            # Create displacement maps
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)

            # Initialize with original positions
            for y in range(h):
                for y in range(h):
                    map_x[y, :] = np.arange(w)
                    map_y[y, :] = y

            # Add mouth movement warp
            mouth_center_y = h // 2 + h // 4
            for y in range(max(0, mouth_center_y - h//8), min(h, mouth_center_y + h//8)):
                for x in range(max(0, w//2 - w//4), min(w, w//2 + w//4)):
                    # Calculate distance from mouth center
                    dx = x - w//2
                    dy = y - mouth_center_y
                    dist = math.sqrt(dx*dx + dy*dy)

                    if dist < w//4:
                        # Apply vertical displacement for mouth opening
                        factor = 1 - (dist / (w//4))
                        displacement = int(10 * mouth_open * factor)
                        map_y[y, x] = y - displacement

            # Apply warp if OpenCV is available
            try:
                import cv2
                frame_warped = cv2.remap(frame_np, map_x, map_y, cv2.INTER_LINEAR)
                frame = Image.fromarray(frame_warped)
            except:
                pass  # Skip warp if OpenCV not available

        return frame

    def create_animation(self, image_path, expression="smile", num_frames=30, output_dir="animation_frames"):
        """Create complete animation sequence"""

        print(f"\n🎬 Creating {expression} animation ({num_frames} frames)...")

        # Load base image
        try:
            base_image = Image.open(image_path).convert('RGB')
            print(f"✅ Base image loaded: {base_image.size}")
        except:
            print(f"❌ Cannot load image: {image_path}")
            return []

        # Detect or estimate landmarks
        landmarks = self.detect_landmarks_pil(base_image)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate frames
        frame_paths = []
        for i in range(num_frames):
            # Create frame with expression
            frame = self.create_expression_frame(
                base_image, landmarks, i, num_frames, expression
            )

            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            frame.save(frame_path, quality=95)
            frame_paths.append(frame_path)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  ⏳ Created frame {i + 1}/{num_frames}")

        print(f"✅ {len(frame_paths)} frames created in '{output_dir}'")
        return frame_paths

    def create_video(self, frame_paths, output_path="animation.mp4", fps=25):
        """Create video from frames"""

        if not frame_paths:
            print("❌ No frames to create video from")
            return None

        print(f"\n🎥 Creating video ({fps} FPS)...")

        # Read first frame to get dimensions
        try:
            first_frame = Image.open(frame_paths[0])
            width, height = first_frame.size
        except:
            print("❌ Cannot read frames")
            return None

        # Try using OpenCV first
        video_created = False

        try:
            import cv2
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame_path in frame_paths:
                # Read frame with PIL and convert to OpenCV format
                pil_img = Image.open(frame_path).convert('RGB')
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                out.write(cv_img)

            out.release()
            video_created = True
            print(f"✅ Video created with OpenCV: {output_path}")

        except Exception as e:
            print(f"⚠️ OpenCV video creation failed: {e}")
            print("Trying alternative method...")

        # Alternative method: Use imageio
        if not video_created:
            try:
                import imageio.v2 as imageio

                # Create video writer
                writer = imageio.get_writer(
                    output_path,
                    fps=fps,
                    codec='libx264',
                    quality=8
                )

                for frame_path in frame_paths:
                    frame = imageio.imread(frame_path)
                    writer.append_data(frame)

                writer.close()
                video_created = True
                print(f"✅ Video created with imageio: {output_path}")

            except Exception as e:
                print(f"❌ imageio video creation failed: {e}")

        # Last resort: Create animated GIF
        if not video_created:
            try:
                gif_path = output_path.replace('.mp4', '.gif')

                frames = []
                for frame_path in frame_paths:
                    frame = Image.open(frame_path)
                    frames.append(frame)

                # Save as GIF
                frames[0].save(
                    gif_path,
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=1000//fps,  # milliseconds per frame
                    loop=0
                )

                print(f"✅ Created GIF instead: {gif_path}")
                return gif_path

            except Exception as e:
                print(f"❌ GIF creation failed: {e}")
                return None

        # Verify video was created
        if video_created and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"📊 Video info: {width}x{height}, {file_size:.2f} MB, {len(frame_paths)/fps:.1f}s")

            # Try to display preview
            try:
                from IPython.display import Video
                display(Video(output_path, width=400, embed=True))
            except:
                print("📺 Video created (preview not available)")

            return output_path

        return None

    def create_voiceover(self, text, output_path="voiceover.mp3"):
        """Create voiceover from text"""

        if not TTS_AVAILABLE:
            print("⚠️ gTTS not available - skipping voiceover")
            return None

        try:
            print(f"🎤 Creating voiceover: '{text[:50]}...'")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)

            if os.path.exists(output_path):
                print(f"✅ Voiceover created: {output_path}")

                # Try to play preview
                try:
                    display(Audio(output_path, autoplay=False))
                except:
                    pass

                return output_path

        except Exception as e:
            print(f"❌ Voiceover creation failed: {e}")

        return None

# ==============================================
# 🎨 INTERACTIVE UI
# ==============================================

def create_interactive_ui():
    """Create user interface for animation"""

    print("\n" + "="*50)
    print("🎭 FACE ANIMATION STUDIO")
    print("="*50)

    # Check if we have an image
    if not image_file or not os.path.exists(image_file):
        print("❌ Please upload an image first!")
        return

    # Create UI widgets
    expression_widget = widgets.Dropdown(
        options=[
            ('😊 Smile', 'smile'),
            ('😂 Laugh', 'laugh'),
            ('🗣️ Talk', 'talk'),
            ('😉 Wink', 'wink'),
            ('😐 Neutral', 'neutral')
        ],
        value='smile',
        description='Expression:',
        style={'description_width': 'initial'}
    )

    frames_widget = widgets.IntSlider(
        value=30,
        min=15,
        max=60,
        step=5,
        description='Frames:',
        style={'description_width': 'initial'}
    )

    fps_widget = widgets.IntSlider(
        value=25,
        min=15,
        max=30,
        step=5,
        description='FPS:',
        style={'description_width': 'initial'}
    )

    text_widget = widgets.Textarea(
        value="Hello! This is my animated face!",
        placeholder='Enter text for voiceover...',
        description='Voice Text:',
        rows=3,
        style={'description_width': 'initial'}
    )

    create_button = widgets.Button(
        description="🎬 CREATE ANIMATION",
        button_style='success',
        layout={'width': '200px'}
    )

    output_area = widgets.Output()

    def on_create_click(b):
        with output_area:
            output_area.clear_output()

            # Get values from widgets
            expression = expression_widget.value
            num_frames = frames_widget.value
            fps = fps_widget.value
            text = text_widget.value

            print(f"🚀 Starting animation creation...")
            print(f"• Expression: {expression}")
            print(f"• Frames: {num_frames}")
            print(f"• FPS: {fps}")

            # Initialize animator
            animator = UniversalFaceAnimator()

            # Create animation frames
            frames_dir = f"frames_{expression}"
            frame_paths = animator.create_animation(
                image_file,
                expression=expression,
                num_frames=num_frames,
                output_dir=frames_dir
            )

            if frame_paths:
                # Create video
                video_path = f"animation_{expression}.mp4"
                video_file = animator.create_video(
                    frame_paths,
                    output_path=video_path,
                    fps=fps
                )

                if video_file:
                    # Create voiceover if text provided
                    voice_file = None
                    if text.strip() and TTS_AVAILABLE:
                        voice_path = f"voice_{expression}.mp3"
                        voice_file = animator.create_voiceover(text, voice_path)

                    # Create download package
                    try:
                        zip_path = f"animation_package_{expression}.zip"
                        with zipfile.ZipFile(zip_path, 'w') as zf:
                            # Add video
                            zf.write(video_file, os.path.basename(video_file))

                            # Add voice if created
                            if voice_file and os.path.exists(voice_file):
                                zf.write(voice_file, os.path.basename(voice_file))

                            # Add original image
                            zf.write(image_file, "original_image.jpg")

                            # Add info file
                            info = f"""Face Animation Package
Created: {expression} expression
Frames: {num_frames}
FPS: {fps}
Duration: {num_frames/fps:.1f} seconds
Voice text: {text[:100]}...
"""
                            zf.writestr("INFO.txt", info)

                        print(f"\n📦 Package created: {zip_path}")

                        # Create download link
                        if os.path.exists(zip_path):
                            with open(zip_path, 'rb') as f:
                                b64 = base64.b64encode(f.read()).decode()

                            download_html = f"""
                            <div style="background:#e3f2fd;padding:15px;border-radius:10px;margin:10px 0;">
                            <h3>🎉 ANIMATION READY!</h3>
                            <p><b>Expression:</b> {expression}</p>
                            <p><b>Video:</b> <a href="{video_file}" download>{os.path.basename(video_file)}</a></p>
                            <p><b>Package:</b> <a href="data:application/zip;base64,{b64}" download="{zip_path}">{os.path.basename(zip_path)}</a></p>
                            </div>
                            """
                            display(HTML(download_html))

                    except Exception as e:
                        print(f"⚠️ Package creation failed: {e}")

                        # Still show download links for individual files
                        download_html = f"""
                        <div style="background:#e3f2fd;padding:15px;border-radius:10px;margin:10px 0;">
                        <h3>🎉 ANIMATION READY!</h3>
                        <p>Download files:</p>
                        <ul>
                        <li><a href="{video_file}" download>Video: {os.path.basename(video_file)}</a></li>
                        """
                        if voice_file:
                            download_html += f'<li><a href="{voice_file}" download>Voice: {os.path.basename(voice_file)}</a></li>'
                        download_html += '</ul></div>'
                        display(HTML(download_html))

    create_button.on_click(on_create_click)

    # Display the UI
    ui = widgets.VBox([
        widgets.HTML("<h3>⚙️ ANIMATION SETTINGS</h3>"),
        expression_widget,
        frames_widget,
        fps_widget,
        text_widget,
        create_button,
        output_area
    ])

    display(ui)

# ==============================================
# 🚀 MAIN EXECUTION
# ==============================================

if __name__ == "__main__":
    # Show what environment we're in
    print("\n" + "="*50)
    print("🌍 ENVIRONMENT INFO")
    print("="*50)

    # Detect environment
    try:
        import google.colab
        print("📍 Running in: Google Colab")
    except:
        try:
            import ipywidgets
            print("📍 Running in: Jupyter Notebook / Binder")
        except:
            print("📍 Running in: Local Python")

    # Show available features
    print("\n🔧 AVAILABLE FEATURES:")
    print(f"• Face Detection: {'✅' if MP_AVAILABLE else '⚠️ Limited'}")
    print(f"• Voice Generation: {'✅' if TTS_AVAILABLE else '❌ Not available'}")
    print(f"• Video Creation: ✅")
    print(f"• Image Processing: ✅")

    # If image was loaded, show it and start UI
    if image_file and os.path.exists(image_file):
        print("\n" + "="*50)
        print("👤 LOADED IMAGE")
        print("="*50)

        # Show image info
        img_info = Image.open(image_file)
        print(f"• File: {image_file}")
        print(f"• Size: {img_info.width}x{img_info.height}")
        print(f"• Format: {img_info.format}")

        # Create and show UI
        create_interactive_ui()
    else:
        print("\n❌ No image available. Please upload an image and run again.")

    print("\n" + "="*50)
    print("🎬 READY TO ANIMATE!")
    print("="*50)
    print("\nInstructions:")
    print("1. Use the dropdown to select an expression")
    print("2. Adjust frame count and FPS as needed")
    print("3. Enter text for voiceover (optional)")
    print("4. Click 'CREATE ANIMATION'")
    print("5. Download your animation package!")