# Face Animator - Binder Compatible Version
# No external dependencies, works out of the box

import os
import math
import numpy as np
import zipfile
import urllib.request
import subprocess
import sys
import base64
from io import BytesIO

print("="*60)
print("🎭 FACE ANIMATOR - BINDER EDITION")
print("="*60)

# ==============================================
# 1. CHECK AND INSTALL DEPENDENCIES
# ==============================================

def install_if_needed(package, import_name=None):
    """Install package only if not available"""
    if import_name is None:
        import_name = package

    try:
        __import__(import_name)
        print(f"✅ {import_name} is available")
        return True
    except ImportError:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
            print(f"✅ {package} installed successfully")
            return True
        except:
            print(f"⚠️ Could not install {package}")
            return False

# Install essential packages
print("\n📦 CHECKING DEPENDENCIES...")
install_if_needed("Pillow", "PIL")
install_if_needed("imageio")
install_if_needed("gtts", "gtts")
install_if_needed("opencv-python-headless", "cv2")
install_if_needed("mediapipe")

# Now import everything
try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
    print("✅ PIL loaded")
except:
    PIL_AVAILABLE = False
    print("❌ PIL not available")

try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV loaded")
except:
    CV2_AVAILABLE = False
    print("❌ OpenCV not available")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print("✅ MediaPipe loaded")
except:
    MP_AVAILABLE = False
    print("❌ MediaPipe not available")

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    print("✅ gTTS loaded")
except:
    TTS_AVAILABLE = False
    print("❌ gTTS not available")

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
    print("✅ imageio loaded")
except:
    IMAGEIO_AVAILABLE = False
    print("❌ imageio not available")

# ==============================================
# 2. SIMPLE IMAGE LOADER FOR BINDER
# ==============================================

def load_image_for_binder():
    """Load image in Binder environment"""

    # Check for existing images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for file in os.listdir('.'):
        if file.lower().endswith(image_extensions):
            print(f"📁 Found: {file}")
            return file

    # Try to download a demo image
    print("📥 Downloading demo image...")
    demo_urls = [
        "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/tasks/testdata/vision/face_landmarker/face.jpg",
        "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512&h=512&fit=crop",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/User_icon_2.svg/512px-User_icon_2.svg.png"
    ]

    for i, url in enumerate(demo_urls):
        try:
            demo_file = f"demo_face_{i}.jpg"
            urllib.request.urlretrieve(url, demo_file)
            if os.path.exists(demo_file) and os.path.getsize(demo_file) > 1000:
                print(f"✅ Downloaded demo: {demo_file}")
                return demo_file
        except:
            continue

    # Create a simple face image
    print("🎨 Creating simple face image...")
    img = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(img)

    # Draw a simple face
    # Face oval
    draw.ellipse([100, 100, 412, 412], outline='black', width=2)

    # Eyes
    draw.ellipse([180, 200, 220, 240], fill='black')  # Left eye
    draw.ellipse([292, 200, 332, 240], fill='black')  # Right eye

    # Mouth (smile)
    draw.arc([180, 300, 332, 380], start=180, end=360, fill='black', width=3)

    # Nose
    draw.line([256, 240, 256, 300], fill='black', width=2)

    demo_file = "generated_face.jpg"
    img.save(demo_file)
    print(f"✅ Created: {demo_file}")

    return demo_file

# Load the image
print("\n" + "="*60)
print("🖼️ LOADING IMAGE")
print("="*60)

image_path = load_image_for_binder()

if image_path and os.path.exists(image_path):
    # Display the image
    try:
        img = Image.open(image_path)
        # Resize for display
        max_size = 300
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))

        print(f"📐 Image size: {img.width}x{img.height}")

        # Convert to base64 for display
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        display_html = f"""
        <div style="text-align:center; padding:10px; background:#f5f5f5; border-radius:10px;">
        <h3>👤 Image Loaded</h3>
        <img src="data:image/jpeg;base64,{img_str}" width="250" style="border-radius:10px;">
        <p><b>{image_path}</b> ({img.width}x{img.height})</p>
        </div>
        """

        # Display in notebook
        from IPython.display import display, HTML
        display(HTML(display_html))

    except Exception as e:
        print(f"❌ Error displaying image: {e}")
else:
    print("❌ No image available")

# ==============================================
# 3. SIMPLE FACE ANIMATOR (No Complex Dependencies)
# ==============================================

class SimpleFaceAnimator:
    """Face animator that works everywhere"""

    def __init__(self):
        print("🤖 Simple Face Animator initialized")

    def create_face_landmarks(self, width, height):
        """Create estimated face landmarks"""
        landmarks = []

        # Face oval points
        center_x, center_y = width // 2, height // 2
        face_width, face_height = width // 3, height // 3

        # Generate oval points
        for i in range(50):
            angle = 2 * math.pi * i / 50
            x = center_x + face_width * math.cos(angle)
            y = center_y + face_height * 0.8 * math.sin(angle)
            landmarks.append((int(x), int(y)))

        # Key facial points
        # Eyes
        landmarks.append((center_x - width // 6, center_y - height // 10))  # Left eye center
        landmarks.append((center_x + width // 6, center_y - height // 10))  # Right eye center

        # Mouth
        landmarks.append((center_x - width // 8, center_y + height // 6))   # Mouth left
        landmarks.append((center_x, center_y + height // 5))               # Mouth center
        landmarks.append((center_x + width // 8, center_y + height // 6))  # Mouth right

        # Nose
        landmarks.append((center_x, center_y))

        # Eyebrows
        landmarks.append((center_x - width // 5, center_y - height // 6))  # Left brow
        landmarks.append((center_x + width // 5, center_y - height // 6))  # Right brow

        return landmarks

    def create_animation_frame(self, base_image, frame_num, total_frames, expression="smile"):
        """Create a single animation frame"""

        # Convert to PIL if needed
        if isinstance(base_image, np.ndarray):
            if CV2_AVAILABLE:
                # Convert BGR to RGB
                if len(base_image.shape) == 3 and base_image.shape[2] == 3:
                    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(base_image)
            else:
                img = Image.fromarray(base_image)
        else:
            img = base_image.copy()

        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Calculate animation progress (0 to 1 and back to 0)
        progress = (frame_num / total_frames) * 2
        if progress > 1:
            progress = 2 - progress

        # Expression parameters
        if expression == "laugh":
            mouth_open = 0.8 * math.sin(progress * math.pi)
            eye_squint = 0.6 * math.sin(progress * math.pi)
            brow_raise = 0.4
            cheek_swell = 0.3 * math.sin(progress * math.pi)

        elif expression == "talk":
            mouth_open = 0.4 * math.sin(progress * 4 * math.pi)  # Fast movement
            eye_squint = 0.1
            brow_raise = 0.1 * math.sin(progress * 2 * math.pi)
            cheek_swell = 0.05

        elif expression == "wink":
            if frame_num < total_frames // 2:
                eye_squint = progress
            else:
                eye_squint = 1 - (progress - 1)
            mouth_open = 0.1 * math.sin(progress * math.pi)
            brow_raise = 0.2
            cheek_swell = 0.1

        elif expression == "sad":
            mouth_open = -0.3  # Downturned mouth
            eye_squint = 0.1
            brow_raise = -0.2  # Lowered brows
            cheek_swell = 0

        else:  # smile (default)
            mouth_open = 0.4 * math.sin(progress * math.pi)
            eye_squint = 0.3 * math.sin(progress * math.pi)
            brow_raise = 0.2 * math.sin(progress * math.pi)
            cheek_swell = 0.2 * math.sin(progress * math.pi)

        # Face center
        center_x, center_y = width // 2, height // 2

        # Draw animated face
        # 1. Eyes
        eye_radius = width // 25
        eye_spacing = width // 4

        # Left eye
        left_eye_x = center_x - eye_spacing // 2
        left_eye_y = center_y - height // 6

        # Right eye
        right_eye_x = center_x + eye_spacing // 2
        right_eye_y = center_y - height // 6

        # Eye squinting (height reduction)
        eye_height = max(2, int(eye_radius * (1 - eye_squint * 0.7)))

        # Left eye
        draw.ellipse([left_eye_x - eye_radius, left_eye_y - eye_height,
                      left_eye_x + eye_radius, left_eye_y + eye_height],
                     fill='black')

        # Right eye (wink for wink expression)
        if expression != "wink" or (frame_num > total_frames // 3 and frame_num < 2 * total_frames // 3):
            draw.ellipse([right_eye_x - eye_radius, right_eye_y - eye_height,
                          right_eye_x + eye_radius, right_eye_y + eye_height],
                         fill='black')
        else:
            # Wink - draw a line
            draw.line([right_eye_x - eye_radius, right_eye_y,
                       right_eye_x + eye_radius, right_eye_y],
                      fill='black', width=3)

        # 2. Eyebrows
        brow_length = width // 8
        brow_height = int(10 * brow_raise)

        # Left brow
        draw.line([left_eye_x - brow_length, left_eye_y - eye_radius * 2 + brow_height,
                   left_eye_x + brow_length, left_eye_y - eye_radius * 2],
                  fill='black', width=3)

        # Right brow
        draw.line([right_eye_x - brow_length, right_eye_y - eye_radius * 2,
                   right_eye_x + brow_length, right_eye_y - eye_radius * 2 + brow_height],
                  fill='black', width=3)

        # 3. Mouth
        mouth_width = width // 4 + int(width // 6 * abs(mouth_open))
        mouth_height = height // 20 + int(height // 15 * abs(mouth_open))
        mouth_y = center_y + height // 4

        if mouth_open > 0.1:
            # Smile or open mouth
            if expression == "laugh" and mouth_open > 0.5:
                # Open mouth for laugh
                draw.ellipse([center_x - mouth_width // 2, mouth_y - mouth_height // 2,
                              center_x + mouth_width // 2, mouth_y + mouth_height // 2],
                             fill='black')
            else:
                # Smile
                start_angle = 180 + 20 * brow_raise
                end_angle = 360 - 20 * brow_raise
                draw.arc([center_x - mouth_width, mouth_y - mouth_height,
                          center_x + mouth_width, mouth_y + mouth_height],
                         start=start_angle, end=end_angle,
                         fill='black', width=4)
        elif mouth_open < -0.1:
            # Sad mouth (downturned)
            draw.arc([center_x - mouth_width, mouth_y - mouth_height,
                      center_x + mouth_width, mouth_y + mouth_height],
                     start=0, end=180,
                     fill='black', width=3)
        else:
            # Neutral mouth
            draw.line([center_x - mouth_width // 2, mouth_y,
                       center_x + mouth_width // 2, mouth_y],
                      fill='black', width=2)

        # 4. Cheeks (subtle effect)
        if cheek_swell > 0.1:
            cheek_radius = int(width // 15 * cheek_swell)

            # Left cheek
            left_cheek = (center_x - width // 4, center_y + height // 8)
            # Right cheek
            right_cheek = (center_x + width // 4, center_y + height // 8)

            for cheek_x, cheek_y in [left_cheek, right_cheek]:
                # Draw subtle cheek blush
                for r in range(cheek_radius, 0, -1):
                    alpha = int(30 * (r / cheek_radius) * cheek_swell)
                    draw.ellipse([cheek_x - r, cheek_y - r,
                                  cheek_x + r, cheek_y + r],
                                 outline=(255, 150, 150, alpha), width=1)

        # 5. Nose (simple)
        nose_width = width // 40
        draw.line([center_x, center_y - height // 12,
                   center_x, center_y + height // 12],
                  fill='black', width=nose_width)

        # Add frame number text (for debugging)
        draw.text((10, 10), f"Frame {frame_num+1}/{total_frames}", fill='black')

        return img

    def create_animation(self, image_path, expression="smile", num_frames=25, output_dir="frames"):
        """Create complete animation"""

        print(f"\n🎬 Creating {expression.upper()} animation ({num_frames} frames)...")

        # Load base image
        try:
            if CV2_AVAILABLE:
                # Load with OpenCV for better color handling
                img_cv = cv2.imread(image_path)
                if img_cv is None:
                    raise ValueError("Could not load image")

                # Convert to PIL
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                base_img = Image.fromarray(img_rgb)
            else:
                # Load with PIL
                base_img = Image.open(image_path).convert('RGB')

            print(f"✅ Base image loaded: {base_img.size}")

        except Exception as e:
            print(f"❌ Error loading image: {e}")
            # Create a simple face
            base_img = Image.new('RGB', (512, 512), color='lightblue')
            draw = ImageDraw.Draw(base_img)
            # Draw basic face
            draw.ellipse([100, 100, 412, 412], outline='black', width=2)
            draw.ellipse([180, 200, 220, 240], fill='black')
            draw.ellipse([292, 200, 332, 240], fill='black')
            draw.arc([180, 300, 332, 380], start=180, end=360, fill='black', width=3)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate frames
        frame_paths = []
        for i in range(num_frames):
            frame = self.create_animation_frame(base_img, i, num_frames, expression)

            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            frame.save(frame_path, quality=90)
            frame_paths.append(frame_path)

            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  ⏳ Frame {i + 1}/{num_frames}")

        print(f"✅ {len(frame_paths)} frames created")
        return frame_paths

    def create_video_from_frames(self, frame_paths, output_path="animation.mp4", fps=20):
        """Create video from frames"""

        if not frame_paths:
            print("❌ No frames to create video")
            return None

        print(f"\n🎥 Creating video ({fps} FPS)...")

        try:
            # Try imageio first
            if IMAGEIO_AVAILABLE:
                # Read first frame for dimensions
                first_frame = imageio.imread(frame_paths[0])
                height, width = first_frame.shape[:2]

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
                print(f"✅ Video created with imageio: {output_path}")

            # Fallback to OpenCV
            elif CV2_AVAILABLE:
                # Read first frame for dimensions
                first_frame = cv2.imread(frame_paths[0])
                height, width = first_frame.shape[:2]

                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                for frame_path in frame_paths:
                    frame = cv2.imread(frame_path)
                    out.write(frame)

                out.release()
                print(f"✅ Video created with OpenCV: {output_path}")

            else:
                # Create GIF as fallback
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
                    duration=1000//fps,
                    loop=0
                )
                output_path = gif_path
                print(f"✅ GIF created: {gif_path}")

        except Exception as e:
            print(f"❌ Error creating video: {e}")
            return None

        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"📊 File size: {file_size:.1f} KB")

            # Display preview
            try:
                from IPython.display import Video, Image as IPImage

                if output_path.endswith('.gif'):
                    display(IPImage(filename=output_path))
                else:
                    display(Video(output_path, width=400, embed=True))
            except:
                print("📺 Video created (preview not available)")

            return output_path

        return None

    def create_voiceover(self, text, output_path="voiceover.mp3"):
        """Create voiceover from text"""

        if not TTS_AVAILABLE:
            print("⚠️ Voice generation not available")
            return None

        try:
            print(f"🎤 Creating voiceover: '{text[:50]}...'")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)

            if os.path.exists(output_path):
                print(f"✅ Voiceover created: {output_path}")

                # Try to play preview
                try:
                    from IPython.display import Audio
                    display(Audio(output_path, autoplay=False))
                except:
                    pass

                return output_path

        except Exception as e:
            print(f"❌ Voiceover creation failed: {e}")

        return None

# ==============================================
# 4. CREATE ANIMATION WITH OPTIONS
# ==============================================

def create_animation_with_options():
    """Create animation based on user choices"""

    print("\n" + "="*60)
    print("⚙️ ANIMATION OPTIONS")
    print("="*60)

    # Define expressions
    expressions = {
        "1": {"name": "smile", "desc": "😊 Gentle smile"},
        "2": {"name": "laugh", "desc": "😂 Big laugh with open mouth"},
        "3": {"name": "talk", "desc": "🗣️ Talking motion"},
        "4": {"name": "wink", "desc": "😉 Wink with one eye"},
        "5": {"name": "sad", "desc": "😔 Sad expression"}
    }

    # Display options
    print("\nSelect expression:")
    for key, expr in expressions.items():
        print(f"  {key}. {expr['desc']}")

    # Get user choice
    choice = input("\nEnter choice (1-5, default=1): ").strip()
    if choice not in expressions:
        choice = "1"

    expression = expressions[choice]["name"]

    # Get frame count
    try:
        frames = int(input(f"Number of frames (15-50, default=25): ").strip() or "25")
        frames = max(15, min(50, frames))
    except:
        frames = 25

    # Get FPS
    try:
        fps = int(input(f"Frames per second (10-30, default=20): ").strip() or "20")
        fps = max(10, min(30, fps))
    except:
        fps = 20

    # Get text for voiceover
    text = input(f"Text for voiceover (optional, press Enter to skip): ").strip()

    print(f"\n🎯 Settings confirmed:")
    print(f"• Expression: {expressions[choice]['desc']}")
    print(f"• Frames: {frames}")
    print(f"• FPS: {fps}")
    print(f"• Voice text: {text[:50]}{'...' if len(text) > 50 else ''}")

    # Create animation
    print("\n" + "="*60)
    print("🚀 CREATING ANIMATION")
    print("="*60)

    animator = SimpleFaceAnimator()

    # 1. Create frames
    frame_paths = animator.create_animation(
        image_path,
        expression=expression,
        num_frames=frames,
        output_dir=f"frames_{expression}"
    )

    if not frame_paths:
        print("❌ Failed to create frames")
        return

    # 2. Create video
    video_file = f"animation_{expression}.mp4"
    video_path = animator.create_video_from_frames(
        frame_paths,
        output_path=video_file,
        fps=fps
    )

    if not video_path:
        print("❌ Failed to create video")
        return

    # 3. Create voiceover if text provided
    voice_path = None
    if text and TTS_AVAILABLE:
        voice_file = f"voice_{expression}.mp3"
        voice_path = animator.create_voiceover(text, voice_file)

    # 4. Create download package
    print("\n" + "="*60)
    print("📦 CREATING DOWNLOAD PACKAGE")
    print("="*60)

    try:
        zip_file = f"face_animation_{expression}.zip"

        with zipfile.ZipFile(zip_file, 'w') as zf:
            # Add video
            zf.write(video_path, os.path.basename(video_path))

            # Add voice if created
            if voice_path and os.path.exists(voice_path):
                zf.write(voice_path, os.path.basename(voice_path))

            # Add original image
            zf.write(image_path, "original_image.jpg")

            # Add info file
            info = f"""FACE ANIMATION PACKAGE
====================
Expression: {expression}
Frames: {frames}
FPS: {fps}
Duration: {frames/fps:.1f} seconds
Created with: Simple Face Animator (Binder Edition)

Voice text: {text if text else "No voiceover"}
"""
            zf.writestr("INFO.txt", info)

        print(f"✅ Package created: {zip_file}")

        # Show download instructions
        print("\n" + "="*60)
        print("📥 DOWNLOAD INSTRUCTIONS")
        print("="*60)

        # Create HTML download links
        html_content = f"""
        <div style="background:#e3f2fd; padding:20px; border-radius:10px; margin:10px 0;">
        <h2>🎉 ANIMATION READY!</h2>
        
        <h3>📁 Files Created:</h3>
        <ul>
        <li><b>{os.path.basename(video_path)}</b> - Main animation video</li>
        """

        if voice_path:
            html_content += f'<li><b>{os.path.basename(voice_path)}</b> - Voiceover audio</li>'

        html_content += f"""
        <li><b>{zip_file}</b> - Complete package (ZIP)</li>
        </ul>
        
        <h3>⬇️ How to Download:</h3>
        <p><b>Option 1:</b> Use the file browser on the left</p>
        <p><b>Option 2:</b> Right-click links below → "Save link as"</p>
        
        <div style="background:white; padding:15px; border-radius:5px; margin:10px 0;">
        <h4>📥 Direct Download Links:</h4>
        <p><a href="{video_path}" download style="color:#2196F3; text-decoration:none; font-weight:bold;">
        ⬇️ Download Video: {os.path.basename(video_path)}
        </a></p>
        """

        if voice_path:
            html_content += f"""
            <p><a href="{voice_path}" download style="color:#2196F3; text-decoration:none; font-weight:bold;">
            ⬇️ Download Voice: {os.path.basename(voice_path)}
            </a></p>
            """

        html_content += f"""
        <p><a href="{zip_file}" download style="color:#4CAF50; text-decoration:none; font-weight:bold;">
        📦 Download Complete Package: {zip_file}
        </a></p>
        </div>
        
        <h3>🎬 Preview:</h3>
        <p>The animation should be displayed above. If not, download and play locally.</p>
        </div>
        """

        from IPython.display import display, HTML
        display(HTML(html_content))

        # Also show file sizes
        print("\n📊 FILE SIZES:")
        for file in [video_path, voice_path, zip_file, image_path]:
            if file and os.path.exists(file):
                size_kb = os.path.getsize(file) / 1024
                print(f"  • {os.path.basename(file):30} - {size_kb:.1f} KB")

    except Exception as e:
        print(f"❌ Error creating package: {e}")

        # Still show individual download links
        print("\n📥 Download individual files:")
        print(f"  • Video: {video_path}")
        if voice_path:
            print(f"  • Voice: {voice_path}")
        print(f"  • Original: {image_path}")

# ==============================================
# 5. MAIN EXECUTION
# ==============================================

if __name__ == "__main__":
    # Create animation
    create_animation_with_options()

    # Show next steps
    print("\n" + "="*60)
    print("🔄 CREATE ANOTHER ANIMATION")
    print("="*60)

    print("\nTo create another animation:")
    print("1. Run the cell again")
    print("2. Or call: create_animation_with_options()")
    print("\nTo change the image:")
    print("1. Upload a new JPG/PNG file")
    print("2. Restart and run the notebook")

    print("\n" + "="*60)
    print("🎭 ENJOY YOUR ANIMATED FACE!")
    print("="*60)