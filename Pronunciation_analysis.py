import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import numpy as np
import whisper
from gtts import gTTS
import os
import wave
from threading import Thread
import noisereduce as nr
import scipy.signal
import arabic_reshaper
from bidi.algorithm import get_display
import Levenshtein
import warnings
import soundfile as sf
import tempfile
import pygame
from pygame import mixer

# Initialize pygame mixer
pygame.init()
mixer.init()

warnings.filterwarnings("ignore")


class PronunciationCoach:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Pronunciation Coach")
        self.root.geometry("850x700")
        self.root.configure(bg="#F0F8FF")

        # Initialize variables
        self.fs = 44100
        self.recording = None
        self.last_recording_path = os.path.join(tempfile.gettempdir(), "user_recording.wav")
        self.last_user_text = ""
        self.current_language = "en"
        self.model = whisper.load_model("small.en")  # Start with English
        self.target_text = ""
        self.current_audio_file = None

        self.setup_main_ui()
        Thread(target=self.warmup_model, daemon=True).start()

    def setup_main_ui(self):
        """Setup main interface with language dropdown"""
        main_frame = tk.Frame(self.root, bg="#F0F8FF")
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Language selection dropdown
        lang_frame = tk.Frame(main_frame, bg="#F0F8FF")
        lang_frame.pack(pady=5, fill=tk.X)

        tk.Label(lang_frame, text="Practice Language:",
                 font=("Arial", 12), bg="#F0F8FF").pack(side=tk.LEFT, padx=5)

        self.lang_var = tk.StringVar(value="English")
        self.lang_dropdown = ttk.Combobox(lang_frame,
                                          textvariable=self.lang_var,
                                          values=["English", "Arabic"],
                                          font=("Arial", 12),
                                          state="readonly",
                                          width=10)
        self.lang_dropdown.pack(side=tk.LEFT)
        self.lang_dropdown.bind("<<ComboboxSelected>>", self.change_language)

        # Rest of UI setup
        tk.Label(main_frame,
                 text="Enter what you want to practice:",
                 font=("Arial", 14), bg="#F0F8FF").pack(pady=10)

        self.target_entry = tk.Entry(main_frame,
                                     font=("Arial", 18),
                                     width=40)
        self.target_entry.pack(pady=10, ipady=5)

        tk.Button(main_frame, text="Show Example",
                  command=self.show_example,
                  font=("Arial", 12), bg="#2196F3", fg="white").pack(pady=5)

        tk.Label(main_frame,
                 text="Press RECORD and say the phrase above",
                 font=("Arial", 14), bg="#F0F8FF").pack(pady=15)

        self.result_frame = tk.Frame(main_frame, bg="#F0F8FF")
        self.result_frame.pack(pady=10)

        self.target_label = tk.Label(self.result_frame, text="",
                                     font=("Arial", 20), bg="#F0F8FF",
                                     fg="#2E7D32", wraplength=600)
        self.target_label.pack()

        self.user_label = tk.Label(self.result_frame, text="",
                                   font=("Arial", 20), bg="#F0F8FF",
                                   fg="#C62828", wraplength=600)
        self.user_label.pack(pady=10)

        btn_frame = tk.Frame(main_frame, bg="#F0F8FF")
        btn_frame.pack(pady=20)

        self.record_btn = tk.Button(btn_frame, text="🎤 RECORD",
                                    command=self.start_recording,
                                    font=("Arial", 18), bg="#FF5722",
                                    fg="white", width=15)
        self.record_btn.pack(side=tk.LEFT, padx=10)

        self.correct_btn = tk.Button(btn_frame, text="🔊 CORRECT",
                                     command=self.play_target,
                                     font=("Arial", 14), bg="#009688",
                                     fg="white", state=tk.DISABLED)
        self.correct_btn.pack(side=tk.LEFT, padx=5)

        self.your_btn = tk.Button(btn_frame, text="🔊 YOURS",
                                  command=self.play_yours,
                                  font=("Arial", 14), bg="#607D8B",
                                  fg="white", state=tk.DISABLED)
        self.your_btn.pack(side=tk.LEFT, padx=5)

        self.feedback_frame = tk.LabelFrame(main_frame,
                                            text=" Pronunciation Feedback ",
                                            font=("Arial", 12), bg="#F0F8FF")
        self.feedback_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.feedback_label = tk.Label(self.feedback_frame, text="",
                                       font=("Arial", 12), bg="#F0F8FF",
                                       wraplength=650, justify="left")
        self.feedback_label.pack(pady=10, padx=10)

        self.accuracy_frame = tk.Frame(main_frame, bg="#F0F8FF")
        self.accuracy_frame.pack(pady=10)

        self.accuracy_label = tk.Label(self.accuracy_frame, text="Accuracy: -",
                                       font=("Arial", 14), bg="#F0F8FF")
        self.accuracy_label.pack()

        self.accuracy_bar = ttk.Progressbar(self.accuracy_frame,
                                            orient="horizontal",
                                            length=300, mode="determinate")
        self.accuracy_bar.pack(pady=5)

        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
        style.configure("yellow.Horizontal.TProgressbar", foreground='orange', background='orange')
        style.configure("red.Horizontal.TProgressbar", foreground='red', background='red')

    def change_language(self, event=None):
        """Handle language change during runtime"""
        lang_choice = self.lang_var.get()
        new_lang = "en" if lang_choice == "English" else "ar"

        if new_lang != self.current_language:
            self.current_language = new_lang
            model_name = "small.en" if new_lang == "en" else "small"

            self.feedback_label.config(text=f"Loading {lang_choice} model...", fg="blue")
            self.root.update()

            Thread(target=self.load_model, args=(model_name,), daemon=True).start()

    def load_model(self, model_name):
        """Load model in background"""
        try:
            self.model = whisper.load_model(model_name)
            self.root.after(0, lambda: self.feedback_label.config(
                text=f"Ready to practice {self.lang_var.get()}!",
                fg="green"))
        except Exception as e:
            self.root.after(0, lambda: self.feedback_label.config(
                text=f"Error loading model: {str(e)}",
                fg="red"))

    def warmup_model(self):
        """Preload model"""
        dummy_audio = np.random.randn(self.fs * 3)
        self.model.transcribe(dummy_audio)

    def show_example(self):
        """Show example phrases"""
        examples = {
            "en": ["Hello world", "Good morning", "How are you?", "door", "tree"],
            "ar": ["مرحبا بالعالم", "صباح الخير", "كيف حالك؟", "باب", "شجرة"]
        }
        example = examples[self.current_language][3]
        self.target_entry.delete(0, tk.END)
        self.target_entry.insert(0, example)

    def start_recording(self):
        """Start recording"""
        self.target_text = self.target_entry.get().strip()

        if not self.target_text:
            messagebox.showwarning("Input Needed", "Please enter text to practice first")
            return

        self.record_btn.config(state=tk.DISABLED, text="🎙️ RECORDING...")
        self.feedback_label.config(text="Listening...", fg="black")
        self.correct_btn.config(state=tk.DISABLED)
        self.your_btn.config(state=tk.DISABLED)

        display_text = self.format_arabic(self.target_text) if self.current_language == "ar" else self.target_text
        self.target_label.config(text=f"Target: {display_text}")
        self.user_label.config(text="You said: ")

        Thread(target=self.record_and_analyze, daemon=True).start()

    def record_and_analyze(self):
        """Record and analyze"""
        try:
            self.recording = sd.rec(int(5 * self.fs), samplerate=self.fs, channels=1)
            sd.wait()
            self.process_audio()
            self.root.after(0, self.analyze_recording)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Recording failed: {str(e)}"))
            self.root.after(0, self.reset_ui)

    def process_audio(self):
        """Process audio"""
        try:
            audio = self.recording.squeeze()
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            cleaned_audio = nr.reduce_noise(
                y=audio,
                sr=self.fs,
                prop_decrease=0.8 if self.current_language == "ar" else 0.9,
                n_fft=1024
            )

            sf.write(self.last_recording_path, cleaned_audio, self.fs)

        except Exception as e:
            raise RuntimeError(f"Audio processing error: {str(e)}")

    def analyze_recording(self):
        """Analyze recording"""
        try:
            self.feedback_label.config(text="🔍 Analyzing your pronunciation...")
            self.root.update()

            result = self.model.transcribe(
                self.last_recording_path,
                language=self.current_language,
                initial_prompt=self.target_text,
                temperature=0.2
            )

            user_text = self.clean_text(result["text"].strip())
            self.last_user_text = user_text

            display_text = self.format_arabic(user_text) if self.current_language == "ar" else user_text
            self.user_label.config(text=f"You said: {display_text}")

            accuracy = self.calculate_accuracy(self.target_text, user_text)
            feedback = self.generate_feedback(self.target_text, user_text, accuracy)

            self.update_feedback_ui(accuracy, feedback)

            self.correct_btn.config(state=tk.NORMAL)
            self.your_btn.config(state=tk.NORMAL)

        except Exception as e:
            self.feedback_label.config(text=f"Analysis error: {str(e)}", fg="red")
        finally:
            self.record_btn.config(state=tk.NORMAL, text="🎤 RECORD AGAIN")

    def clean_text(self, text):
        """Clean text"""
        if self.current_language == "ar":
            replacements = {
                "أ": "ا", "إ": "ا", "آ": "ا",
                "ة": "ه", "ى": "ي", "ئ": "ء", "ؤ": "ء"
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
        return text.strip()

    def calculate_accuracy(self, target, actual):
        """Calculate accuracy"""
        if not target or not actual:
            return 0

        distance = Levenshtein.distance(target, actual)
        max_len = max(len(target), len(actual))
        return max(0, 100 - (distance / max_len * 100))

    def generate_feedback(self, target, actual, accuracy):
        """Generate detailed feedback with language-specific tips"""
        if accuracy == 0:
            return "Could not understand. Please try again." if self.current_language == "en" else "لم أتمكن من الفهم. حاول مرة أخرى"

        # More realistic accuracy thresholds
        if accuracy > 95:
            return "🌟 Excellent pronunciation! Perfect match!" if self.current_language == "en" else "🌟 ممتاز! تطابق تام في النطق"
        elif accuracy > 85:
            base_feedback = "Good, but needs work:" if self.current_language == "en" else "جيد، ولكن يحتاج تحسينًا:"
        else:
            base_feedback = "Needs practice:" if self.current_language == "en" else "يحتاج تدريبًا:"

        feedback = [base_feedback]

        if self.current_language == "ar":
            # Enhanced Arabic feedback
            arabic_guides = {
                "ض": "ض: اضغط اللسان على الأسنان العلوية بقوة",
                "ص": "ص: شد الشفتين مع ضغط اللسان",
                "ق": "ق: من أقصى الحلق مع صوت مجهور",
                "غ": "غ: مثل الفرنسية R من الحلق",
                "ع": "ع: اهتزاز صوتي من الحلق",
                "ح": "ح: هواء حاد من الحلق بدون صوت",
                "خ": "خ: هواء من أقصى الحلق مثل الهمس",
                "ط": "ط: لسان يضغط على سقف الحلق بقوة"
            }

            for letter, tip in arabic_guides.items():
                if letter in target and letter not in actual:
                    feedback.append(f"• {tip}")

            # Special cases
            if 'ة' in target and 'ه' not in actual:
                feedback.append("• ة: تنطق هاء ساكنة في نهاية الكلمات")
            if any(c in target for c in ['أ', 'إ', 'آ']) and 'ا' not in actual:
                feedback.append("• الهمزات (أ،إ،آ) تختلف عن الألف (ا)")

        else:  # English
            # Enhanced English feedback
            english_guides = {
                "th": "'th': Tongue between teeth ('this'/'think')",
                "r": "'r': Tongue curls back (American) or taps (British)",
                "l": "'l': Tongue tip touches alveolar ridge",
                "v": "'v': Lower lip touches upper teeth",
                "w": "'w': Lips rounded tightly like 'oo'",
                "n": "'n': Tongue touches alveolar ridge",
                "ʃ": "'sh': Flat tongue with rounded lips",
                "θ": "Unvoiced 'th' (thin): Tongue between teeth + air",
                "ð": "Voiced 'th' (this): Tongue between teeth + voice"
            }

            for sound, tip in english_guides.items():
                if sound in target.lower() and sound not in actual.lower():
                    feedback.append(f"• {tip}")

        # Fallback advice
        if len(feedback) == 1:
            if self.current_language == "en":
                feedback.extend([
                    "• Record and compare with native speakers",
                    "• Practice minimal pairs (ship/sheep)",
                    "• Slow down and exaggerate sounds"
                ])
            else:
                feedback.extend([
                    "• سجل صوتك وقارنه بالناطقين الأصليين",
                    "• تدرب على الأزواج الصوتية (قال/قيل)",
                    "• اخفض السرعة ووضوح الأصوات"
                ])

        return "\n".join(feedback[:5])  # Max 5 tips

    def update_feedback_ui(self, accuracy, feedback):
        """Update UI"""
        self.accuracy_bar["value"] = accuracy
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.1f}%")

        if accuracy > 70:
            self.accuracy_bar["style"] = "green.Horizontal.TProgressbar"
        elif accuracy > 40:
            self.accuracy_bar["style"] = "yellow.Horizontal.TProgressbar"
        else:
            self.accuracy_bar["style"] = "red.Horizontal.TProgressbar"

        self.feedback_label.config(text=feedback, fg="#333333")

    def format_arabic(self, text):
        """Format Arabic"""
        if self.current_language == "ar":
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        return text

    def play_target(self):
        """Play target pronunciation using pygame"""
        if self.target_text:
            try:
                # Stop any currently playing audio
                mixer.music.stop()

                # Create temp file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tts = gTTS(text=self.target_text, lang=self.current_language, slow=True)
                    tts.save(tmp.name)
                    self.current_audio_file = tmp.name

                # Load and play the audio
                mixer.music.load(self.current_audio_file)
                mixer.music.play()

            except Exception as e:
                messagebox.showerror("Playback Error", f"Couldn't play audio: {str(e)}")
                if hasattr(self, 'current_audio_file') and self.current_audio_file and os.path.exists(
                        self.current_audio_file):
                    os.unlink(self.current_audio_file)

    def play_yours(self):
        """Play user recording using pygame"""
        if os.path.exists(self.last_recording_path):
            try:
                # Stop any currently playing audio
                mixer.music.stop()

                # Load and play the recording
                mixer.music.load(self.last_recording_path)
                mixer.music.play()

            except Exception as e:
                messagebox.showerror("Playback Error", f"Couldn't play recording: {str(e)}")
        else:
            messagebox.showwarning("Error", "No recording available")

    def reset_ui(self):
        """Reset UI"""
        self.record_btn.config(state=tk.NORMAL, text="🎤 RECORD")
        self.feedback_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = PronunciationCoach(root)
    root.mainloop()