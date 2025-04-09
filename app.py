from flask import Flask, request, jsonify, send_file, render_template

import sounddevice as sd

import numpy as np

import whisper

from gtts import gTTS

import os

import tempfile

import noisereduce as nr

import scipy.signal

import Levenshtein

import uuid

from werkzeug.utils import secure_filename

import pygame

from pygame import mixer

import threading

import soundfile as sf

import time

import shutil

import logging



# Initialize pygame mixer with proper settings

pygame.mixer.pre_init(44100, -16, 2, 4096)

pygame.init()

mixer.init()



app = Flask(__name__)



# Configuration

UPLOAD_FOLDER = 'uploads'

TEMPLATE_FOLDER = 'templates'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Set up logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



# Global variables

models = {

    'en': whisper.load_model("small.en"),

    'ar': whisper.load_model("small")

}

current_language = 'en'

fs = 44100  # Sample rate





def clean_text(text, language):

    """Clean recognized text"""

    if language == "ar":

        replacements = {

            "أ": "ا", "إ": "ا", "آ": "ا",

            "ة": "ه", "ى": "ي", "ئ": "ء", "ؤ": "ء"

        }

        for old, new in replacements.items():

            text = text.replace(old, new)

    return text.strip()





def calculate_accuracy(target, actual):

    """Calculate pronunciation accuracy"""

    if not target or not actual:

        return 0

    distance = Levenshtein.distance(target, actual)

    max_len = max(len(target), len(actual))

    return max(0, 100 - (distance / max_len * 100))


def generate_feedback(target, actual, accuracy, language):
    """Generate detailed language-specific feedback"""
    if not target or not actual:
        return "Could not understand. Please try again." if language == "en" else "لم أتمكن من الفهم. حاول مرة أخرى"

    # Adjusted thresholds for more realistic feedback
    if accuracy > 95:
        return "🌟 Excellent pronunciation! Perfectly said!" if language == "en" else "🌟 ممتاز! النطق واضح وصحيح تمامًا"
    elif accuracy > 85:
        base_feedback = "Good, but needs slight improvement:" if language == "en" else "جيد، ولكن يحتاج تحسينًا بسيطًا:"
    else:
        base_feedback = "Needs improvement:" if language == "en" else "يحتاج تحسينًا:"

    feedback = [base_feedback]

    if language == "ar":
        # Detailed Arabic pronunciation tips
        arabic_letter_guides = {
            "ض": "للضاد (ض): اضغط لسانك على الأسنان العلوية مع صوت مرتفع",
            "ص": "للصاد (ص): شد شفتيك مع ضغط اللسان على الأسنان",
            "ق": "للقاف (ق): أخرجه من أعمق الحلق مثل صوت العقعقة",
            "ظ": "للظاء (ظ): أخرج لسانك قليلاً بين الأسنان مع صوت مجهور",
            "ط": "للطاء (ط): اضغط اللسان على سقف الحلق بقوة",
            "ع": "للعين (ع): احرص على الاهتزاز الصوتي من الحلق",
            "ح": "للحاء (ح): أخرج الهواء بحدة من الحلق بدون صوت",
            "غ": "للغين (غ): مثل الفرنسية R من الحلق",
            "خ": "للخاء (خ): مثل تخشية الصوت من أقصى الحلق"
        }

        # Check for missing Arabic letters
        for letter, tip in arabic_letter_guides.items():
            if letter in target and letter not in actual:
                feedback.append(f"• {tip}")

        # Common Arabic mistakes
        if any(c in target for c in ['أ', 'إ', 'آ']) and 'ا' not in actual:
            feedback.append("• الهمزات (أ، إ، آ) تختلف عن الألف (ا) في النطق")
        if 'ة' in target and 'ه' not in actual:
            feedback.append("• التاء المربوطة (ة) تنطق هاء ساكنة في آخر الكلمة")

    else:  # English
        # Detailed English pronunciation tips
        english_sound_guides = {
            "th": "For 'th': Place tongue between teeth (voiced as in 'this'/voiceless as in 'think')",
            "r": "For 'r': Curl tongue back without touching roof (American) or tap (British)",
            "l": "For 'l': Touch alveolar ridge (behind teeth) with tongue tip",
            "v": "For 'v': Gently bite lower lip with upper teeth",
            "w": "For 'w': Round lips tightly like saying 'oo' quickly",
            "ʃ": "For 'sh': Flatten tongue and round lips (like 'shoe')",
            "θ": "For unvoiced 'th': Tongue between teeth with air (like 'three')",
            "ð": "For voiced 'th': Tongue between teeth with voice (like 'this')"
        }

        # Check vowel length issues
        if len(target.split()) == len(actual.split()):
            for sound, tip in english_sound_guides.items():
                if sound in target.lower() and sound not in actual.lower():
                    feedback.append(f"• {tip}")
        else:
            feedback.append("• Check word boundaries and syllable count")

    # Fallback generic advice if no specific issues found
    if len(feedback) == 1:
        if language == "en":
            feedback.extend([
                "• Speak more slowly and articulate each sound",
                "• Record yourself and compare to native speakers",
                "• Practice difficult sounds in isolation first"
            ])
        else:
            feedback.extend([
                "• تحدث ببطء ووضوح أكثر",
                "• سجل صوتك وقارنه بالنطق الأصلي",
                "• تدرب على الأصوات الصعبة منفردة أولاً"
            ])

    return "\n".join(feedback[:5])  # Return max 5 tips

@app.route('/')

def home():

    """Serve the interactive web interface"""

    return render_template('index.html')





@app.route('/set_language', methods=['POST'])

def set_language():

    """Endpoint to set the current language"""

    global current_language

    data = request.json

    lang = data.get('language', 'en')

    if lang in ['en', 'ar']:

        current_language = lang

        logger.info(f"Language set to {current_language}")

        return jsonify({'status': 'success', 'language': current_language})

    return jsonify({'status': 'error', 'message': 'Invalid language'}), 400





@app.route('/record', methods=['POST'])

def record_audio():

    """Endpoint to record and analyze pronunciation"""

    try:

        data = request.json

        target_text = data.get('text', '').strip()

        if not target_text:

            return jsonify({'status': 'error', 'message': 'No text provided'}), 400



        # Record audio

        duration = 5  # seconds

        logger.info(f"Starting recording for {duration} seconds...")

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)

        sd.wait()



        # Process audio

        audio = recording.squeeze()

        if audio.ndim > 1:

            audio = np.mean(audio, axis=1)



        cleaned_audio = nr.reduce_noise(

            y=audio,

            sr=fs,

            prop_decrease=0.8 if current_language == 'ar' else 0.9,

            n_fft=1024

        )



        # Save recording

        recording_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"recording_{uuid.uuid4()}.wav")

        sf.write(recording_filename, cleaned_audio, fs)

        logger.info(f"Recording saved to {recording_filename}")



        # Transcribe

        result = models[current_language].transcribe(

            recording_filename,

            language=current_language,

            initial_prompt=target_text,

            temperature=0.2

        )

        user_text = clean_text(result["text"].strip(), current_language)



        # Calculate accuracy

        accuracy = calculate_accuracy(target_text, user_text)

        feedback = generate_feedback(target_text, user_text, accuracy, current_language)



        # Generate correct pronunciation audio

        tts_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"tts_{uuid.uuid4()}.mp3")

        try:

            tts = gTTS(

                text=target_text,

                lang='ar' if current_language == 'ar' else 'en',

                slow=True,

                lang_check=False

            )

            tts.save(tts_filename)

            logger.info(f"TTS audio saved to {tts_filename}")

        except Exception as e:

            logger.error(f"TTS generation failed: {str(e)}")

            return jsonify({

                'status': 'error',

                'message': f'TTS generation failed: {str(e)}'

            }), 500



        response = {

            'status': 'success',

            'target_text': target_text,

            'user_text': user_text,

            'accuracy': accuracy,

            'feedback': feedback,

            'recording_url': f"/get_audio/{os.path.basename(recording_filename)}",

            'correct_pronunciation_url': f"/get_audio/{os.path.basename(tts_filename)}"

        }

        return jsonify(response)



    except Exception as e:

        logger.error(f"Recording error: {str(e)}")

        return jsonify({'status': 'error', 'message': str(e)}), 500





@app.route('/get_audio/<filename>', methods=['GET'])

def get_audio(filename):

    """Endpoint to serve audio files"""

    try:

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

        if not os.path.exists(filepath):

            return jsonify({'status': 'error', 'message': 'File not found'}), 404

        return send_file(filepath, mimetype='audio/mp3')

    except Exception as e:

        logger.error(f"Audio file error: {str(e)}")

        return jsonify({'status': 'error', 'message': str(e)}), 500





@app.route('/stop_audio', methods=['POST'])

def stop_audio():

    """Endpoint to stop currently playing audio"""

    try:

        mixer.music.stop()

        return jsonify({'status': 'success', 'message': 'Audio stopped'})

    except Exception as e:

        logger.error(f"Stop audio error: {str(e)}")

        return jsonify({'status': 'error', 'message': str(e)}), 500





@app.route('/play_audio', methods=['POST'])

def play_audio():

    """Endpoint to play audio directly with language-specific handling"""

    data = request.json

    filename = data.get('filename')

    language = data.get('language', current_language)



    if not filename:

        return jsonify({'status': 'error', 'message': 'No filename provided'}), 400



    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

    if not os.path.exists(filepath):

        return jsonify({'status': 'error', 'message': 'File not found'}), 404



    try:

        # Stop any currently playing audio

        mixer.music.stop()



        # Reinitialize mixer with proper settings

        try:

            mixer.quit()

        except:

            pass

        mixer.init(44100, -16, 2, 4096)



        # Load the audio file and get duration

        sound = mixer.Sound(filepath)

        duration = int(sound.get_length() * 1000)  # Convert to milliseconds



        # Play the audio

        mixer.music.load(filepath)

        mixer.music.set_volume(1.0)

        mixer.music.play()



        return jsonify({

            'status': 'success',

            'message': 'Playing audio',

            'duration': duration

        })

    except Exception as e:

        logger.error(f"Playback error: {str(e)}")

        return jsonify({'status': 'error', 'message': str(e)}), 500





if __name__ == '__main__':

    # Initialize directories

    os.makedirs('uploads', exist_ok=True)

    os.makedirs('templates', exist_ok=True)

    app.run(host='0.0.0.0', port=5000, debug=True)