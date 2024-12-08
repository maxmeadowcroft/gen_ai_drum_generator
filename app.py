import os
import random
import numpy as np
import streamlit as st
from io import BytesIO
import mido
from tensorflow import keras
from pydub import AudioSegment
from pathlib import Path
from tempfile import NamedTemporaryFile

# -------------------------------------------
# Functions for model loading and generation
# -------------------------------------------

def load_model_and_data(model_choice):
    model_path = f"{model_choice}_model.keras"
    model = keras.models.load_model(model_path, compile=False)
    X = np.load('X_data.npy')
    if model_choice in ['lstm', 'transformer']:
        X = X.reshape(X.shape[0], 128, 3)
    return model, X


def apply_temperature(probabilities, temperature):
    if temperature == 1.0:
        return probabilities
    return np.power(probabilities, 1.0 / temperature)


BARS = 8
BEATS_PER_BAR = 4
TOTAL_BEATS = BARS * BEATS_PER_BAR
STEPS = 128


def generate_pattern(model, X, bpm, threshold, noise_level, temperature):
    random_index = random.randint(0, len(X) - 1)
    seed = X[random_index:random_index + 1]
    generated = model.predict(seed)[0]

    generated = apply_temperature(generated, temperature)
    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level, size=generated.shape)
        generated += noise
        generated = np.clip(generated, 0.0, 1.0)

    generated_bin = (generated > threshold).astype(np.int32)
    gen_combined = generated_bin[0:128]
    gen_kick = generated_bin[128:256]
    gen_hh = generated_bin[256:384]
    return gen_combined, gen_kick, gen_hh


def pattern_to_audio_full(gen_kick, gen_combined, gen_hh, bpm, kick_sound, snare_sound, hh_sound):
    ms_per_beat = 60000.0 / bpm
    total_duration_ms = TOTAL_BEATS * ms_per_beat
    step_duration_ms = total_duration_ms / STEPS

    output = AudioSegment.silent(duration=int(total_duration_ms))

    for i in range(STEPS):
        start_time_ms = int(i * step_duration_ms)
        if gen_kick[i] == 1:
            output = output.overlay(kick_sound, position=start_time_ms)
        if gen_combined[i] == 1:
            output = output.overlay(snare_sound, position=start_time_ms)
        if gen_hh[i] == 1:
            output = output.overlay(hh_sound, position=start_time_ms)

    buffer = BytesIO()
    output.export(buffer, format="wav")
    buffer.seek(0)
    return buffer


def generate_individual_midi_tracks(kick_pattern, snare_pattern, hh_pattern, bpm):
    KICK_NOTE = 60
    SNARE_NOTE = 60
    HH_NOTE = 60

    microseconds_per_beat = int(60_000_000 / bpm)
    step_ticks = 120

    def create_midi_track(pattern, note):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
        abs_time = 0
        for step in pattern:
            if step == 1:
                track.append(mido.Message('note_on', note=note, velocity=64, time=abs_time))
                track.append(mido.Message('note_off', note=note, velocity=64, time=step_ticks))
                abs_time = 0
            else:
                abs_time += step_ticks
        return mid

    # Create MIDI tracks
    kick_midi = create_midi_track(kick_pattern, KICK_NOTE)
    snare_midi = create_midi_track(snare_pattern, SNARE_NOTE)
    hh_midi = create_midi_track(hh_pattern, HH_NOTE)

    def save_to_temp_file(midi_file):
        with NamedTemporaryFile(delete=False, suffix=".mid") as temp_file:
            midi_file.save(temp_file.name)
            temp_file.seek(0)
            buffer = BytesIO(temp_file.read())
        os.unlink(temp_file.name)  # Clean up temp file
        return buffer

    # Save each MIDI track
    kick_buffer = save_to_temp_file(kick_midi)
    snare_buffer = save_to_temp_file(snare_midi)
    hh_buffer = save_to_temp_file(hh_midi)

    return kick_buffer, snare_buffer, hh_buffer


default_kick_path = Path("./kick.wav")
default_snare_path = Path("./snare.wav")
default_hh_path = Path("./hh.wav")

# -------------------------------------------
# Streamlit UI
# -------------------------------------------

# App Description
st.title("Gen AI Drum Pattern Generator")
st.markdown("""
Welcome to the Gen AI Drum Pattern Generator! This app uses cutting-edge generative AI models to create custom drum patterns with options for dynamic control and sound customization.

### Features:
- **Default Drums**: If you don't upload custom samples, we use our high-quality default drum sounds.
- **Customizable Controls**: Adjust parameters like BPM, noise, and temperature for unique variations in drum patterns.
- **Volume Control**: Fine-tune the volume of each track (kick, snare, hi-hat) to create your desired mix.

### How Each Slider Works:
- **BPM**: Controls the speed of the drum pattern.
- **Threshold**: Sets the intensity of generated beats (lower values = more beats).
- **Noise Level**: Adds randomness to the pattern for more organic results.
- **Temperature**: Affects creativity in AI generation (higher = more varied patterns).
- **Volume Sliders**: Adjust the loudness of each track (kick, snare, hi-hat).
""")

# Section: Model Selection
st.header("1. Select Model and Settings")
model_choices = ["lstm", "transformer"]
model_choice = st.selectbox("Model Choice", model_choices, index=0)

bpm = st.slider("BPM", min_value=120, max_value=240, value=170)
threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
noise_level = st.slider("Noise Level", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Section: Upload Drum Samples
st.header("2. Upload Drum Samples")
uploaded_kick = st.file_uploader("Upload Kick Sample (WAV)", type="wav")
uploaded_snare = st.file_uploader("Upload Snare Sample (WAV)", type="wav")
uploaded_hh = st.file_uploader("Upload Hi-Hat Sample (WAV)", type="wav")

# Section: Volume Controls
st.header("3. Adjust Track Volumes")
kick_volume = st.slider("Kick Volume", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
snare_volume = st.slider("Snare Volume", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
hh_volume = st.slider("Hi-Hat Volume", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

# Generate Button
if st.button("Generate"):
    try:
        # Load drum sounds
        kick_sound = AudioSegment.from_file(uploaded_kick) if uploaded_kick else AudioSegment.from_file(default_kick_path)
        snare_sound = AudioSegment.from_file(uploaded_snare) if uploaded_snare else AudioSegment.from_file(default_snare_path)
        hh_sound = AudioSegment.from_file(uploaded_hh) if uploaded_hh else AudioSegment.from_file(default_hh_path)

        # Adjust volumes
        kick_sound = kick_sound + (20 * np.log10(kick_volume)) if kick_volume > 0 else AudioSegment.silent()
        snare_sound = snare_sound + (20 * np.log10(snare_volume)) if snare_volume > 0 else AudioSegment.silent()
        hh_sound = hh_sound + (20 * np.log10(hh_volume)) if hh_volume > 0 else AudioSegment.silent()

        # Load model and generate patterns
        model, X = load_model_and_data(model_choice)
        gen_combined, gen_kick, gen_hh = generate_pattern(model, X, bpm, threshold, noise_level, temperature)

        # Generate audio and MIDI files
        audio_buffer = pattern_to_audio_full(gen_kick, gen_combined, gen_hh, bpm, kick_sound, snare_sound, hh_sound)
        kick_midi, snare_midi, hh_midi = generate_individual_midi_tracks(gen_kick, gen_combined, gen_hh, bpm)

        st.session_state.generated_data = {
            'audio_buffer': audio_buffer,
            'kick_midi': kick_midi,
            'snare_midi': snare_midi,
            'hh_midi': hh_midi
        }
        st.success("WAV and MIDI files generated successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Provide download buttons for generated files
if st.session_state.get("generated_data"):
    st.audio(st.session_state['generated_data']['audio_buffer'], format="audio/wav")
    st.download_button("Download WAV", data=st.session_state['generated_data']['audio_buffer'], file_name="generated_sample.wav", mime="audio/wav")
    st.download_button("Download Kick MIDI", data=st.session_state['generated_data']['kick_midi'], file_name="kick_track.mid", mime="audio/midi")
    st.download_button("Download Snare MIDI", data=st.session_state['generated_data']['snare_midi'], file_name="snare_track.mid", mime="audio/midi")
    st.download_button("Download Hi-Hat MIDI", data=st.session_state['generated_data']['hh_midi'], file_name="hh_track.mid", mime="audio/midi")
