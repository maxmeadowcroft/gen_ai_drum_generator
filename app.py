import os
import random
import numpy as np
import streamlit as st
from io import BytesIO
import mido
import tensorflow as tf
from tensorflow import keras
from pydub import AudioSegment
from pathlib import Path

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


def pattern_to_midi(gen_combined, gen_kick, gen_hh, bpm):
    SNARE_NOTE = 38
    KICK_NOTE = 36
    HH_NOTE = 42

    mid = mido.MidiFile()

    kick_track = mido.MidiTrack()
    snare_track = mido.MidiTrack()
    hh_track = mido.MidiTrack()

    mid.tracks.append(kick_track)
    mid.tracks.append(snare_track)
    mid.tracks.append(hh_track)

    microseconds_per_beat = int((60_000_000 / bpm))
    mid.ticks_per_beat = 480
    kick_track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
    snare_track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
    hh_track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))

    step_ticks = 120

    for i in range(STEPS):
        if gen_kick[i] == 1:
            kick_track.append(mido.Message('note_on', note=KICK_NOTE, velocity=64, time=0))
            kick_track.append(mido.Message('note_off', note=KICK_NOTE, velocity=64, time=step_ticks))
        if gen_combined[i] == 1:
            snare_track.append(mido.Message('note_on', note=SNARE_NOTE, velocity=64, time=0))
            snare_track.append(mido.Message('note_off', note=SNARE_NOTE, velocity=64, time=step_ticks))
        if gen_hh[i] == 1:
            hh_track.append(mido.Message('note_on', note=HH_NOTE, velocity=64, time=0))
            hh_track.append(mido.Message('note_off', note=HH_NOTE, velocity=64, time=step_ticks))

    buffer = BytesIO()
    mid.save(buffer)
    buffer.seek(0)
    return buffer


def generate(model_choice, bpm, threshold, noise_level, temperature,
             kick_sound, snare_sound, hh_sound, output_format):
    model, X = load_model_and_data(model_choice)
    gen_combined, gen_kick, gen_hh = generate_pattern(model, X, bpm, threshold, noise_level, temperature)

    if output_format == "WAV":
        audio_buffer = pattern_to_audio_full(gen_kick, gen_combined, gen_hh, bpm, kick_sound, snare_sound, hh_sound)
        return ("Here is your WAV file!", audio_buffer, None)
    else:
        midi_buffer = pattern_to_midi(gen_combined, gen_kick, gen_hh, bpm)
        return ("Here is your MIDI file!", None, midi_buffer)


# -------------------------------------------
# Streamlit UI
# -------------------------------------------

st.title("Gen AI Drum Pattern Generator")

# Default sound paths
default_kick_path = Path("./kick.wav")
default_snare_path = Path("./snare.wav")
default_hh_path = Path("./hh.wav")

model_choices = ["lstm", "transformer"]
output_choices = ["WAV", "MIDI"]

model_choice = st.selectbox("Model Choice", model_choices, index=0)
output_format = st.radio("Output Format", output_choices, index=0)

bpm = st.slider("BPM", min_value=60, max_value=200, value=170)
threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
noise_level = st.slider("Noise Level", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

st.markdown("### Upload Your Drum Samples")
uploaded_kick = st.file_uploader("Upload Kick Sample (WAV)", type="wav")
uploaded_snare = st.file_uploader("Upload Snare Sample (WAV)", type="wav")
uploaded_hh = st.file_uploader("Upload Hi-Hat Sample (WAV)", type="wav")

if st.button("Generate"):
    try:
        # Use uploaded files or defaults
        kick_sound = AudioSegment.from_file(uploaded_kick) if uploaded_kick else AudioSegment.from_file(default_kick_path)
        snare_sound = AudioSegment.from_file(uploaded_snare) if uploaded_snare else AudioSegment.from_file(default_snare_path)
        hh_sound = AudioSegment.from_file(uploaded_hh) if uploaded_hh else AudioSegment.from_file(default_hh_path)

        label, wav_buffer, midi_buffer = generate(
            model_choice, bpm, threshold, noise_level, temperature,
            kick_sound, snare_sound, hh_sound, output_format
        )
        st.success(label)
        if wav_buffer:
            st.audio(wav_buffer, format="audio/wav")
            st.download_button("Download WAV", data=wav_buffer, file_name="generated_sample.wav", mime="audio/wav")
        if midi_buffer:
            st.download_button("Download MIDI", data=midi_buffer, file_name="generated_sample.mid", mime="audio/midi")
    except Exception as e:
        st.error(f"An error occurred: {e}")
