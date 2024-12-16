
# Gen AI Drum Pattern Generator  

## Overview  
  
The **Gen AI Drum Pattern Generator** is a cutting-edge Streamlit application that leverages machine learning models to create unique drum patterns. Customize your beats, export WAV and MIDI files, and add your personal flair with lo-fi effects and custom drum samples.

## Features  
  
- **Generative AI Models**: Choose between LSTM and Transformer models for generating drum patterns.
- **Custom Drum Samples**: Upload your own kick, snare, and hi-hat samples or use high-quality defaults.
- **Dynamic Controls**:
    - Adjust BPM, noise level, threshold, and temperature for creative variations.
    - Set individual track volumes (kick, snare, hi-hat).
- **Lo-fi Mode**: Add vinyl crackle, downsample audio, and apply low-pass filters for an old-school vibe.
- **Export Options**: Download generated audio (WAV) and MIDI files for your projects.

## Preview  

You can access the app at think [link](https://www.drumgenai.streamlit.app).

## Installation and Setup

### Step 1: Clone the Repository  

```shell
git clone https://github.com/maxmeadowcroft/gen_ai_drum_generator.git  
cd gen_ai_drum_generator  
```

### Step 2: Set Up the Environment  
  
- **Install Python Dependencies** Make sure you have Python 3.8+ installed, then install the required packages:  

```Shell
pip install -r requirements.txt  
```

- **Install System-Level Dependencies** If deploying, install any additional packages listed in `packages.txt`.

### Step 3: Run the Application  
  
Launch the Streamlit app locally:  
  
```Shell
streamlit run app/app.py  
```
  
### Optional: Prepare the Environment  
  
- **Custom Audio Samples**: Replace the default `kick.wav`, `snare.wav`, or `hh.wav` in the `app/static/` folder.
- **Add Custom Models**: Place additional models in the `models/` directory.

## Usage

1. **Load the App**: Access the app in your web browser via the URL shown in the terminal.
2. **Select a Model**: Choose between `lstm` and `transformer` models.
3. **Adjust Parameters**:
    - **BPM**: Set the tempo for your drum pattern.
    - **Threshold, Noise, and Temperature**: Fine-tune the creativity and style of the generated patterns.
4. **Upload Samples** (Optional): Provide custom kick, snare, and hi-hat samples.
5. **Enable Lo-fi Mode**: Add a retro vibe with noise and low-pass filters.
6. **Generate and Download**:
    - WAV: High-quality audio file.
    - MIDI: Separate MIDI tracks for kick, snare, and hi-hat.

## Project Structure
  
```  
gen_ai_drum_generator/  
├── app/  
│   ├── app.py              # Main Streamlit app  
│   ├── static/             # Default audio samples  
│   └── templates/          # Placeholder for optional templates  
├── models/                 # Pre-trained ML models  
├── data/                   # Dataset and processed files  
├── requirements.txt        # Python dependencies  
├── packages.txt            # System dependencies  
├── README.md               # Project documentation  
├── LICENSE                 # License information  
└── .gitignore              # Ignore unnecessary files in version control  
```  
  
## Technologies Used

- **Streamlit**: For the interactive web app interface.
- **TensorFlow/Keras**: For generative AI models.
- **Pydub**: For audio processing and effects.
- **Mido**: For MIDI file generation.
  
## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:

- Enhancements to existing features.
- Additional drum models.
- Bug fixes or optimizations. 
  
## License

This project is licensed under the MIT License.

## Contact

Have questions or suggestions? Reach out:

- **GitHub Issues**: Submit a ticket in the [issues section](https://github.com/maxmeadowcroft/gen_ai_drum_generator/issues).
- **Email**: maxmeadowcroft61@gmail.com