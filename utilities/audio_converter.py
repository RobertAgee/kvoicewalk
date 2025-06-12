from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from main import OUT_DIR


# TODO: Consider merge into single audio utilities

def convert_to_wav_mono_24k(target_audio: str) -> str:
    try:
        with sf.SoundFile(target_audio, 'r') as f:
            if f.format != 'WAV' or f.samplerate != 24000 or f.channels != 1:
                # Create output filename with proper extension handling
                audio_path = Path(target_audio)
                converted_audio_file = OUT_DIR + str(audio_path.with_suffix('')) + "_converted.wav"

                # Read the audio data
                audio_data = f.read()

                # Convert to mono if needed
                if f.channels > 1:
                    converted_audio_data = np.mean(audio_data, axis=1)
                else:
                    converted_audio_data = audio_data

                # Resample if needed 
                if f.samplerate != 24000:
                    converted_audio_data = librosa.resample(
                        converted_audio_data,
                        orig_sr=f.samplerate,
                        target_sr=24000
                    )

                # Write processed audio
                sf.write(converted_audio_file, converted_audio_data, samplerate=24000, format='WAV')
                print(f"{target_audio} successfully converted to Mono WAV 24K format: {converted_audio_file}")
                return converted_audio_file
            else:
                print(f"{target_audio} already matches Mono WAV 24K format")
                return target_audio

    except Exception as e:
        raise Exception(f"Error converting {target_audio}: {e}\n")
