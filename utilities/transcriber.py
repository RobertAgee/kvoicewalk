import argparse
import datetime
import os
from pathlib import Path

from faster_whisper import WhisperModel

from main import TEXTS_DIR


# TODO: Consider merge into single audio utilities
# TODO: Integrate into automated workflows

def transcribe(input_audio):
    model_size = "large-v3"
    print('Starting Transcriber...')
    start_time = datetime.datetime.now()
    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

    try:
        # or run on CPU with INT8 !!(more than sufficient for 30s clip)!!
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f'Loading {input_audio}...')
        segments, info = model.transcribe(input_audio, beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        print('Transcribing {input_audio}...')
        transcription = ''
        for segment in segments:
            transcription += segment.text
            # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)) # Optional timestamps if parsing longer audio clips

        transcription_output = Path(TEXTS_DIR / str(f"{input_audio[:-4]}.txt"))
        with open(f"{transcription_output}", "w") as file:
            file.write(f"{transcription}")

        end_time = datetime.datetime.now()
        print(f"Transcription completed in {(end_time - start_time).total_seconds()} seconds")
        print(f"Transcription available at ./texts/{input_audio[:-4]}.txt")
        print(f"{input_audio} Transcription:\n{transcription}")
        return transcription

    except Exception as e:
        print(f"Transcription failed for {input_audio} - Error: {e}")
        return

def main():
    parser = argparse.ArgumentParser(description="Transcribe Your Audio Files:")

    # Common required arguments
    parser.add_argument("--output", type=str,
                        help="Filename or Folder for the target audio.\n"
                             "Folders will transcribe all .wav.\n"
                             "Transcripts will got to ./texts folder.")

    args = parser.parse_args()

    audio_path = args.output
    try:
        if os.path.isdir(audio_path):
            for audio in audio_path:
                if audio.endswith('.wav'):
                    transcribe(audio)
                else:
                    print(f"File Format Error: {audio} is not a .wav file!")
                    continue
        elif audio_path.endswith('.wav'):
            transcribe(audio_path)
        else:
            print(f"Input Format Error: {audio_path} must be a .wav file or a folder!")
    except Exception as e:
        print(f"Error: {e}")
    print('Transcriptions finished.')

if __name__ == "__main__":
    main()
