import argparse
import os

import numpy as np
import soundfile as sf

from utilities.kvoicewalk import KVoiceWalk
from utilities.pytorch_sanitizer import load_multiple_voices
from utilities.speech_generator import SpeechGenerator
from utilities.transcriber import transcribe

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.environ.get("KVOICEWALK_TEXTS_DIR", "../in")
OUT_DIR = os.environ.get("KVOICEWALK_OUT_DIR", "../out")
INTERPOLATED_DIR = os.environ.get("KVOICEWALK_INTERPOLATED_DIR", "../interpolated")
TEXTS_DIR = os.environ.get("KVOICEWALK_TEXTS_DIR", "../texts")
VOICES_DIR = os.environ.get("KVOICEWALK_VOICES_DIR", "../voices")

def main():
    parser = argparse.ArgumentParser(description="A random walk Kokoro voice cloner.")

    # Common required arguments
    parser.add_argument("--target_text", type=str, help="The words contained in the target audio file. Should be around 100-200 tokens (two sentences). Alternatively, can point to a txt file of the transcription.")

    # Optional arguments
    parser.add_argument("--other_text", type=str,
                      help="A segment of text used to compare self similarity. Should be around 100-200 tokens.",
                      default="If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale.")
    parser.add_argument("--voice_folder", type=str,
                        help="Input: filepath to the voices you want to use as part of the random walk.",
                      default="./voices")
    parser.add_argument("--transcribe_start",
                        help='Input: filepath to wav file\nOutput: Transcription .txt in ./texts\nTranscribes a target wav or wav folder. Replaces --target_text')
    parser.add_argument("--interpolate_start",
                      help="Goes through an interpolation search step before random walking",
                      action='store_true')
    parser.add_argument("--population_limit", type=int,
                        help="Limit the number of voices used as part of the random walk",
                      default=10)
    parser.add_argument("--step_limit", type=int,
                        help="Specify the max steps in the random walk",
                      default=10000)
    parser.add_argument("--output", type=str,
                        help="Input: Name for the generated output audio",
                        default="out")
    # Arguments for random walk mode
    group_walk = parser.add_argument_group('Random Walk Mode')
    group_walk.add_argument("--target_audio", type=str,
                            help="Input: filepath to the target audio file.")
    group_walk.add_argument("--starting_voice", type=str,
                            help="Input: filepath to the starting voice tensor")

    # Arguments for test mode
    group_test = parser.add_argument_group('Test Mode')
    group_test.add_argument("--test_voice", type=str,
                            help="Input: filepath to the voice tensor you want to test")

    # Arguments for util mode
    group_util = parser.add_argument_group('Utility Mode')
    # TODO: Add import function
    # TODO: Add convert to/from npy/npz
    group_util.add_argument("--import_voices",
                            help='Input: filepath to npy (safer), npz (safer), pt, or bin\nOutput: ./out folder\nImports voice models into the --voice_folder directory')
    group_util.add_argument("--export_voices", default='npz',
                            help='Exports all voice models from the --voice_folder directory to ./out folder,\nExport formats: npz, bin',
                      action='store_true')
    group_util.add_argument("--transcribe_many",
                            help='Input: filepath to wav file or folder\nOutput: Individualized transcriptions in ./texts folder\nTranscribes a target wav or wav folder. Replaces --target_text', )
    args = parser.parse_args()

    # Normal Transcription Mode
    if args.transcribe_start:
        try:
            if os.path.isfile(args.target_audio) and args.target_audio.endswith('.wav'):
                args.target_text = transcribe(args.target_audio)
            elif os.path.isdir(args.target_audio):
                parser.error("--transcribe_start allows a wav file only. Perhaps you're looking for --transcribe_many?")
            else:
                parser.error("--transcribe_start allows a wav file only. Please check your file or path.")
            return
        except Exception as e:
            print(f"Error during Transcription: {e}")

    # Utility Functions
    # Import Voices
    if args.import_npz:
        if not args.voice_folder:
            parser.error("--voice_folder is required to import a voices .npz file")
        # TODO: Add import function, validation

    # Export Voices
    if args.export_npz:
        if not args.voice_folder:
            parser.error("--voice_folder is required to export a voices .npz file")

        # Collect all .npy file paths
        file_paths = [os.path.join(args.voice_folder, f) for f in os.listdir(args.voice_folder) if f.endswith('.npy')]
        voices = load_multiple_voices(file_paths)

        with open("voices.npz", "wb") as f:
            np.savez(f,**voices)

        return

    # Transcription
    if args.transcribe_many:
        try:
            if os.path.isfile(args.transcribe_many) and args.target_audio.endswith('.wav'):
                transcribe(args.transcribe_many)
                return
            elif os.path.isdir(args.transcribe_many):
                for audio in args.transcribe_many:
                    if audio.endswith('.wav'):
                        transcribe(audio)
                    else:
                        print(f"File Format Error: {audio} is not a .wav file!")
                        continue
            else:
                print(f"Input Format Error: {args.transcribe_many} must be a .wav file or a folder!")
        except Exception as e:
            print(f"Error during Transcription: {e}")

    # If text file, read and assign transcription.
    if args.target_text.endswith('.txt'):
        try:
            if os.path.isfile(args.transcribe_many) and args.target_audio.endswith('.txt'):
                with open(args.target_text, "r") as file:
                    args.target_text = file.read()
            else:
                print(f"Input Format Error: {args.target_text} must be a txt string or a .txt file!!")
        except Exception as e:
            print(f"Error during target_text assignment: {e}")

    # Validate arguments based on mode
    if args.test_voice:
        # Test mode
        if not args.target_text:
            parser.error("--target_text is required when using --test_voice")

        speech_generator = SpeechGenerator()
        audio = speech_generator.generate_audio(args.target_text, args.test_voice)
        sf.write(f"{args.output}.wav", audio, 24000)
    else:
        # Random walk mode
        if not args.target_audio:
            parser.error("--target_audio is required for random walk mode")
        if not args.target_text:
            parser.error("--target_text is required for random walk mode")

        ktb = KVoiceWalk(args.target_audio,
                        args.target_text,
                        args.other_text,
                        args.voice_folder,
                        args.interpolate_start,
                        args.population_limit,
                         args.starting_voice,
                         args.output)
        ktb.random_walk(args.step_limit)

if __name__ == "__main__":
    main()
