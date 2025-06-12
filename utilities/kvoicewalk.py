import datetime
import os
import random
from typing import Any

import soundfile as sf
import torch
from tqdm import tqdm

from main import OUT_DIR
from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.speech_generator import SpeechGenerator
from utilities.voice_generator import VoiceGenerator


class KVoiceWalk:
    def __init__(self, target_audio: str, target_text: str, other_text: str, voice_folder: str, interpolate_start: bool,
                 population_limit: int, starting_voice: str, output_name: str) -> None:
        self.target_audio = target_audio
        self.target_text = target_text
        self.other_text = other_text
        self.initial_selector = InitialSelector(target_audio,target_text,other_text,voice_folder=voice_folder)
        voices: list[torch.Tensor] = []
        if interpolate_start:
            voices = self.initial_selector.interpolate_search(population_limit)
        else:
            voices = self.initial_selector.top_performer_start(population_limit)
        self.speech_generator = SpeechGenerator()
        self.fitness_scorer = FitnessScorer(target_audio)
        self.voice_generator = VoiceGenerator(voices,starting_voice)
        # Either the mean or the supplied voice tensor
        self.starting_voice = self.voice_generator.starting_voice
        self.output_name = output_name

    def random_walk(self,step_limit: int):
        # Score Initial Voice
        best_voice = self.starting_voice
        best_results = self.score_voice(self.starting_voice)
        t = tqdm()
        t.write(f'Target Sim:{best_results["target_similarity"]:.3f}, Self Sim:{best_results["self_similarity"]:.3f}, Feature Sim:{best_results["feature_similarity"]:.2f}, Score:{best_results["score"]:.2f}')

        # Create Results Directory
        results_dir = f"{OUT_DIR}/{self.output_name}_{self.target_audio}_{str(datetime.datetime.now())}"
        os.makedirs(results_dir, exist_ok=True)

        # Random Walk Loop
        for i in tqdm(range(step_limit)):
            # TODO: Expose to CLI
            diversity = random.uniform(0.01,0.15)
            voice = self.voice_generator.generate_voice(best_voice,diversity)

            # Early function return saves audio generation compute
            min_similarity = best_results["target_similarity"] * 0.98
            voice_results = self.score_voice(voice,min_similarity)

            # Set new winner if score is better
            if voice_results["score"] > best_results["score"]:
                best_results = voice_results
                best_voice = voice
                t.write(f'Step:{i:<4} Target Sim:{best_results["target_similarity"]:.3f} Self Sim:{best_results["self_similarity"]:.3f} Feature Sim:{best_results["feature_similarity"]:.3f} Score:{best_results["score"]:.2f} Diversity:{diversity:.2f}')
                # Save results so folks can listen
                torch.save(best_voice,
                           f'{results_dir}/{self.output_name}_{self.target_audio}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{i}.pt')
                sf.write(f'{results_dir}/{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{i}.wav', best_results["audio"], 24000)

        # Print Final Results for Random Walk
        # TODO solve for tracking naming of tensor.
        print("Random Walk Final Results")
        print(f"Duration: {t.format_dict['elapsed']}")
        print(f"Voice Name: {self.output_name}")
        print(f"Best Score: {best_results['score']:.2f}_")
        print(f"Best Similarity: {best_results['target_similarity']:.2f}_")
        print(f"Random Walk pt and wav files ---> {results_dir}")
        return

    def score_voice(self,voice: torch.Tensor,min_similarity: float = 0.0) -> dict[str,Any]:
        """Using a harmonic mean calculation to provide a score for the voice in similarity"""
        audio = self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity(audio)
        results: dict[str,Any] = {
            'audio': audio
        }
        # Bail early and save the compute if the similarity sucks
        if target_similarity > min_similarity:
            audio2 = self.speech_generator.generate_audio(self.other_text, voice)
            results.update(self.fitness_scorer.hybrid_similarity(audio,audio2,target_similarity))
        else:
            results["score"] = 0.0
            results["target_similarity"] = target_similarity

        return results
