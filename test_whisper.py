#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour le modèle Whisper fine-tuné pour le créole haïtien
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import sys
from pathlib import Path

def test_whisper_model(model_path, audio_path):
    """
    Teste le modèle Whisper fine-tuné sur un fichier audio
    
    Args:
        model_path (str): Chemin vers le modèle fine-tuné
        audio_path (str): Chemin vers le fichier audio à transcrire
    """
    
    print(f"Chargement du modèle depuis: {model_path}")
    
    try:
        # Charger le processeur et le modèle
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
        print(f"Traitement du fichier audio: {audio_path}")
        
        # Charger l'audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Traiter l'audio
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Générer la transcription
        print("Génération de la transcription...")
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs.input_features,
                max_length=448,
                num_beams=5,
                early_stopping=True
            )
            
        # Décoder la transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print("="*50)
        print(transcription)
        print("="*50)
        
        return transcription
        
    except Exception as e:
        print(f"Erreur lors du test: {str(e)}")
        return None

def test_base_model(audio_path):
    """
    Teste le modèle de base (avant fine-tuning) pour comparaison
    """
    print("Test du modèle de base (misterkissi/whisper-small-haitian-creole)")
    return test_whisper_model("misterkissi/whisper-small-haitian-creole", audio_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_whisper.py <chemin_audio> [chemin_modele_fine_tune]")
        print("Exemple: python test_whisper.py ./audio/test.mp3 ./whisper-finetuned-potomitan")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not Path(audio_path).exists():
        print(f"Erreur: Le fichier audio {audio_path} n'existe pas.")
        sys.exit(1)
    
    # Tester le modèle de base
    print("1. Test avec le modèle de base...")
    base_result = test_base_model(audio_path)
    
    # Tester le modèle fine-tuné si fourni
    if len(sys.argv) > 2:
        fine_tuned_path = sys.argv[2]
        if Path(fine_tuned_path).exists():
            print("\n2. Test avec le modèle fine-tuné...")
            fine_tuned_result = test_whisper_model(fine_tuned_path, audio_path)
            
            # Comparaison
            print("\n" + "="*60)
            print("COMPARAISON:")
            print("="*60)
            print(f"Modèle de base: {base_result}")
            print(f"Modèle fine-tuné: {fine_tuned_result}")
            print("="*60)
        else:
            print(f"Le modèle fine-tuné {fine_tuned_path} n'existe pas encore.")
            print("Lancez d'abord l'entraînement avec FineTunePotomitan.py")