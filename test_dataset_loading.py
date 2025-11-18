#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test rapide pour vérifier le chargement du dataset
"""

from pathlib import Path
import sys

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

from FineTunePotomitan import load_data_from_local

def test_loading():
    dataset_path = r"C:\Users\medhi\SourceCode\Dataset\potomitan-gcf-transcription"
    
    print("="*60)
    print("TEST DE CHARGEMENT DU DATASET")
    print("="*60)
    print(f"Chemin: {dataset_path}\n")
    
    audio_files, transcripts = load_data_from_local(dataset_path)
    
    print(f"\n✅ Résultats:")
    print(f"  Fichiers audio chargés: {len(audio_files)}")
    print(f"  Transcriptions: {len(transcripts)}")
    
    if len(audio_files) > 0:
        print(f"\n📋 Exemples (3 premiers):")
        for i in range(min(3, len(audio_files))):
            print(f"\n  {i+1}. Audio: {Path(audio_files[i]).name}")
            print(f"     Transcription: {transcripts[i][:80]}{'...' if len(transcripts[i]) > 80 else ''}")
        
        print(f"\n✅ Dataset prêt pour l'entraînement!")
        print(f"   Total: {len(audio_files)} paires audio/transcription")
    else:
        print(f"\n❌ Aucune donnée chargée!")

if __name__ == "__main__":
    test_loading()
