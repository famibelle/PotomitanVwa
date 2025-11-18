#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour inspecter le dataset POTOMITAN local
"""

import json
from pathlib import Path
import librosa

def inspect_dataset(dataset_path):
    """Inspecte le contenu du dataset local"""
    
    dataset_path = Path(dataset_path)
    
    print("="*60)
    print("INSPECTION DU DATASET POTOMITAN LOCAL")
    print("="*60)
    print(f"Chemin: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ Le répertoire {dataset_path} n'existe pas!")
        return
    
    print("✓ Répertoire trouvé")
    
    # Chercher les fichiers metadata
    metadata_jsonl = dataset_path / "metadata.jsonl"
    metadata_json = dataset_path / "metadata.json"
    audio_dir = dataset_path / "audio"
    
    print(f"\n📂 Structure:")
    print(f"  metadata.jsonl: {'✓' if metadata_jsonl.exists() else '❌'}")
    print(f"  metadata.json:  {'✓' if metadata_json.exists() else '❌'}")
    print(f"  audio/:         {'✓' if audio_dir.exists() else '❌'}")
    
    # Compter les fichiers audio
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
        print(f"\n🎵 Fichiers audio trouvés: {len(audio_files)}")
    
    # Analyser metadata.jsonl
    if metadata_jsonl.exists():
        print(f"\n📋 Analyse de metadata.jsonl:")
        
        entries = []
        with open(metadata_jsonl, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"  ❌ Erreur ligne {i+1}: {e}")
        
        print(f"  Total d'entrées: {len(entries)}")
        
        if entries:
            # Afficher la structure du premier exemple
            print(f"\n  Structure d'un exemple:")
            first_entry = entries[0]
            for key, value in first_entry.items():
                value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"    {key}: {value_preview}")
            
            # Statistiques
            print(f"\n  📊 Statistiques:")
            
            # Longueur des transcriptions
            transcription_field = 'transcription' if 'transcription' in first_entry else 'text'
            if transcription_field in first_entry:
                lengths = [len(e.get(transcription_field, '')) for e in entries]
                print(f"    Transcriptions:")
                print(f"      Moyenne: {sum(lengths)/len(lengths):.1f} caractères")
                print(f"      Min/Max: {min(lengths)}/{max(lengths)} caractères")
            
            # Vérifier les fichiers audio
            audio_field = 'file_name' if 'file_name' in first_entry else 'audio_file'
            if audio_field in first_entry and audio_dir.exists():
                missing = 0
                existing = 0
                for entry in entries:
                    audio_file = audio_dir / entry.get(audio_field, '')
                    if audio_file.exists():
                        existing += 1
                    else:
                        missing += 1
                
                print(f"    Fichiers audio:")
                print(f"      Existants: {existing}")
                print(f"      Manquants: {missing}")
            
            # Afficher quelques exemples
            print(f"\n  📝 Exemples (3 premiers):")
            for i, entry in enumerate(entries[:3], 1):
                trans = entry.get('transcription') or entry.get('text', 'N/A')
                audio = entry.get('file_name') or entry.get('audio_file', 'N/A')
                print(f"    {i}. Audio: {audio}")
                print(f"       Texte: {trans[:80]}{'...' if len(trans) > 80 else ''}")
    
    # Analyser quelques fichiers audio
    if audio_dir.exists() and len(audio_files) > 0:
        print(f"\n🎵 Analyse audio (5 premiers fichiers):")
        for i, audio_file in enumerate(audio_files[:5], 1):
            try:
                audio, sr = librosa.load(audio_file, sr=None, duration=1)
                duration = librosa.get_duration(path=audio_file)
                print(f"  {i}. {audio_file.name}")
                print(f"     Durée: {duration:.2f}s, Sample rate: {sr}Hz")
            except Exception as e:
                print(f"  {i}. {audio_file.name} - Erreur: {e}")
    
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    
    if metadata_jsonl.exists() and audio_dir.exists():
        print("✅ Dataset prêt à être utilisé!")
        print(f"   Lancez: python FineTunePotomitan.py")
    else:
        print("⚠️  Configuration incomplète")
        if not metadata_jsonl.exists():
            print("   - Fichier metadata.jsonl manquant")
        if not audio_dir.exists():
            print("   - Dossier audio/ manquant")

if __name__ == "__main__":
    # Chemin vers le dataset local
    dataset_path = r"C:\Users\medhi\SourceCode\Dataset\potomitan-gcf-transcription"
    
    inspect_dataset(dataset_path)
