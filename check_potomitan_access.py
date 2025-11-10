#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour tester l'accès et explorer le dataset POTOMITAN audio/transcription
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path

def check_huggingface_login():
    """Vérifie si l'utilisateur est connecté à Hugging Face"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✓ Connecté à Hugging Face comme: {user_info.get('name', 'Utilisateur')}")
        return True
    except Exception as e:
        print("❌ Non connecté à Hugging Face")
        print("Pour vous connecter, lancez: huggingface-cli login")
        print("Ou installez d'abord: pip install huggingface-hub")
        return False

def test_audio_dataset_access():
    """Teste l'accès au dataset POTOMITAN audio/transcription"""
    
    print("="*60)
    print("TEST D'ACCÈS AU DATASET POTOMITAN AUDIO/TRANSCRIPTION")
    print("="*60)
    
    # Vérifier la connexion HF
    if not check_huggingface_login():
        return None
    
    print("\n📡 Tentative d'accès au dataset POTOMITAN audio...")
    
    try:
        # Essayer de charger le dataset audio
        ds = load_dataset("POTOMITAN/potomitan-gcf-transcription")
        
        print("✅ SUCCÈS! Accès au dataset POTOMITAN audio accordé!")
        print(f"Structure du dataset: {ds}")
        
        # Explorer le dataset
        for split_name, split_data in ds.items():
            print(f"\n📊 Split '{split_name}': {len(split_data)} exemples")
            
            if len(split_data) > 0:
                example = split_data[0]
                print(f"Colonnes disponibles: {list(example.keys())}")
                
                # Afficher un exemple
                print("\n🎵 Exemple d'entrée:")
                for key, value in example.items():
                    if key == 'audio':
                        print(f"  {key}: Audio array de {len(value['array'])} échantillons à {value['sampling_rate']}Hz")
                    else:
                        print(f"  {key}: {value}")
        
        return ds
        
    except Exception as e:
        error_message = str(e)
        
        if "gated" in error_message.lower() or "request access" in error_message.lower():
            print("🔒 DATASET PROTÉGÉ - ACCÈS REQUIS")
            print("\nLe dataset POTOMITAN audio nécessite une demande d'accès.")
            print("\n📋 ÉTAPES POUR OBTENIR L'ACCÈS:")
            print("1. Visitez: https://huggingface.co/datasets/POTOMITAN/potomitan-gcf-transcription")
            print("2. Cliquez sur 'Request access to this repo'")
            print("3. Remplissez le formulaire de demande")
            print("4. Expliquez votre projet (recherche, éducation, etc.)")
            print("5. Attendez l'approbation de l'équipe POTOMITAN")
            print("\n⏱️  L'approbation peut prendre quelques heures à quelques jours.")
            
        else:
            print(f"❌ Erreur inattendue: {error_message}")
        
        return None

def test_translation_dataset():
    """Teste l'accès au dataset de traduction (fallback)"""
    
    print("\n" + "="*60)
    print("TEST D'ACCÈS AU DATASET POTOMITAN TRADUCTION")
    print("="*60)
    
    try:
        ds = load_dataset("POTOMITAN/potomitan-gcf-fr-translation")
        print("✅ Dataset de traduction accessible!")
        
        # Explorer
        for split_name, split_data in ds.items():
            print(f"\n📊 Split '{split_name}': {len(split_data)} exemples")
            
            if len(split_data) > 0:
                example = split_data[0]
                print(f"Colonnes: {list(example.keys())}")
                
                # Quelques exemples
                print("\n📝 Exemples de traductions:")
                for i in range(min(3, len(split_data))):
                    ex = split_data[i]
                    print(f"  {i+1}. GCF: {ex.get('gcf', 'N/A')}")
                    print(f"     FR:  {ex.get('fr', 'N/A')}")
                    print()
        
        return ds
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def show_dataset_statistics(ds, dataset_name):
    """Affiche les statistiques détaillées du dataset"""
    
    print(f"\n📈 STATISTIQUES - {dataset_name}")
    print("="*50)
    
    for split_name, split_data in ds.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Nombre total: {len(split_data)}")
        
        if len(split_data) > 0:
            # Analyser quelques échantillons pour les stats
            sample_size = min(100, len(split_data))
            sample = split_data.select(range(sample_size))
            
            # Statistiques de transcription
            if 'transcription' in sample[0] or 'text' in sample[0]:
                text_field = 'transcription' if 'transcription' in sample[0] else 'text'
                text_lengths = [len(item[text_field]) for item in sample if item.get(text_field)]
                
                if text_lengths:
                    print(f"  Longueur transcription (chars):")
                    print(f"    Moyenne: {sum(text_lengths)/len(text_lengths):.1f}")
                    print(f"    Min/Max: {min(text_lengths)}/{max(text_lengths)}")
            
            # Statistiques audio si présent
            if 'audio' in sample[0]:
                audio_durations = []
                for item in sample:
                    if 'audio' in item and item['audio']:
                        duration = len(item['audio']['array']) / item['audio']['sampling_rate']
                        audio_durations.append(duration)
                
                if audio_durations:
                    total_hours = sum(audio_durations) / 3600 * len(split_data) / sample_size
                    print(f"  Durée audio:")
                    print(f"    Moyenne par clip: {sum(audio_durations)/len(audio_durations):.2f}s")
                    print(f"    Estimation totale: {total_hours:.1f}h")
                    print(f"    Fréquence: {sample[0]['audio']['sampling_rate']}Hz")

def main():
    """Fonction principale"""
    
    print("🎯 POTOMITAN Dataset Access Checker")
    print("Ce script vérifie l'accès aux datasets POTOMITAN et explore leur contenu.")
    print()
    
    # Test 1: Dataset audio (principal pour Whisper)
    audio_ds = test_audio_dataset_access()
    
    # Test 2: Dataset traduction (fallback)
    translation_ds = test_translation_dataset()
    
    # Statistiques détaillées
    if audio_ds:
        show_dataset_statistics(audio_ds, "AUDIO/TRANSCRIPTION")
    
    if translation_ds:
        show_dataset_statistics(translation_ds, "TRADUCTION")
    
    # Recommandations
    print("\n" + "="*60)
    print("📋 RECOMMANDATIONS")
    print("="*60)
    
    if audio_ds:
        print("✅ PARFAIT! Vous avez accès au dataset audio.")
        print("   → Vous pouvez lancer directement: python FineTunePotomitan.py")
        
    elif translation_ds:
        print("⚠️  Accès partiel: dataset traduction seulement.")
        print("   → Demandez l'accès au dataset audio pour un entraînement optimal")
        print("   → En attendant, utilisez: python prepare_potomitan_data.py")
        
    else:
        print("❌ Aucun dataset accessible.")
        print("   → Connectez-vous d'abord: huggingface-cli login")
        print("   → Demandez l'accès aux datasets POTOMITAN")

if __name__ == "__main__":
    main()