import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime

def load_model_and_processor(model_path):
    """Charge le modèle fine-tuné et le processeur"""
    print(f"Chargement du modèle depuis {model_path}...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Déplacer sur GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ Modèle chargé sur {device}")
    return model, processor, device

def transcribe_audio(audio_path, model, processor, device):
    """Transcrit un fichier audio avec le modèle"""
    try:
        # Charger l'audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Préparer l'input
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        # Générer la transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Décoder
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f"\n⚠️  Erreur sur {audio_path}: {e}")
        return None

def find_all_audio_files(dataset_dir):
    """Trouve tous les fichiers audio MP3 dans le dataset et ses sous-répertoires"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Le répertoire {dataset_dir} n'existe pas")
        return []
    
    # Chercher récursivement tous les fichiers .mp3
    audio_files = list(dataset_path.rglob("*.mp3"))
    
    print(f"✓ {len(audio_files)} fichiers audio trouvés dans {dataset_dir}")
    return audio_files

def transcribe_dataset(model_path="./whisper-finetuned-potomitan", 
                       dataset_dir=r"C:\Users\medhi\SourceCode\Dataset\potomitan-gcf-transcription",
                       output_file="transcriptions_potomitan.json"):
    """Transcrit tous les fichiers audio du dataset"""
    
    print("="*70)
    print("TRANSCRIPTION DU DATASET POTOMITAN")
    print("="*70)
    print(f"📁 Dataset: {dataset_dir}")
    print(f"🤖 Modèle: {model_path}")
    print("="*70)
    
    # Charger le modèle
    model, processor, device = load_model_and_processor(model_path)
    
    # Trouver tous les fichiers audio
    audio_files = find_all_audio_files(dataset_dir)
    
    if not audio_files:
        print("\n❌ Aucun fichier audio trouvé")
        return
    
    print(f"\n🚀 Début de la transcription de {len(audio_files)} fichiers...\n")
    
    # Transcriptions
    results = []
    successful = 0
    failed = 0
    
    for audio_file in tqdm(audio_files, desc="Transcription"):
        transcription = transcribe_audio(str(audio_file), model, processor, device)
        
        if transcription:
            results.append({
                'file': str(audio_file.relative_to(dataset_dir)),
                'absolute_path': str(audio_file),
                'transcription': transcription,
                'timestamp': datetime.now().isoformat()
            })
            successful += 1
        else:
            failed += 1
    
    # Sauvegarder les résultats
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("📊 RÉSULTATS DE LA TRANSCRIPTION")
    print("="*70)
    print(f"✅ Transcriptions réussies: {successful}")
    print(f"❌ Échecs: {failed}")
    print(f"📄 Total: {len(audio_files)}")
    print(f"💾 Résultats sauvegardés dans: {output_path.absolute()}")
    print("="*70)
    
    # Afficher quelques exemples
    if results:
        print("\n📝 Exemples de transcriptions:")
        print("-"*70)
        for i, result in enumerate(results[:5]):
            print(f"\nFichier {i+1}: {result['file']}")
            print(f"  Transcription: {result['transcription']}")
        print("-"*70)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcrire tous les fichiers audio d'un dataset")
    parser.add_argument("--model-path", default="./whisper-finetuned-potomitan", 
                        help="Chemin vers le modèle fine-tuné")
    parser.add_argument("--dataset-dir", 
                        default=r"C:\Users\medhi\SourceCode\Dataset\potomitan-gcf-transcription",
                        help="Répertoire du dataset")
    parser.add_argument("--output", default="transcriptions_potomitan.json",
                        help="Fichier de sortie JSON")
    
    args = parser.parse_args()
    
    transcribe_dataset(args.model_path, args.dataset_dir, args.output)
