import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
from pathlib import Path
import json
from jiwer import wer
from tqdm import tqdm

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

def load_test_data(data_dir="./audio"):
    """Charge les données de test depuis le JSON"""
    data_path = Path(data_dir)
    json_file = data_path / "gcf_fr_translation_dataset_v3.json"
    
    if not json_file.exists():
        print(f"❌ Fichier {json_file} non trouvé")
        return []
    
    print(f"Chargement depuis {json_file}...")
    try:
        with open(json_file, "r", encoding="utf-8-sig") as f:
            metadata = json.load(f)
    except UnicodeDecodeError:
        with open(json_file, "r", encoding="latin-1") as f:
            metadata = json.load(f)
    
    test_samples = []
    for item in metadata:
        audio_field = item.get('audio')
        transcript_field = item.get('gcf')
        
        if audio_field and transcript_field:
            # Retirer le préfixe 'audio/' si présent
            if audio_field.startswith('audio/'):
                audio_filename = audio_field[6:]
            elif audio_field.startswith('audio\\'):
                audio_filename = audio_field[6:]
            else:
                audio_filename = audio_field
            
            audio_path = data_path / audio_filename
            
            if audio_path.exists():
                test_samples.append({
                    'audio': str(audio_path),
                    'reference': transcript_field
                })
    
    print(f"✓ {len(test_samples)} échantillons de test chargés")
    return test_samples

def calculate_wer(model_path="./whisper-finetuned-potomitan", data_dir="./audio", num_samples=None):
    """Calcule le WER sur le dataset de test"""
    
    print("="*70)
    print("ÉVALUATION DU MODÈLE WHISPER - CALCUL DU WER")
    print("="*70)
    
    # Charger le modèle
    model, processor, device = load_model_and_processor(model_path)
    
    # Charger les données de test
    test_samples = load_test_data(data_dir)
    
    if not test_samples:
        print("❌ Aucune donnée de test disponible")
        return
    
    # Limiter le nombre d'échantillons si spécifié
    if num_samples:
        test_samples = test_samples[:num_samples]
        print(f"Évaluation sur {num_samples} échantillons")
    
    print(f"\n🚀 Début de l'évaluation sur {len(test_samples)} échantillons...\n")
    
    references = []
    predictions = []
    
    # Transcription de chaque audio
    for sample in tqdm(test_samples, desc="Transcription"):
        try:
            prediction = transcribe_audio(sample['audio'], model, processor, device)
            reference = sample['reference']
            
            predictions.append(prediction)
            references.append(reference)
            
        except Exception as e:
            print(f"\n⚠️  Erreur sur {sample['audio']}: {e}")
            continue
    
    # Calculer le WER
    if references and predictions:
        wer_score = wer(references, predictions)
        
        print("\n" + "="*70)
        print("📊 RÉSULTATS DE L'ÉVALUATION")
        print("="*70)
        print(f"Échantillons évalués: {len(references)}")
        print(f"WER (Word Error Rate): {wer_score*100:.2f}%")
        print("="*70)
        
        # Qualité du modèle
        if wer_score < 0.1:
            quality = "🎯 Excellente qualité"
        elif wer_score < 0.2:
            quality = "✅ Bonne qualité"
        elif wer_score < 0.3:
            quality = "⚠️  Qualité acceptable"
        else:
            quality = "❌ Qualité médiocre"
        
        print(f"\nQualité du modèle: {quality}")
        
        # Afficher quelques exemples
        print("\n📝 Exemples de transcriptions:")
        print("-"*70)
        for i in range(min(5, len(references))):
            print(f"\nÉchantillon {i+1}:")
            print(f"  Référence  : {references[i]}")
            print(f"  Prédiction : {predictions[i]}")
        print("-"*70)
        
        # Sauvegarder les résultats
        results = {
            'wer': wer_score,
            'num_samples': len(references),
            'examples': [
                {'reference': ref, 'prediction': pred}
                for ref, pred in zip(references[:10], predictions[:10])
            ]
        }
        
        results_file = Path(model_path) / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Résultats sauvegardés dans {results_file}")
    
    else:
        print("\n❌ Aucune transcription réussie")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Évaluer le modèle Whisper fine-tuné")
    parser.add_argument("--model-path", default="./whisper-finetuned-potomitan", help="Chemin vers le modèle")
    parser.add_argument("--data-dir", default="./audio", help="Répertoire des données")
    parser.add_argument("--num-samples", type=int, default=None, help="Nombre d'échantillons à évaluer (tous par défaut)")
    
    args = parser.parse_args()
    
    calculate_wer(args.model_path, args.data_dir, args.num_samples)
