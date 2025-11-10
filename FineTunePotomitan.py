import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, Audio, load_dataset
import librosa
import numpy as np
from pathlib import Path
import json
import os

class WhisperDataset:
    def __init__(self, audio_files, transcripts, processor):
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.processor = processor
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Charger l'audio
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Traiter l'audio et le texte
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(self.transcripts[idx], return_tensors="pt").input_ids
        
        return {
            "input_features": inputs.input_features.squeeze(),
            "labels": labels.squeeze()
        }

class WhisperHuggingFaceDataset:
    def __init__(self, hf_dataset, processor):
        """Dataset wrapper pour les données Hugging Face"""
        self.dataset = hf_dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extraire l'audio (peut être sous différents noms)
        audio_data = item.get('audio') or item.get('speech') or item.get('sound')
        if audio_data is None:
            raise ValueError(f"Aucun champ audio trouvé dans l'item {idx}")
        
        # L'audio est déjà un dict avec 'array' et 'sampling_rate'
        audio_array = audio_data['array']
        sampling_rate = audio_data['sampling_rate']
        
        # Rééchantillonner à 16kHz si nécessaire
        if sampling_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        
        # Extraire le texte de transcription
        transcript = item.get('transcription') or item.get('text') or item.get('sentence')
        if transcript is None:
            raise ValueError(f"Aucun champ transcription trouvé dans l'item {idx}")
        
        # Traiter avec Whisper
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(transcript, return_tensors="pt").input_ids
        
        return {
            "input_features": inputs.input_features.squeeze(),
            "labels": labels.squeeze()
        }

def load_potomitan_audio_dataset():
    """Charge le dataset POTOMITAN audio/transcription depuis Hugging Face"""
    try:
        print("Chargement du dataset POTOMITAN audio/transcription...")
        # Dataset avec paires audio/transcription
        ds = load_dataset("POTOMITAN/potomitan-gcf-transcription")
        return ds
    except Exception as e:
        print(f"Erreur lors du chargement du dataset POTOMITAN audio: {e}")
        print("Ce dataset nécessite une autorisation d'accès.")
        print("Étapes requises:")
        print("1. Connectez-vous: huggingface-cli login")
        print("2. Demandez l'accès sur: https://huggingface.co/datasets/POTOMITAN/potomitan-gcf-transcription")
        print("3. Attendez l'approbation de l'équipe POTOMITAN")
        return None

def load_potomitan_translation_dataset():
    """Charge le dataset POTOMITAN traduction depuis Hugging Face (fallback)"""
    try:
        print("Chargement du dataset POTOMITAN traduction (fallback)...")
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("POTOMITAN/potomitan-gcf-fr-translation")
        return ds
    except Exception as e:
        print(f"Erreur lors du chargement du dataset POTOMITAN traduction: {e}")
        print("Assurez-vous d'être connecté avec: huggingface-cli login")
        return None

def load_data_from_local(data_dir):
    """Charge les paires audio/texte depuis un répertoire local"""
    audio_files = []
    transcripts = []
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Le répertoire {data_dir} n'existe pas.")
        return [], []
    
    metadata_file = data_path / "metadata.json"
    if not metadata_file.exists():
        print(f"Le fichier {metadata_file} n'existe pas.")
        return [], []
    
    # Charger le fichier JSON avec les paires
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    for item in metadata:
        audio_path = data_path / item["audio_file"]
        if audio_path.exists():
            audio_files.append(str(audio_path))
            transcripts.append(item["transcript"])
        else:
            print(f"Fichier audio manquant: {audio_path}")
    
    return audio_files, transcripts

def create_transcripts_from_potomitan(potomitan_ds, output_file="potomitan_transcripts.json"):
    """Extrait les textes créoles du dataset POTOMITAN pour créer un guide de transcription"""
    if potomitan_ds is None:
        return
    
    transcripts = []
    
    # Utiliser la version par défaut (v3)
    train_data = potomitan_ds['train'] if 'train' in potomitan_ds else potomitan_ds
    
    for i, example in enumerate(train_data):
        # Extraire le texte en créole guadeloupéen
        creole_text = example.get('gcf', '') or example.get('guadeloupean_creole', '')
        
        if creole_text and len(creole_text.strip()) > 0:
            transcripts.append({
                "id": f"potomitan_{i:04d}",
                "transcript": creole_text.strip(),
                "french_translation": example.get('fr', ''),
                "section": example.get('section', ''),
                "emergency_context": example.get('emergency_context', False)
            })
    
    # Sauvegarder les transcriptions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    
    print(f"Extrait {len(transcripts)} transcriptions du dataset POTOMITAN")
    print(f"Sauvegardé dans: {output_file}")
    
    return transcripts

def fine_tune_whisper(model_name="misterkissi/whisper-small-haitian-creole", data_dir="./data", output_dir="./whisper-finetuned-potomitan", use_potomitan_dataset=True):
    # Charger le modèle et le processeur pré-entraîné pour le créole haïtien
    print(f"Chargement du modèle: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    dataset = None
    
    # Option 1: Essayer d'utiliser le dataset POTOMITAN audio/transcription
    if use_potomitan_dataset:
        print("\n=== Tentative de chargement du dataset POTOMITAN audio ===")
        potomitan_audio_ds = load_potomitan_audio_dataset()
        
        if potomitan_audio_ds:
            print(f"✓ Dataset POTOMITAN audio chargé!")
            # Utiliser le split 'train' ou le premier disponible
            train_split = potomitan_audio_ds.get('train', list(potomitan_audio_ds.values())[0])
            print(f"  Nombre d'exemples audio: {len(train_split)}")
            
            # Créer le dataset Whisper depuis les données HuggingFace
            dataset = WhisperHuggingFaceDataset(train_split, processor)
            print(f"✓ Dataset prêt avec {len(dataset)} exemples audio/transcription")
            
        else:
            print("❌ Impossible de charger le dataset POTOMITAN audio")
            print("Tentative avec le dataset de traduction comme fallback...")
            
            potomitan_translation_ds = load_potomitan_translation_dataset()
            if potomitan_translation_ds:
                transcripts_data = create_transcripts_from_potomitan(potomitan_translation_ds)
                print(f"Dataset POTOMITAN traduction chargé avec {len(transcripts_data)} exemples")
                print("Note: Ces transcriptions peuvent servir de référence pour créer vos enregistrements audio.")
    
    # Option 2: Utiliser les données audio locales si pas de dataset HF ou en complément
    if dataset is None:
        print("\n=== Chargement des données audio locales ===")
        audio_files, transcripts = load_data_from_local(data_dir)
        
        if len(audio_files) > 0:
            dataset = WhisperDataset(audio_files, transcripts, processor)
            print(f"✓ Dataset local créé avec {len(dataset)} fichiers audio")
        else:
            print("❌ Aucune donnée trouvée (ni HuggingFace ni locale)")
    
    # Vérifier qu'on a un dataset pour l'entraînement
    if dataset is None:
        print("\n" + "="*60)
        print("AUCUNE DONNÉE DISPONIBLE POUR L'ENTRAÎNEMENT")
        print("="*60)
        print("Options pour obtenir des données:")
        print("\n1. DATASET POTOMITAN AUDIO (Recommandé):")
        print("   - Visitez: https://huggingface.co/datasets/POTOMITAN/potomitan-gcf-transcription")
        print("   - Demandez l'accès au dataset")
        print("   - Connectez-vous: huggingface-cli login")
        print("\n2. DONNÉES LOCALES:")
        print("   - Créez un dossier 'data/'")
        print("   - Placez vos fichiers audio (.mp3, .wav, etc.)")
        print("   - Créez 'metadata.json' avec les transcriptions")
        print("\n3. UTILISEZ LES TRANSCRIPTIONS POTOMITAN:")
        print("   - Lancez: python prepare_potomitan_data.py")
        print("   - Utilisez les transcriptions comme guide pour vos enregistrements")
        print("="*60)
        return
    
    print(f"\n✓ Dataset prêt avec {len(dataset)} exemples pour l'entraînement")
    
    # Arguments d'entraînement optimisés pour le fine-tuning d'un modèle déjà adapté au créole
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # Réduit pour éviter les erreurs de mémoire
        gradient_accumulation_steps=4,  # Augmenté pour compenser la batch size réduite
        warmup_steps=100,               # Réduit car on part d'un modèle déjà adapté
        max_steps=2000,                 # Moins d'étapes nécessaires
        learning_rate=5e-6,             # Learning rate plus bas pour un fine-tuning délicat
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=200,                 # Évaluation plus fréquente
        save_steps=200,
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_drop_last=True,      # Évite les problèmes de batch size
        remove_unused_columns=False,    # Important pour Whisper
    )
    
    # Entraîneur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
    )
    
    # Démarrer l'entraînement
    trainer.train()
    
    # Sauvegarder le modèle
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    print(f"Fine-tuning terminé! Modèle sauvegardé dans: {output_dir}")

def test_model(model_path, audio_file_path):
    """Teste le modèle fine-tuné sur un fichier audio"""
    print(f"Test du modèle: {model_path}")
    
    # Charger le modèle fine-tuné
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Charger et traiter l'audio
    audio, sr = librosa.load(audio_file_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    # Générer la transcription
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    print(f"Transcription: {transcription[0]}")
    return transcription[0]

if __name__ == "__main__":
    # Structure attendue du répertoire data (optionnel si dataset HF disponible):
    # data/
    #   ├── metadata.json  (contient [{"audio_file": "audio1.mp3", "transcript": "texte..."}, ...])
    #   ├── audio1.mp3
    #   ├── audio2.mp3
    #   └── ...
    
    print("="*70)
    print("FINE-TUNING WHISPER POUR LE CRÉOLE GUADELOUPÉEN - PROJET POTOMITAN")
    print("="*70)
    print("🎯 Modèle de base: misterkissi/whisper-small-haitian-creole")
    print("📊 Dataset audio: POTOMITAN/potomitan-gcf-transcription") 
    print("📚 Dataset traduction: POTOMITAN/potomitan-gcf-fr-translation")
    print("="*70)
    
    print("\n🚀 Démarrage du processus de fine-tuning...")
    
    # Lancer le fine-tuning avec priorité sur le dataset POTOMITAN audio
    fine_tune_whisper(
        model_name="misterkissi/whisper-small-haitian-creole",
        data_dir="./data",
        output_dir="./whisper-finetuned-potomitan",
        use_potomitan_dataset=True
    )