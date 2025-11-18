import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, Audio, load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import librosa
import numpy as np
from pathlib import Path
import json
import os

def setup_device():
    """Détecte et configure le device (GPU/CPU) pour l'entraînement"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n🚀 GPU détecté: {gpu_name}")
        print(f"   Mémoire GPU: {gpu_memory:.2f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
        
        # Optimisations CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        return device, True
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  Aucun GPU détecté - Utilisation du CPU")
        print(f"   L'entraînement sera beaucoup plus lent sur CPU")
        print(f"   Conseil: Utilisez Google Colab ou un service cloud avec GPU")
        return device, False

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator pour Whisper - gère le padding des audio et des labels"""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Séparer les inputs et les labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Padding des inputs audio
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Padding des labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Remplacer les tokens de padding par -100 pour ignorer dans la loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Si bos token au début, le supprimer (Whisper ajoute automatiquement le token de début)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

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
    
    # Chercher gcf_fr_translation_dataset_v3.json, metadata.json ou metadata.jsonl
    gcf_translation_json = data_path / "gcf_fr_translation_dataset_v3.json"
    metadata_json = data_path / "metadata.json"
    metadata_jsonl = data_path / "metadata.jsonl"
    
    if gcf_translation_json.exists():
        # Format JSON du dataset de traduction GCF-FR
        print(f"Chargement depuis {gcf_translation_json}")
        try:
            with open(gcf_translation_json, "r", encoding="utf-8-sig") as f:
                metadata = json.load(f)
        except UnicodeDecodeError:
            with open(gcf_translation_json, "r", encoding="latin-1") as f:
                metadata = json.load(f)
        
        for item in metadata:
            # Dans ce dataset: 'audio' contient le chemin, 'gcf' contient le texte créole
            audio_field = item.get('audio')
            transcript_field = item.get('gcf')
            
            if audio_field and transcript_field:
                # Retirer le préfixe 'audio/' si présent car data_path pointe déjà vers ./audio
                if audio_field.startswith('audio/'):
                    audio_filename = audio_field[6:]  # Retirer 'audio/'
                elif audio_field.startswith('audio\\'):
                    audio_filename = audio_field[6:]  # Retirer 'audio\\'
                else:
                    audio_filename = audio_field
                
                audio_path = data_path / audio_filename
                
                if audio_path.exists():
                    audio_files.append(str(audio_path))
                    transcripts.append(transcript_field)
                else:
                    if len(audio_files) <= 3:  # N'afficher que les 3 premières erreurs
                        print(f"Fichier audio manquant: {audio_path}")
        
    elif metadata_jsonl.exists():
        # Format JSONL (une ligne JSON par entrée)
        print(f"Chargement depuis {metadata_jsonl}")
        # Essayer différents encodages
        try:
            with open(metadata_jsonl, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(metadata_jsonl, "r", encoding="latin-1") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(metadata_jsonl, "r", encoding="cp1252") as f:
                    content = f.read()
        
        for line_num, line in enumerate(content.strip().split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line.strip())
                # Détecter le champ audio (peut être 'file_name', 'audio', 'audio_file', etc.)
                audio_field = item.get('file_name') or item.get('audio') or item.get('audio_file')
                # Détecter le champ transcription
                transcript_field = item.get('transcription') or item.get('text') or item.get('transcript')
                
                if audio_field and transcript_field:
                    # Le chemin peut déjà contenir 'audio/' ou non
                    if audio_field.startswith('audio/') or audio_field.startswith('audio\\'):
                        audio_path = data_path / audio_field
                    else:
                        # Essayer d'abord avec audio/, sinon directement
                        audio_path = data_path / "audio" / audio_field
                        if not audio_path.exists():
                            audio_path = data_path / audio_field
                    
                    if audio_path.exists():
                        audio_files.append(str(audio_path))
                        transcripts.append(transcript_field)
                    else:
                        if line_num <= 3:  # N'afficher que les 3 premières erreurs
                            print(f"Fichier audio manquant (ligne {line_num}): {audio_path}")
            except json.JSONDecodeError as e:
                print(f"Erreur JSON ligne {line_num}: {e}")
                    
    elif metadata_json.exists():
        # Format JSON classique
        print(f"Chargement depuis {metadata_json}")
        with open(metadata_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        for item in metadata:
            audio_field = item.get('audio_file') or item.get('file_name') or item.get('audio')
            transcript_field = item.get('transcript') or item.get('transcription') or item.get('text')
            
            if audio_field and transcript_field:
                audio_path = data_path / audio_field
                if audio_path.exists():
                    audio_files.append(str(audio_path))
                    transcripts.append(transcript_field)
                else:
                    print(f"Fichier audio manquant: {audio_path}")
    else:
        print(f"Aucun fichier metadata trouvé dans {data_dir}")
        print("Formats supportés: metadata.json ou metadata.jsonl")
        return [], []
    
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
    # Détecter et configurer le device (GPU/CPU)
    device, has_gpu = setup_device()
    
    # Charger le modèle et le processeur pré-entraîné pour le créole haïtien
    print(f"\nChargement du modèle: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Déplacer le modèle sur le device approprié
    model = model.to(device)
    print(f"✓ Modèle chargé sur {device}")
    
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
    
    # Data collator pour gérer le padding
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Ajuster les paramètres selon GPU/CPU et taille du dataset
    dataset_size = len(dataset)
    
    if has_gpu:
        # Configuration réduite pour éviter OOM (Out Of Memory)
        batch_size = 4  # Réduit de 8 à 4
        gradient_accumulation = 4  # Augmenté de 2 à 4
        fp16_enabled = True
        # Calculer max_steps basé sur la taille du dataset (environ 3 époques)
        max_steps = min(2000, (dataset_size * 3) // (batch_size * gradient_accumulation))
        print(f"\n⚙️  Configuration GPU: batch_size={batch_size}, gradient_accumulation={gradient_accumulation}, fp16=True")
        print(f"   Dataset: {dataset_size} exemples, max_steps={max_steps}")
    else:
        # Configuration réduite pour CPU
        batch_size = 2
        gradient_accumulation = 8
        fp16_enabled = False
        max_steps = min(500, (dataset_size * 2) // (batch_size * gradient_accumulation))
        print(f"\n⚙️  Configuration CPU: batch_size={batch_size}, steps réduits à {max_steps}")

    # Arguments d'entraînement optimisés pour le fine-tuning d'un modèle déjà adapté au créole
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=100,
        max_steps=max_steps,
        learning_rate=5e-6,
        fp16=fp16_enabled,
        eval_strategy="no",
        save_steps=200,
        logging_steps=10,
        report_to=["tensorboard"],
        push_to_hub=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        save_total_limit=3,
        # Optimisations supplémentaires pour GPU
        dataloader_num_workers=4 if has_gpu else 0,
        dataloader_pin_memory=has_gpu,
        # Gradient checkpointing désactivé (cause des conflits avec backward)
        gradient_checkpointing=False,
    )
    
    # Entraîneur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
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
    # Structure attendue du répertoire data:
    # data/
    #   ├── metadata.jsonl  (format JSONL avec file_name et transcription)
    #   ├── audio/
    #   │   ├── audio1.mp3
    #   │   ├── audio2.mp3
    #   │   └── ...
    # OU
    #   ├── metadata.json  (format JSON classique)
    #   ├── audio1.mp3
    #   └── ...
    
    print("="*70)
    print("FINE-TUNING WHISPER POUR LE CRÉOLE GUADELOUPÉEN - PROJET POTOMITAN")
    print("="*70)
    print("🎯 Modèle de base: misterkissi/whisper-small-haitian-creole")
    print("📊 Dataset local: ./audio (1808 exemples)")
    print("📚 Dataset HuggingFace (fallback): POTOMITAN/potomitan-gcf-transcription")
    print("="*70)
    
    print("\n🚀 Démarrage du processus de fine-tuning...")
    
    # Dataset local par défaut (dans le répertoire du projet)
    local_dataset_path = "./audio"
    
    # Vérifier si le dataset local existe
    if Path(local_dataset_path).exists():
        print(f"\n✓ Dataset local trouvé: {local_dataset_path}")
        use_hf = False
        data_dir = local_dataset_path
    else:
        print(f"\n⚠️  Dataset local non trouvé: {local_dataset_path}")
        print("Tentative avec HuggingFace ou ./data...")
        use_hf = True
        data_dir = "./data"
    
    # Lancer le fine-tuning
    fine_tune_whisper(
        model_name="misterkissi/whisper-small-haitian-creole",
        data_dir=data_dir,
        output_dir="./whisper-finetuned-potomitan",
        use_potomitan_dataset=use_hf
    )