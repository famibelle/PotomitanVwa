# Fine-tuning Whisper pour le Créole Guadeloupéen - Projet POTOMITAN

Ce projet permet de fine-tuner un modèle Whisper pour le créole guadeloupéen (Karukéya/GCF) en utilisant les datasets POTOMITAN. Le projet prend en charge à la fois les données audio/transcription directes et les données de traduction comme référence.

## Sources de Données

### 🎯 **Dataset Principal** (Recommandé)
- **POTOMITAN Audio/Transcription:** `POTOMITAN/potomitan-gcf-transcription`
- Contient des paires audio/transcription en créole guadeloupéen
- Extraites d'émissions radio/TV authentiques
- **Status:** Dataset protégé - [Demander l'accès](https://huggingface.co/datasets/POTOMITAN/potomitan-gcf-transcription)

### 📚 **Dataset de Référence**
- **POTOMITAN Traduction:** `POTOMITAN/potomitan-gcf-fr-translation`
- Paires de traduction français ↔ créole guadeloupéen
- Utilisé comme référence pour créer vos propres enregistrements

### 🤖 **Modèle de Base**
- **Whisper Créole Haïtien:** `misterkissi/whisper-small-haitian-creole`
- Point de départ pré-entraîné pour les langues créoles
- [Lien du modèle](https://hf.co/misterkissi/whisper-small-haitian-creole)

## Structure du Projet

```
PotomitanVwa/
├── FineTunePotomitan.py     # Script principal de fine-tuning
├── test_whisper.py          # Script de test et comparaison
├── requirements.txt         # Dépendances Python
├── example_metadata.json    # Exemple de structure des données
├── README.md               # Ce fichier
└── data/                   # Répertoire pour vos données (à créer)
    ├── metadata.json       # Fichier de métadonnées
    ├── audio1.mp3         # Fichiers audio
    ├── audio2.mp3
    └── ...
```

## Installation

### 1. Cloner le Repository
```bash
git clone <votre-repo-url>
cd PotomitanVwa
```

### 2. Créer un Environnement Virtuel
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### 3. Installer les Dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration Hugging Face
```bash
# Installer le CLI Hugging Face
pip install huggingface-hub

# Se connecter
huggingface-cli login
```

## Préparation des Données

1. Créez un dossier `data/` dans le répertoire du projet
2. Placez vos fichiers audio (.mp3, .wav, etc.) dans ce dossier
3. Créez un fichier `metadata.json` avec la structure suivante :

```json
[
  {
    "audio_file": "audio1.mp3",
    "transcript": "Kijan ou ye jodi a? Mwen byen, mèsi."
  },
  {
    "audio_file": "audio2.mp3",
    "transcript": "Nou bezwen travay ansanm pou nou fè pwojè a réussi."
  }
]
```

**Important:** Assurez-vous que les transcriptions sont en créole haïtien correct.

## Utilisation

### 1. Vérification de l'Accès aux Données
```bash
# Vérifier l'accès aux datasets POTOMITAN
python check_potomitan_access.py
```

### 2. Option A: Avec Dataset POTOMITAN Audio (Recommandé)
Si vous avez accès au dataset audio :
```bash
python FineTunePotomitan.py
```

### 2. Option B: Préparation de Données Personnalisées
Si vous n'avez pas accès au dataset audio :
```bash
# Générer les templates depuis le dataset de traduction
python prepare_potomitan_data.py

# Puis créer vos enregistrements et lancer l'entraînement
python FineTunePotomitan.py
```

### 3. Test du Modèle
```bash
# Test avec le modèle de base seulement
python test_whisper.py ./chemin/vers/audio.mp3

# Test et comparaison avec le modèle fine-tuné
python test_whisper.py ./chemin/vers/audio.mp3 ./whisper-finetuned-potomitan
```

## Paramètres d'Entraînement

Le script utilise des paramètres optimisés pour le fine-tuning d'un modèle déjà adapté :

- **Batch size:** 4 (réduit pour éviter les problèmes de mémoire)
- **Learning rate:** 5e-6 (plus bas que d'habitude car on part d'un modèle déjà adapté)
- **Steps:** 2000 (moins que pour un entraînement from scratch)
- **Warmup:** 100 steps seulement

## Conseils pour de Meilleures Performances

1. **Qualité Audio:** Utilisez des fichiers audio de bonne qualité (16kHz minimum)
2. **Diversité:** Incluez différents locuteurs et accents de la région
3. **Transcriptions:** Vérifiez la précision des transcriptions
4. **Quantité:** Plus vous avez de données, meilleur sera le résultat (minimum 1h d'audio)

## Monitoring

Le script génère des logs TensorBoard. Pour les visualiser :
```bash
tensorboard --logdir ./whisper-finetuned-potomitan/logs
```

## Troubleshooting

- **Erreur de mémoire:** Réduisez `per_device_train_batch_size` à 2 ou 1
- **Modèle ne converge pas:** Augmentez le `learning_rate` à 1e-5
- **Fichiers audio non trouvés:** Vérifiez les chemins dans `metadata.json`

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. **Fork** le projet
2. **Créez** une branche pour votre fonctionnalité (`git checkout -b feature/ma-fonctionnalite`)
3. **Committez** vos changements (`git commit -m 'Ajout de ma fonctionnalité'`)
4. **Push** vers la branche (`git push origin feature/ma-fonctionnalite`)
5. **Ouvrez** une Pull Request

### Types de Contributions Appréciées
- 🐛 Correction de bugs
- ✨ Nouvelles fonctionnalités
- 📚 Amélioration de la documentation
- 🎵 Ajout de données d'entraînement
- 🧪 Tests et validation
- 🌍 Traductions et localisations

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour les détails.

### Licences des Dépendances
- **Datasets POTOMITAN:** Voir https://huggingface.co/datasets/POTOMITAN/
- **Whisper Créole Haïtien:** CC-BY-NC-SA-4.0
- **OpenAI Whisper:** MIT License