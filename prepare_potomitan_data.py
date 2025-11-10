#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour préparer les données d'entraînement depuis le dataset POTOMITAN
"""

from datasets import load_dataset
import json
from pathlib import Path
import pandas as pd

def load_and_explore_potomitan():
    """Charge et explore le dataset POTOMITAN"""
    
    print("Chargement du dataset POTOMITAN...")
    try:
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("POTOMITAN/potomitan-gcf-fr-translation")
        print("✓ Dataset chargé avec succès!")
        
        # Explorer la structure
        print(f"\nStructure du dataset: {ds}")
        
        # Regarder les différentes splits
        for split_name, split_data in ds.items():
            print(f"\n{split_name}: {len(split_data)} exemples")
            
            # Afficher les colonnes
            if len(split_data) > 0:
                example = split_data[0]
                print(f"Colonnes: {list(example.keys())}")
                
                # Afficher quelques exemples
                print("\nPremiers exemples:")
                for i in range(min(3, len(split_data))):
                    ex = split_data[i]
                    print(f"  {i+1}. Créole: {ex.get('gcf', 'N/A')}")
                    print(f"     Français: {ex.get('fr', 'N/A')}")
                    print(f"     Section: {ex.get('section', 'N/A')}")
                    print()
        
        return ds
        
    except Exception as e:
        print(f"Erreur: {e}")
        print("\nPour accéder à ce dataset, vous devez:")
        print("1. Créer un compte sur https://huggingface.co")
        print("2. Installer huggingface-hub: pip install huggingface-hub")
        print("3. Se connecter: huggingface-cli login")
        return None

def create_training_templates(ds, output_dir="./training_templates"):
    """Crée des templates pour l'entraînement basés sur le dataset POTOMITAN"""
    
    if ds is None:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Utiliser la split 'train' ou la première disponible
    train_data = ds['train'] if 'train' in ds else list(ds.values())[0]
    
    # Extraire les données
    transcripts = []
    metadata_for_audio = []
    
    for i, example in enumerate(train_data):
        creole_text = example.get('gcf', '') or example.get('guadeloupean_creole', '')
        french_text = example.get('fr', '')
        section = example.get('section', '')
        
        if creole_text and len(creole_text.strip()) > 0:
            # Pour les transcriptions de référence
            transcripts.append({
                "id": f"potomitan_{i:04d}",
                "creole": creole_text.strip(),
                "french": french_text.strip(),
                "section": section,
                "emergency": example.get('emergency_context', False)
            })
            
            # Template pour les métadonnées audio (à compléter manuellement)
            metadata_for_audio.append({
                "audio_file": f"potomitan_{i:04d}.mp3",  # À enregistrer
                "transcript": creole_text.strip(),
                "french_reference": french_text.strip(),
                "section": section,
                "notes": "À enregistrer - utilisez cette transcription comme guide"
            })
    
    # Sauvegarder les transcriptions de référence
    transcripts_file = output_path / "potomitan_transcriptions_reference.json"
    with open(transcripts_file, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    
    # Sauvegarder le template pour les métadonnées audio
    metadata_template = output_path / "metadata_template.json"
    with open(metadata_template, 'w', encoding='utf-8') as f:
        json.dump(metadata_for_audio[:10], f, ensure_ascii=False, indent=2)  # Seulement 10 premiers pour l'exemple
    
    # Créer un CSV pour faciliter l'organisation
    df = pd.DataFrame(transcripts)
    csv_file = output_path / "potomitan_transcriptions.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Créer un guide d'utilisation
    guide_file = output_path / "GUIDE_UTILISATION.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write("""# Guide d'Utilisation des Templates POTOMITAN

## Fichiers Créés

1. **potomitan_transcriptions_reference.json** - Toutes les transcriptions du dataset POTOMITAN
2. **metadata_template.json** - Template pour vos enregistrements audio
3. **potomitan_transcriptions.csv** - Format CSV pour faciliter la consultation
4. **GUIDE_UTILISATION.md** - Ce guide

## Étapes pour Créer vos Données d'Entraînement

### 1. Choisir les Transcriptions
- Consultez le fichier CSV ou JSON pour choisir les phrases à enregistrer
- Commencez par les phrases courtes et simples
- Variez les thèmes (urgences, conversation quotidienne, etc.)

### 2. Enregistrer les Audios
- Utilisez un micro de bonne qualité
- Enregistrez dans un environnement calme
- Format recommandé: WAV 16kHz mono ou MP3 de bonne qualité
- Nommez les fichiers selon le template: potomitan_XXXX.mp3

### 3. Créer le metadata.json
- Copiez le contenu de metadata_template.json
- Ajoutez vos enregistrements réels
- Vérifiez que les chemins des fichiers audio sont corrects

### 4. Structure du Dossier Final
```
data/
├── metadata.json
├── potomitan_0001.mp3
├── potomitan_0002.mp3
└── ...
```

## Conseils

- **Qualité**: Privilégiez la qualité à la quantité (10 bonnes heures > 50 heures moyennes)
- **Diversité**: Enregistrez avec différents locuteurs si possible
- **Vérification**: Écoutez vos enregistrements pour vérifier qu'ils correspondent aux transcriptions
- **Progression**: Commencez petit (1-2h) puis augmentez graduellement

## Support

Pour des questions sur les transcriptions créoles, consultez:
- Le dataset original: https://huggingface.co/datasets/POTOMITAN/potomitan-gcf-fr-translation
- La documentation POTOMITAN
""")
    
    print(f"\n✓ Templates créés dans: {output_path}")
    print(f"  - {len(transcripts)} transcriptions de référence")
    print(f"  - Template pour {len(metadata_for_audio)} enregistrements")
    print(f"  - Guide d'utilisation inclus")
    
    return transcripts

def show_statistics(ds):
    """Affiche des statistiques sur le dataset"""
    
    if ds is None:
        return
    
    print("\n" + "="*50)
    print("STATISTIQUES DU DATASET POTOMITAN")
    print("="*50)
    
    for split_name, split_data in ds.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Nombre d'exemples: {len(split_data)}")
        
        # Statistiques sur les longueurs de texte
        if len(split_data) > 0:
            creole_lengths = []
            french_lengths = []
            sections = set()
            
            for example in split_data:
                creole_text = example.get('gcf', '') or example.get('guadeloupean_creole', '')
                french_text = example.get('fr', '')
                section = example.get('section', '')
                
                if creole_text:
                    creole_lengths.append(len(creole_text))
                if french_text:
                    french_lengths.append(len(french_text))
                if section:
                    sections.add(section)
            
            if creole_lengths:
                print(f"  Longueur moyenne (créole): {sum(creole_lengths)/len(creole_lengths):.1f} caractères")
                print(f"  Longueur min/max (créole): {min(creole_lengths)}/{max(creole_lengths)}")
            
            if sections:
                print(f"  Sections: {', '.join(sorted(sections))}")

if __name__ == "__main__":
    print("="*60)
    print("PRÉPARATION DES DONNÉES DEPUIS LE DATASET POTOMITAN")
    print("="*60)
    
    # Charger et explorer le dataset
    ds = load_and_explore_potomitan()
    
    if ds:
        # Afficher les statistiques
        show_statistics(ds)
        
        # Créer les templates
        transcripts = create_training_templates(ds)
        
        print("\n" + "="*60)
        print("PROCHAINES ÉTAPES:")
        print("="*60)
        print("1. Consultez le dossier './training_templates/'")
        print("2. Choisissez les phrases à enregistrer")
        print("3. Enregistrez les audios correspondants")
        print("4. Créez votre dossier 'data/' avec metadata.json")
        print("5. Lancez l'entraînement avec FineTunePotomitan.py")
        print("="*60)
    else:
        print("\nImpossible de charger le dataset. Vérifiez votre connexion Hugging Face.")