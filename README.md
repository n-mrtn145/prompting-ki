# Hate Speech Detection mit Mistral AI

Dieses Projekt klassifiziert deutsche Texte als Hassrede oder keine Hassrede mithilfe der Mistral AI Batch-API und verschiedenen Prompting-Strategien.

## Überblick

Das Projekt verwendet den `hatespeech_hocon34k`-Datensatz und vergleicht folgende Prompting-Ansätze:

- **Zero-Shot**: Klassifikation ohne Beispiele
- **Single-Shot**: Klassifikation mit einem Beispiel
- **Few-Shot**: Klassifikation mit mehreren Beispielen

Die Ergebnisse werden mit Precision, Recall und F1-Score ausgewertet.

## Projektstruktur

```
prompting/
├── main.py          # Datenvorbereitung und Balancierung des Datensatzes
├── batchFile.py     # Erstellung, Start und Auswertung von Mistral Batch-Jobs
├── create-file.py   # Erstellt JSONL-Batch-Dateien und startet den Batch-Job
├── eveluate.py      # Wertet einen abgeschlossenen Batch-Job aus
└── requirements.txt
```

## Voraussetzungen

- Python 3.9+
- Ein Mistral AI API-Key
- Der Datensatz `hatespeech_hocon34k.csv` im Projektverzeichnis

## Installation

```bash
pip install -r requirements.txt
```

Anschließend den API-Key als Umgebungsvariable setzen:

```bash
# Windows (PowerShell)
$env:MISTRAL_API_KEY = "dein-api-key"

# Linux / macOS
export MISTRAL_API_KEY="dein-api-key"
```

## Verwendung

### 1. Prompt und Dateiname konfigurieren

In `create-file.py` den gewünschten Prompt als Variable definieren und an `create_batch_file()` übergeben:

```python
prompt = "Ist folgende Aussage Hassrede? Antworte nur mit ja oder nein: "

filename = "batch_requests_mein_experiment.jsonl"

batchFile.create_batch_file(df, prompt, filename)
batchFile.start_batch_file(filename)
```

> **Achtung:** Existiert eine Datei mit demselben `filename` bereits, wird sie ohne Warnung überschrieben. Einen eindeutigen Dateinamen pro Experiment verwenden.

### 2. Batch-Job starten

```bash
python create-file.py
```

Gibt die Job-ID des gestarteten Batch-Jobs aus.

### 3. Ergebnisse auswerten

Die Job-ID in `eveluate.py` eintragen und ausführen:

```bash
python eveluate.py
```

> **Achtung:** Zwischen dem Start des Batch-Jobs und der Auswertung darf `aufbereiten()` in `main.py` nicht verändert werden. Die Batch-Ergebnisse werden anhand von Zeilenindizes (`req_0`, `req_1`, ...) den Originaldaten zugeordnet. Ändert sich die Reihenfolge oder Anzahl der Zeilen, stimmen die Vorhersagen nicht mehr mit den richtigen Labels überein.

## Datenvorbereitung

`main.py` führt folgende Schritte durch:

- Entfernen von URLs
- Normalisierung von Umlauten (ä → ae, ö → oe, ü → ue, ß → ss)
- Filterung auf den Test-Split (`split_12 == 'test'`)
- Balancierung des Datensatzes (gleich viele Hassrede- und Nicht-Hassrede-Beispiele)

## Ausgabe

```
--- Batch Auswertung (N Texte) ---
Precision: 0.XXXX
Recall:    0.XXXX
F1-Score:  0.XXXX
```
