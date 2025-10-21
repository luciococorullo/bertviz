# BertViz per Modelli BERT Italiani

Questo progetto estende [BertViz](https://github.com/jessevig/bertviz) per supportare la visualizzazione della **Neuron View** con modelli BERT in lingua italiana.

## 📋 Descrizione

BertViz è uno strumento potente per visualizzare i meccanismi di attenzione nei modelli Transformer. Questo progetto fornisce un'implementazione personalizzata che permette di utilizzare la Neuron View con modelli BERT italiani pre-addestrati da Hugging Face.

### Cosa è la Neuron View?

La Neuron View mostra come i **query vectors** e i **key vectors** interagiscono per produrre i punteggi di attenzione in un modello Transformer. Questa visualizzazione aiuta a comprendere:

- Come il modello pesa le diverse parole in una frase
- Quali relazioni sintattiche e semantiche vengono catturate
- Come diversi layer e head di attenzione si specializzano

## 🚀 Setup dell'Ambiente

### Prerequisiti

- Python 3.8 o superiore
- pip (gestore pacchetti Python)

### Installazione

1. **Clona il repository (o scarica i file)**

```bash
git clone <repository-url>
cd bertviz
```

2. **Crea un ambiente virtuale**

```bash
python3 -m venv venv
source venv/bin/activate  # Su macOS/Linux
# oppure
venv\Scripts\activate  # Su Windows
```

3. **Installa le dipendenze**

```bash
pip install --upgrade pip
pip install transformers torch ipywidgets jupyterlab
pip install -e .
```

## 📁 Struttura del Progetto

```
bertviz/
├── modeling_bert_italian.py          # Modello BERT italiano personalizzato
├── notebooks/
│   └── neuron_view_bert_italian.ipynb  # Notebook di esempio
├── bertviz/                          # Libreria BertViz originale
│   ├── neuron_view.py
│   ├── head_view.py
│   ├── model_view.py
│   └── transformers_neuron_view/    # Implementazioni personalizzate
│       └── modeling_bert.py
└── README_ITALIAN.md                 # Questo file
```

## 🎯 Utilizzo

### Metodo 1: Notebook Jupyter (Consigliato)

1. Avvia JupyterLab:

```bash
source venv/bin/activate
jupyter lab
```

2. Apri il notebook `notebooks/neuron_view_bert_italian.ipynb`

3. Esegui le celle in sequenza

### Metodo 3: Google Colab

Se preferisci usare Google Colab, puoi caricare i file e installare le dipendenze direttamente in una cella:

```python
!pip install transformers torch ipywidgets bertviz

# Carica il file modeling_bert_italian.py
# (puoi caricarlo manualmente o da Google Drive)
```

## 🔧 Come Funziona

### Architettura

Il progetto si basa su una **strategia di subclassing** pulita e manutenibile:

```
BertModelIT (modello principale)
    └── BertEncoderIT
        └── BertLayerIT (x12 layers)
            └── BertAttentionIT
                └── BertSelfAttentionIT (⭐ modifica chiave)
```

### Modifica Chiave

La modifica principale si trova nel metodo `forward` di `BertSelfAttentionIT`:

**Versione Originale:**
```python
return (context_layer, attention_probs)
```

**Versione Personalizzata:**
```python
return (context_layer, attention_probs, query_layer, key_layer)
```

Questa semplice modifica permette alla Neuron View di accedere ai tensori query e key necessari per la visualizzazione.

## 📊 Esempi di Visualizzazione

### Esempio 1: Relazioni Semplici

```python
sentence_a = "Il gatto si è seduto sul tappeto."
sentence_b = "Il cane dormiva sulla poltrona."
show(model, "bert", tokenizer, sentence_a, sentence_b, layer=2, head=8)
```

### Esempio 2: Relazioni Complesse

```python
sentence_a = "La studentessa che studia matematica ha superato l'esame."
sentence_b = "Il professore le ha fatto i complimenti."
show(model, "bert", tokenizer, sentence_a, sentence_b, layer=5, head=3)
```

### Esempio 3: Testo Tecnico

```python
sentence_a = "L'intelligenza artificiale trasforma il modo in cui lavoriamo."
sentence_b = "Gli algoritmi di machine learning apprendono dai dati."
show(model, "bert", tokenizer, sentence_a, sentence_b, layer=8, head=5)
```

## 🎛️ Parametri

### show()

- `model`: Il modello BERT personalizzato (`BertModelIT`)
- `model_type`: Sempre `"bert"` per questo caso d'uso
- `tokenizer`: Il tokenizzatore di Hugging Face
- `sentence_a`: Prima frase (obbligatoria)
- `sentence_b`: Seconda frase (opzionale, per task di sentence pair)
- `layer`: Numero del layer da visualizzare (0-11 per BERT base)
- `head`: Numero della testa di attenzione (0-11 per BERT base)

### Interpretazione dei Layer

- **Layer 0-3** (Bassi): Tendono a catturare caratteristiche sintattiche e morfologiche
- **Layer 4-8** (Medi): Catturano relazioni grammaticali e dipendenze sintattiche
- **Layer 9-11** (Alti): Si concentrano su semantica astratta e inferenza

## 🔬 Modelli BERT Italiani Supportati

Il codice è compatibile con qualsiasi modello BERT italiano di Hugging Face, inclusi:

- `dbmdz/bert-base-italian-xxl-cased` (Consigliato)
- `dbmdz/bert-base-italian-cased`
- `dbmdz/bert-base-italian-uncased`
- `Musixmatch/umberto-commoncrawl-cased-v1`
- `Musixmatch/umberto-wikipedia-uncased-v1`

Per cambiare modello, basta modificare la variabile `model_name`:

```python
model_name = "Musixmatch/umberto-commoncrawl-cased-v1"
```

## 🐛 Risoluzione Problemi

### Problema: ModuleNotFoundError

**Soluzione**: Assicurati di aver attivato l'ambiente virtuale:
```bash
source venv/bin/activate
```

### Problema: Il notebook non mostra la visualizzazione

**Soluzione**: Assicurati che `ipywidgets` sia installato e abilitato:
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Problema: Out of Memory

**Soluzione**: Se hai problemi di memoria, puoi ridurre la lunghezza delle frasi o usare un batch size più piccolo.

## 📚 Risorse

### Documentazione

- [BertViz GitHub](https://github.com/jessevig/bertviz)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

### Modelli

- [dbmdz BERT Italian](https://huggingface.co/dbmdz/bert-base-italian-xxl-cased)
- [Umberto Models](https://huggingface.co/Musixmatch)

### Tutorial

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Visualizing Attention in Transformers](https://towardsdatascience.com/visualizing-attention-in-transformers-1d031f6b69ec)

## 🙏 Crediti

- **BertViz**: Jesse Vig
- **Transformers**: Hugging Face
- **BERT Italiano**: dbmdz (Bayerische Staatsbibliothek)

---
