# 🎓 Guida Rapida - BertViz per BERT Italiano

## 📁 File

1. **modeling_bert_italian.py** - Modello BERT personalizzato
   - `BertSelfAttentionIT` - Attenzione modificata per restituire query/key
   - `BertAttentionIT` - Wrapper dell'attenzione
   - `BertLayerIT` - Layer del transformer
   - `BertEncoderIT` - Encoder completo
   - `BertModelIT` - Modello principale

2. **notebooks/neuron_view_bert_italian.ipynb** - Notebook interattivo
   - Setup e import
   - Caricamento modello
   - Esempi di visualizzazione
   - Sezione per sperimentazione

## 🚀 Come Iniziare

### Notebook Jupyter

```bash
# Attiva l'ambiente virtuale
source venv/bin/activate

# Avvia JupyterLab
jupyter lab

# Poi apri il notebook:
# notebooks/neuron_view_bert_italian.ipynb
```

### 1. Esplorare le Attenzioni

Prova diversi valori per `layer` e `head`:

- **Layer bassi (0-3)**: Sintassi e morfologia
- **Layer medi (4-8)**: Grammatica e dipendenze
- **Layer alti (9-11)**: Semantica astratta

### 2. Testare Diversi Testi

```python
# Testi narrativi
"Il protagonista del romanzo attraversa una crisi esistenziale."

# Testi tecnici
"L'algoritmo di apprendimento profondo elabora i dati in parallelo."

# Dialoghi
"Quanto costa questo libro? Vorrei anche quello rosso."

# Testi poetici
"Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura."
```

### 3. Confrontare Modelli

Puoi facilmente cambiare modello modificando `model_name`:

```python
# Modello diverso
model_name = "Musixmatch/umberto-commoncrawl-cased-v1"
```

Modelli BERT italiani disponibili:

- `dbmdz/bert-base-italian-xxl-cased` (Consigliato, ~110M parametri)
- `dbmdz/bert-base-italian-cased`
- `dbmdz/bert-base-italian-uncased`
- `Musixmatch/umberto-commoncrawl-cased-v1`
- `Musixmatch/umberto-wikipedia-uncased-v1`

## 📚 Prossimi Passi

### Approfondimenti

1. **Leggi la documentazione completa**: `README_ITALIAN.md`
2. **Esplora il codice sorgente**: `modeling_bert_italian.py`
3. **Studia gli esempi**: Notebook e script di test

### Risorse Esterne

- [BertViz GitHub](https://github.com/jessevig/bertviz)
- [Paper BERT](https://arxiv.org/abs/1810.04805)
- [Hugging Face Course](https://huggingface.co/course)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

## 🛠️ Troubleshooting Rapido

### Problema: Import Error

**Soluzione**:

```bash
source venv/bin/activate
pip install transformers torch ipywidgets jupyterlab
pip install -e .
```

### Problema: Jupyter non mostra la visualizzazione

**Soluzione**:
```bash
pip install --upgrade ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

---

**Ultimo aggiornamento**: Ottobre 2025  
**Versione**: 1.0  
**Compatibilità**: Python 3.8+, PyTorch 2.0+, Transformers 4.0+
