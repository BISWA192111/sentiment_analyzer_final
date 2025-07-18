sentiment-analysis-app-1/
│
├── src/
│   └── models/
│       ├── sentiment5.py                # Main backend logic (Flask app, sentiment analysis, phase detection)
│       └── templates/
│           └── index.html               # Frontend template for user input and result display
│
├── output_cleaned.csv                   # Labeled data for fine-tuning DistilBERT (statements + sentiment labels)
├── common_synonyms_before_during_after_1000.csv  # Wide-format CSV with before/during/after phrases for phase classifier
├── finetune_distilbert.py               # Script to fine-tune DistilBERT on your sentiment data
├── train_phase_classifier.py            # Script to train a phase classifier using your CSV
├── requirements.txt                     # List of Python dependencies
│
├── models/
│   └── distilbert-finetuned-patient/    # Directory containing your fine-tuned DistilBERT model and tokenizer
│       ├── model.safetensors            # Model weights
│       ├── config.json                  # Model config
│       ├── tokenizer.json               # Tokenizer config
│       ├── label_mapping.json           # Mapping of label indices to human-readable labels
│       ├── vocab.txt                    # Vocabulary file
│       ├── special_tokens_map.json      # Special tokens config
│       ├── tokenizer_config.json        # Tokenizer config
│       └── checkpoint-*                 # (Optional) Training checkpoints
│
├── logs/                                # (Optional) Training logs
├── data/                                # (Optional) Additional data files
├── user_responses.csv                   # (Optional) User-submitted responses
├── patient_sentiment_db.csv             # (Optional) Additional data
├── Combined Data.csv                    # (Optional) Additional data
├── output.csv                           # (Optional) Additional data
├── output.zip                           # (Optional) Zipped output
├── WhatsApp Audio ...                   # (Optional) Miscellaneous files
└── ... (other files as needed)
