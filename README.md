# Tennis Prediction Project

Questa applicazione predice i risultati delle partite di tennis utilizzando un modello di Machine Learning (XGBoost).

## Descrizione

Il progetto include script per scaricare dati, allenare un modello e una web app Streamlit per visualizzare le predizioni.

## Installazione

1.  Clona il repository.
2.  Crea un virtual environment (opzionale ma consigliato):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Mac/Linux
    ```
3.  Installa le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```

## Utilizzo

Per avviare l'applicazione web:

```bash
streamlit run tennis_app.py
```

## Struttura del Progetto

-   `tennis_app.py`: Applicazione principale Streamlit.
-   `tennis_bot.py`: Logica di predizione e bot.
-   `scrape_tennis.py`: Script per scaricare i dati dei match.
-   `evaluate_model.py`: Script per valutare l'accuratezza del modello.
