# Модерация токсичных комментариев (HF Datasets + Polars + Transformers)

Учебный проект: бинарная классификация токсичности комментариев на датасете Civil Comments с использованием Polars для подготовки данных и Hugging Face Transformers для модели (DistilBERT/альтернативы). В репозитории есть ноутбуки для экспериментов, минимальный FastAPI-сервис для инференса и Streamlit-приложение для интерактивной проверки.

## Структура репозитория

- `notebooks/` — исследования и обучение
  - `text_classification_polars.ipynb` — основной ноутбук с пайплайном
  - `civil_comments_distilbert/best/` — сохранённая лучшая модель HF-формата (пример)
- `data/` — локальные parquet-данные (train/val/test) для Civil Comments

- `streamlit_app.py` — Streamlit UI
- `requirements.txt` — зафиксированные версии пакетов текущего окружения

## Среда разработки (uv + .venv)

Проект использует uv и локальное окружение `.venv`.

1. Создать окружение (если ещё не создано):

```bash
uv venv --seed .venv
```

2. Активировать:

```bash
source .venv/bin/activate
```

3. Установить зависимости:

```bash
uv pip install -r requirements.txt
```

4. (Опционально) Запуск ноутбуков:

```bash
uv run jupyter lab
```

## Данные

- Используется датасет: https://huggingface.co/datasets/google/civil_comments
- В ноутбуке показаны варианты загрузки через HF Hub и локальные parquet-файлы.

## Обучение модели (ноутбук)

Откройте `notebooks/text_classification_polars.ipynb` и выполните ячейки по порядку:

- Подготовка (Polars), бинаризация метки `toxicity >= 0.5` → `label`{0,1}
- Токенизация (AutoTokenizer), датасеты, Trainer
- Сравнение чекпоинтов (distilbert/bert/roberta) — быстрый прогон
- Финальное обучение, сохранение модели в `notebooks/civil_comments_distilbert/best/`

Переменные окружения:

- `MODEL_DIR` — путь к модели HF (по умолчанию: `./civil_comments_distilbert/best` внутри контекста сервиса)
- `THRESHOLD` — порог для класса «токсичный» (по умолчанию: `0.5`)

## Streamlit-приложение

Интерактивное приложение для ручной проверки одного комментария.

Запуск:

```bash
source .venv/bin/activate
export MODEL_DIR="notebooks/civil_comments_distilbert/best"
export THRESHOLD=0.5
streamlit run streamlit_app.py
```

## Примечания

- Если целевая машина без CUDA, перед установкой зависимостей на ней можно удалить CUDA-пакеты (`nvidia-*-cu12`) из `requirements.txt`.
