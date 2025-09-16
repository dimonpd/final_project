# Инструкции для AI-агента по работе с репозиторием mipt_ds

Кратко: репозиторий учебный, основная работа ведется в Jupyter-ноутбуках; в папке `final_project/` есть минимальный сервис FastAPI для инференса модели из Hugging Face. Следуйте локальным конвенциям и ничего не создавайте/не правьте без явного запроса.

## Архитектура и ключевые пути
- Ноутбуки и исследования: корень репозитория и `final_project/notebooks/` (например, `text_classification_polars.ipynb`, папки с экспериментами `ag_news_distilbert_polars/`, `civil_comments_distilbert/…`).
- Данные: `data/` и `final_project/data/`.
- Прод-сервис (инференс): `final_project/service/app.py` (FastAPI + HF Transformers, DistilBERT для токсичности).
- Контейнеризация: `final_project/Dockerfile` (CPU PyTorch, FastAPI, uvicorn).
- Зависимости проекта: `pyproject.toml` (+ `uv.lock`, менеджер — uv).

## Базовые рабочие процессы
- Среда
	- Используйте uv: установка из `pyproject.toml` — «uv sync», запуск ноутбуков — «uv run jupyter lab». Модель spaCy `en-core-web-sm` уже закреплена как зависимость через URL.
- Эксперименты/обучение
	- Работайте в существующих ноутбуках. Для табличной подготовки данных используйте Polars (в духе `pl.read_csv`/`pl.scan_csv`, `pyarrow` под капотом).
	- Финальные веса сохраняйте в стандартной структуре Hugging Face (`…/best/` рядом с чекпойнтами), как сделано в `final_project/notebooks/civil_comments_distilbert/best/`.
- Локальный инференс API
	- Сервис читает модель из `MODEL_DIR` (по умолчанию `./civil_comments_distilbert/best`) и порог из `THRESHOLD` (по умолчанию `0.5`). Для запуска из корня укажите абсолютный путь к «best» из ноутбуков, например: `MODEL_DIR=final_project/notebooks/civil_comments_distilbert/best`.
	- Эндпоинты: `GET /health` → `{status:"ok"}`; `POST /moderate` → `{toxic: bool, score: float, reasons?: string[]}` с телом `{text: string}`.
- Docker
	- Билд-контекст — `final_project/`. Внимание: `Dockerfile` копирует `civil_comments_distilbert/best/` из корня контекста, а в репозитории «best» лежит в `notebooks/civil_comments_distilbert/best/`. Перед сборкой либо скопируйте папку `best` на ожидаемый путь внутри `final_project/`, либо адаптируйте `Dockerfile` (предпочтительно — но только по запросу владельца).

## Конвенции и ограничения (важно)
- Не создавайте новые файлы и не редактируйте существующие без явной просьбы владельца.
- Пишите просто и безопасно; избегайте побочных эффектов, сетевых вызовов без необходимости и тяжелых принтов (используйте их только по делу).
- Предпочитайте Polars для обработки данных; для NLP — Hugging Face Transformers + Torch; метрики — `scikit-learn`.
- Пути и конфигурация — через переменные окружения и относительные пути от корня репо/модуля; не хардкодьте абсолютные пути.
- Артефакты моделей храните в структуре Hugging Face; лучший чекпойнт именуйте `best/`.

## Интеграции и зависимости
- HF Transformers/Tokenizers + Torch (CPU по умолчанию в Docker).
- Polars (+ PyArrow) для табличных данных.
- spaCy `en-core-web-sm` подтягивается из URL, прописан в `pyproject.toml`.
- В репозитории встречается `catboost_info/` (артефакты ранее обучавшихся моделей) — для сервисной части не требуется.

## Примеры из кода
- Инференс: см. `final_project/service/app.py` — токенизация `AutoTokenizer`, модель `AutoModelForSequenceClassification`, softmax по классу-1, порог `THRESHOLD`.
- Контейнер: `final_project/Dockerfile` — устанавливает Torch CPU, Transformers, FastAPI; запускает `uvicorn service.app:app` на 8000 порту.

Если какое-то из допущений выше не совпадает с вашим процессом (например, как решать несоответствие путей модели для Docker), дайте знать — уточню и обновлю инструкцию. 
