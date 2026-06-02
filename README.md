# TeamTrustGate 🤖

AI-агент для обработки клиентских запросов: **Telegram → LLM → Jira**

## Что делает
- Принимает запросы от sales через Telegram
- AI анализирует проблему, ROI, охват
- Дедуплицирует с существующими тикетами в Jira
- Задает уточняющие вопросы при недостатке данных
- Создает структурированный тикет в Jira с приоритетом
- Сообщает sales результат

## Архитектура
```
Telegram Bot → AI Engine (Gemini/OpenAI/Claude) → Jira Cloud
                ↓
         SQLite (сессии уточнения)
```

## Быстрый старт

### 1. Переменные окружения
Скопируй `.env.example` в `.env` и заполни:
```bash
cp .env.example .env
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Запуск
```bash
python bot.py
```

## Деплой на Railway
1. Загрузи репозиторий на GitHub
2. Зайди на [railway.app](https://railway.app) → Login with GitHub
3. New Project → Deploy from GitHub repo → выбери этот репо
4. В Variables добавь все переменные из `.env.example`
5. Railway автоматически задеплоит бота

## Команды бота
- `/start` — приветствие и инструкция
- `/cancel` — отменить текущую сессию уточнения
- `/status TT-42` — проверить статус тикета в Jira

## Структура проекта
```
teamtrustgate/
├── bot.py              # Точка входа, Telegram handlers
├── config.py           # Конфигурация из env
├── llm_adapter.py      # Адаптер для Gemini/OpenAI/Anthropic
├── jira_client.py      # Асинхронный клиент Jira API
├── deduplicator.py     # Дедупликация по семантике
├── scorer.py           # RICE-скоринг
├── state_manager.py    # SQLite для сессий
├── prompts/
│   ├── extraction.txt  # Промпт извлечения проблемы
│   ├── dedup.txt       # Промпт сравнения дубликатов
│   └── scoring.txt     # Промпт приоритизации
├── requirements.txt
├── Procfile
└── .env.example
```

## Лицензия
MIT
