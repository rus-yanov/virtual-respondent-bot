import os
import json
import logging
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI


# ------------------ базовая настройка ------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert TELEGRAM_TOKEN, "TELEGRAM_TOKEN is missing"
assert OPENAI_API_KEY, "OPENAI_API_KEY is missing"

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("vr-bot")


# ------------------ данные персон ------------------
def load_default_persona() -> Dict:
    with open("persona.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_personas() -> List[Dict]:
    with open("personas_library.json", "r", encoding="utf-8") as f:
        return json.load(f)


DEFAULT_PERSONA = load_default_persona()
PERSONAS = load_personas()


# ------------------ утилиты истории чата ------------------
MAX_HISTORY = 8  # храним последние 8 реплик (user+assistant)


def get_user_state(context: ContextTypes.DEFAULT_TYPE) -> Dict:
    if "state" not in context.user_data:
        context.user_data["state"] = {
            "persona_id": None,
            "persona_title": None,
            "persona_prompt": DEFAULT_PERSONA["prompt"],
            "segment_details": "",
            "awaiting_segment_details": False,
            "history": []
        }
    return context.user_data["state"]


def push_history(state: Dict, role: str, content: str):
    state["history"].append({"role": role, "content": content})
    while len(state["history"]) > MAX_HISTORY:
        state["history"].pop(0)


def history_to_messages(state: Dict) -> List[Dict]:
    persona_prompt = state["persona_prompt"]
    if state.get("segment_details"):
        persona_prompt += f"\nКонтекст уточнения аудитории: {state['segment_details']}"
    msgs = [{"role": "system", "content": persona_prompt}]
    msgs.extend(state["history"])
    return msgs


def log_chat(user_id: int, user_text: str, bot_text: str):
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"{user_id}.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] USER: {user_text}\n")
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] BOT : {bot_text}\n\n")


# ------------------ OpenAI вызовы ------------------
async def call_llm(messages: List[Dict], max_tokens: int = 200) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()


# ------------------ handlers ------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = get_user_state(context)
    state.update({
        "persona_id": None,
        "persona_title": None,
        "persona_prompt": DEFAULT_PERSONA["prompt"],
        "segment_details": "",
        "awaiting_segment_details": False,
        "history": []
    })

    # reply-кнопка для быстрого рестарта
    reply_keyboard = [[KeyboardButton("🔄 Начать заново")]]

    inline_keyboard = [
        [InlineKeyboardButton("🧑‍🎤 Выбрать персону", callback_data="pick_persona")],
        [InlineKeyboardButton("💬 Начать чат", callback_data="begin_chat")],
        [InlineKeyboardButton("📄 Сводка (/summary)", callback_data="summary_hint")],
    ]

    await update.message.reply_text(
        "Привет! Я — «Виртуальный респондент».\n"
        "• Выбери персону (или оставь по умолчанию)\n"
        "• Напиши любой вопрос — отвечу от первого лица\n"
        "• В конце используй /summary для итогов",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
    )

    await update.message.reply_text(
        "Выбери действие:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard),
    )


def get_persona_question(persona_id: str) -> str:
    questions = {
        "young_mom_moscow": (
            "Чтобы ответы респондента звучали реалистично и соответствовали задачам исследования, "
            "укажи ключевой критерий сегмента, который важен именно для твоего исследования.\n\n"
            "✍️ Напиши коротко: какой именно подтип аудитории тебе нужен и чем он важен для теста.\n\n"
            "Примеры признаков:\n• возраст ребёнка\n• семейное положение\n• занятость\n• тип жилья\n• интересы\n• уровень дохода семьи"
        ),
        "it_engineer": (
            "Чтобы ответы респондента звучали реалистично и соответствовали задачам исследования, "
            "укажи ключевой критерий сегмента, который важен именно для твоего исследования.\n\n"
            "✍️ Напиши коротко: какой именно подтип аудитории тебе нужен и чем он важен для теста.\n\n"
            "Примеры признаков:\n• специализация\n• уровень (junior, middle, senior)\n• формат работы\n• страна\n• тип компании\n• приоритеты"
        ),
        "smb_owner": (
            "Чтобы ответы респондента звучали реалистично и соответствовали задачам исследования, "
            "укажи ключевой критерий сегмента, который важен именно для твоего исследования.\n\n"
            "✍️ Напиши коротко: какой именно подтип аудитории тебе нужен и чем он важен для теста.\n\n"
            "Примеры признаков:\n• отрасль\n• размер бизнеса и команды\n• стаж предпринимателя\n• регион\n• модель бизнеса"
        ),
    }
    return questions.get(persona_id, "Опиши, пожалуйста, уточнения по сегменту целевой аудитории.")


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    state = get_user_state(context)

    if query.data == "pick_persona":
        rows = []
        for p in PERSONAS:
            rows.append([InlineKeyboardButton(p["title"], callback_data=f"persona:{p['id']}")])
        rows.append([InlineKeyboardButton("Назад", callback_data="back_home")])
        await query.edit_message_text("Выбери персону:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if query.data.startswith("persona:"):
        persona_id = query.data.split(":", 1)[1]
        persona = next((p for p in PERSONAS if p["id"] == persona_id), None)
        if persona:
            state["persona_id"] = persona_id
            state["persona_title"] = persona["title"]
            state["persona_prompt"] = persona["prompt"]
            state["history"].clear()
            state["segment_details"] = ""
            state["awaiting_segment_details"] = True

            question_text = get_persona_question(persona_id)
            await query.edit_message_text(
                f"Персона установлена: *{persona['title']}*.\n\n{question_text}",
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text("Не нашёл такую персону. Попробуй ещё раз.")
        return

    if query.data == "begin_chat":
        await query.edit_message_text("Ок, пиши вопрос. Я отвечу от лица выбранной персоны.")
        return

    if query.data == "summary_hint":
        await query.edit_message_text("В конце сеанса отправь команду /summary — соберу выводы и инсайты.")
        return

    if query.data == "back_home":
        await query.edit_message_text("Готово. Можешь начать чат или выбрать персону.")
        return


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — начать заново\n"
        "/summary — краткий отчёт по текущему диалогу\n"
        "Также можно в любой момент писать вопросы."
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    state = get_user_state(context)

    # обработка кнопки «Начать заново»
    if user_text.lower() in ["начать заново", "🔄 начать заново"]:
        await start(update, context)
        return

    # если бот ждёт уточнение после выбора персоны
    if state.get("awaiting_segment_details"):
        state["awaiting_segment_details"] = False
        state["segment_details"] = user_text
        await update.message.reply_text(
            "Отлично, контекст зафиксирован. Можешь начать задавать вопросы пользователю."
        )
        return

    # обычная логика чата
    push_history(state, "user", user_text)
    try:
        answer = await call_llm(history_to_messages(state))
    except Exception:
        logger.exception("LLM error")
        answer = "Хм, у меня сейчас сложности с ответом. Попробуй ещё раз через минуту."
    push_history(state, "assistant", answer)
    log_chat(update.effective_user.id, user_text, answer)
    await update.message.reply_text(answer)


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = get_user_state(context)
    if not state["history"]:
        await update.message.reply_text("Пока нечего суммировать — напиши пару вопросов.")
        return

    persona_title = state.get("persona_title", "Без персоны")
    segment_details = state.get("segment_details", "")
    summary_intro = f"Отчёт по персоне: *{persona_title}*"
    if segment_details:
        summary_intro += f"\nУточнение сегмента: {segment_details}"

    msgs = [
        {"role": "system", "content": state["persona_prompt"]},
        *state["history"],
        {"role": "user", "content":
            "Сделай краткий отчёт по нашей беседе: 3–5 пунктов инсайтов, что понравилось/не понравилось, "
            "ожидания/триггеры, и 3 тестовых next steps для продукта. Формат — маркированный список."}
    ]
    try:
        report = await call_llm(msgs, max_tokens=350)
    except Exception:
        logger.exception("LLM summary error")
        report = "Не удалось собрать сводку. Попробуй ещё раз позже."

    log_chat(update.effective_user.id, "[/summary]", report)
    await update.message.reply_text(f"{summary_intro}\n\n📄 Итоги:\n{report}", parse_mode="Markdown")


# ------------------ запуск ------------------
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Bot is running (polling).")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()