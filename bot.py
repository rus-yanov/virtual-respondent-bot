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
            "persona_prompt": DEFAULT_PERSONA["prompt"],
            "history": []  # list[{"role": "user"/"assistant", "content": "..."}]
        }
    return context.user_data["state"]

def push_history(state: Dict, role: str, content: str):
    state["history"].append({"role": role, "content": content})
    # ограничиваем длину
    while len(state["history"]) > MAX_HISTORY:
        state["history"].pop(0)

def history_to_messages(state: Dict) -> List[Dict]:
    msgs = [{"role": "system", "content": state["persona_prompt"]}]
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
    """
    Минимальный вызов Chat Completions для gpt-4o-mini.
    """
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
    state["persona_id"] = None
    state["persona_prompt"] = DEFAULT_PERSONA["prompt"]
    state["history"].clear()

    keyboard = [
        [InlineKeyboardButton("🧑‍🎤 Выбрать персону", callback_data="pick_persona")],
        [InlineKeyboardButton("💬 Начать чат", callback_data="begin_chat")],
        [InlineKeyboardButton("📄 Сводка (/summary)", callback_data="summary_hint")],
    ]
    await update.message.reply_text(
        "Привет! Я — «Виртуальный респондент».\n"
        "• Выбери персону (или оставь по умолчанию)\n"
        "• Напиши любой вопрос — отвечу от первого лица\n"
        "• В конце используй /summary для итогов",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "pick_persona":
        # показать список персон
        rows = []
        for p in PERSONAS:
            rows.append([InlineKeyboardButton(p["title"], callback_data=f"persona:{p['id']}")])
        rows.append([InlineKeyboardButton("Назад", callback_data="back_home")])
        await query.edit_message_text("Выбери персону:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if query.data.startswith("persona:"):
        persona_id = query.data.split(":", 1)[1]
        persona = next((p for p in PERSONAS if p["id"] == persona_id), None)
        state = get_user_state(context)
        if persona:
            state["persona_id"] = persona_id
            state["persona_prompt"] = persona["prompt"]
            state["history"].clear()
            await query.edit_message_text(f"Персона установлена: *{persona['title']}*.\n"
                                          f"Напиши вопрос.", parse_mode="Markdown")
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
    push_history(state, "user", user_text)

    try:
        answer = await call_llm(history_to_messages(state))
    except Exception as e:
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

    # подсказка для сводки
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

    # лог
    log_chat(update.effective_user.id, "[/summary]", report)
    await update.message.reply_text("📄 Итоги:\n" + report)

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