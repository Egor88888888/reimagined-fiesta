"""
DocLens Telegram Bot
Принимает фото документов, распознаёт через Doc Lens API,
показывает результат, сохраняет в SQLite, отправляет по webhook.

Использование:
  1. Создать бота у @BotFather, получить токен
  2. Получить API-ключ Doc Lens (https://egor8888888-ocr.hf.space/demo)
  3. Заполнить .env файл
  4. pip install aiogram httpx aiosqlite
  5. python bot.py
"""
import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
import aiosqlite
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, ContentType
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import WebAppInfo, MenuButtonWebApp, InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DOCLENS_API_URL = os.getenv("DOCLENS_API_URL", "https://egor8888888-ocr.hf.space")
DOCLENS_API_KEY = os.getenv("DOCLENS_API_KEY", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # Optional: URL to forward data
DB_PATH = os.getenv("DB_PATH", "doclens_bot.db")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("doclens_bot")

# Bot & Dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
router = Router()

# DocLens API client
api_client = httpx.AsyncClient(
    base_url=DOCLENS_API_URL,
    headers={"X-API-Key": DOCLENS_API_KEY},
    timeout=60.0,
)


# ============================================================
# Database
# ============================================================

async def init_db():
    """Create tables if not exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_user_id INTEGER NOT NULL,
                telegram_username TEXT,
                document_type TEXT,
                overall_confidence REAL DEFAULT 0,
                fields_json TEXT,
                validation_json TEXT,
                warnings_json TEXT,
                recognition_id TEXT,
                processing_time_ms INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS document_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT,
                confidence REAL DEFAULT 0,
                ml_corrected INTEGER DEFAULT 0,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        await db.commit()
    logger.info("Database initialized")


async def save_recognition(user_id: int, username: str, result: dict) -> int:
    """Save recognition result to SQLite. Returns document ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Safely serialize all values for SQLite
        doc_type = str(result.get("document_type", "unknown"))
        confidence = float(result.get("overall_confidence", 0) or 0)
        fields_j = json.dumps(result.get("fields") or {}, ensure_ascii=False, default=str)
        validation_j = json.dumps(result.get("validation") or {}, ensure_ascii=False, default=str)
        warnings_j = json.dumps(result.get("warnings") or [], ensure_ascii=False, default=str)
        rec_id = str(result.get("id", ""))
        proc_time = int(result.get("processing_time_ms") or 0)

        cursor = await db.execute(
            """INSERT INTO documents
               (telegram_user_id, telegram_username, document_type,
                overall_confidence, fields_json, validation_json,
                warnings_json, recognition_id, processing_time_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, username, doc_type, confidence, fields_j,
             validation_j, warnings_j, rec_id, proc_time),
        )
        doc_id = cursor.lastrowid

        # Save individual fields
        for field_name, fdata in (result.get("fields") or {}).items():
            if isinstance(fdata, dict):
                val = fdata.get("value", "")
                # Convert non-string values to string
                if not isinstance(val, str):
                    val = json.dumps(val, ensure_ascii=False, default=str)
                await db.execute(
                    """INSERT INTO document_fields
                       (document_id, field_name, field_value, confidence, ml_corrected)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        doc_id,
                        str(field_name),
                        val,
                        float(fdata.get("confidence") or 0),
                        1 if fdata.get("ml_corrected") else 0,
                    ),
                )

        await db.commit()
    return doc_id


# ============================================================
# DocLens API Integration
# ============================================================

async def recognize_document(photo_bytes: bytes, filename: str = "photo.jpg") -> dict:
    """Send image to DocLens API and get recognition result."""
    try:
        response = await api_client.post(
            "/api/v1/recognize",
            files={"file": (filename, photo_bytes, "image/jpeg")},
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"DocLens API error {e.response.status_code}: {e.response.text}")
        return {"error": f"API ошибка: {e.response.status_code}", "fields": {}}
    except Exception as e:
        logger.error(f"DocLens API connection error: {e}")
        return {"error": f"Ошибка подключения: {e}", "fields": {}}


async def send_correction(recognition_id: str, document_type: str,
                          field_name: str, original: str, corrected: str):
    """Send field correction to DocLens ML learning endpoint."""
    try:
        response = await api_client.post(
            "/api/v1/ml/corrections",
            json={
                "recognition_id": recognition_id,
                "document_type": document_type,
                "corrections": [
                    {
                        "field_name": field_name,
                        "original_value": original,
                        "corrected_value": corrected,
                    }
                ],
            },
        )
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send correction: {e}")
        return None


async def send_to_webhook(data: dict):
    """Forward recognition data to external system (CRM/1C/etc)."""
    if not WEBHOOK_URL:
        return

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(WEBHOOK_URL, json=data)
            logger.info(f"Webhook sent: {response.status_code}")
    except Exception as e:
        logger.error(f"Webhook failed: {e}")


# ============================================================
# Formatting
# ============================================================

# Human-readable field names
FIELD_LABELS = {
    "last_name": "Фамилия",
    "first_name": "Имя",
    "patronymic": "Отчество",
    "birth_date": "Дата рождения",
    "birth_place": "Место рождения",
    "sex": "Пол",
    "series": "Серия",
    "number": "Номер",
    "issue_date": "Дата выдачи",
    "issuer": "Кем выдан",
    "department_code": "Код подразделения",
}

DOC_TYPE_LABELS = {
    "passport_rf": "Паспорт РФ",
    "passport_cis": "Паспорт СНГ",
    "driver_license": "Водительское удостоверение",
    "snils": "СНИЛС",
    "inn": "ИНН",
    "bank_statement": "Банковская выписка",
}


def format_result(result: dict) -> str:
    """Format recognition result as a Telegram message (max 4096 chars)."""
    if "error" in result:
        return f"❌ {result['error']}"

    doc_type = result.get("document_type", "unknown")
    doc_label = DOC_TYPE_LABELS.get(doc_type, doc_type)
    confidence = result.get("overall_confidence", 0)
    fields = result.get("fields", {})
    time_ms = result.get("processing_time_ms", 0)

    lines = [
        f"📄 <b>{doc_label}</b>",
        f"🎯 Точность: {confidence:.0%} | ⏱ {time_ms}мс",
        "",
    ]

    if not fields:
        lines.append("⚠️ Поля не распознаны")
        return "\n".join(lines)

    # Only show known important fields, skip raw/debug data
    important_fields = [
        "last_name", "first_name", "patronymic", "series", "number",
        "birth_date", "birth_place", "sex", "issue_date", "issuer",
        "department_code",
    ]

    shown = 0
    for field_name in important_fields:
        if field_name not in fields:
            continue
        fdata = fields[field_name]
        if not isinstance(fdata, dict):
            continue

        label = FIELD_LABELS.get(field_name, field_name)
        value = str(fdata.get("value", "—"))[:100]  # Truncate long values
        conf = fdata.get("confidence", 0)
        ml = " 🤖" if fdata.get("ml_corrected") else ""

        if conf >= 0.95:
            icon = "🟢"
        elif conf >= 0.7:
            icon = "🟡"
        else:
            icon = "🔴"

        lines.append(f"{icon} <b>{label}:</b> {value}{ml}")
        shown += 1

    # Show remaining fields not in important list (up to 5 more)
    for field_name, fdata in fields.items():
        if field_name in important_fields or not isinstance(fdata, dict):
            continue
        if shown >= 15:
            lines.append(f"... и ещё {len(fields) - shown} полей")
            break
        label = FIELD_LABELS.get(field_name, field_name)
        value = str(fdata.get("value", "—"))[:80]
        lines.append(f"  <b>{label}:</b> {value}")
        shown += 1

    # Warnings (max 2)
    warnings = result.get("warnings", [])
    if warnings:
        lines.append("")
        for w in warnings[:2]:
            w_short = str(w)[:150]
            lines.append(f"⚠️ {w_short}")

    text = "\n".join(lines)
    # Telegram limit is 4096 chars
    if len(text) > 4000:
        text = text[:3990] + "\n..."
    return text


# ============================================================
# Bot Handlers
# ============================================================

def get_webapp_url() -> str:
    """Build MiniApp URL with API credentials."""
    from urllib.parse import urlencode
    params = urlencode({"api": DOCLENS_API_URL, "key": DOCLENS_API_KEY})
    return f"{DOCLENS_API_URL}/webapp?{params}"


@router.message(Command("start"))
async def cmd_start(message: Message):
    webapp_url = get_webapp_url()
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="📱 Открыть DocLens",
            web_app=WebAppInfo(url=webapp_url),
        )],
    ])
    await message.answer(
        "👋 Привет! Я бот для распознавания документов.\n\n"
        "📸 Отправьте фото документа или откройте MiniApp:\n\n"
        "Команды:\n"
        "/app — открыть MiniApp\n"
        "/history — последние распознавания\n"
        "/stats — статистика",
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
    )


@router.message(Command("app"))
async def cmd_app(message: Message):
    webapp_url = get_webapp_url()
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="📱 Открыть DocLens",
            web_app=WebAppInfo(url=webapp_url),
        )],
    ])
    await message.answer("Нажмите кнопку ниже:", reply_markup=kb)


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "📋 <b>Как пользоваться:</b>\n\n"
        "1. Отправьте фото документа (или файл)\n"
        "2. Бот распознает тип и извлечёт данные\n"
        "3. Результат сохраняется в базу\n\n"
        "🤖 <b>ML обучение:</b>\n"
        "Если данные распознались неверно, нажмите «Исправить» "
        "и отправьте правильное значение. Система учится на ваших исправлениях.\n\n"
        "🟢 — высокая точность (95%+)\n"
        "🟡 — средняя точность (70-95%)\n"
        "🔴 — низкая точность (<70%)\n"
        "🤖 — поле исправлено ML",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("history"))
async def cmd_history(message: Message):
    """Show last 5 recognitions for this user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT document_type, overall_confidence, created_at, fields_json
               FROM documents
               WHERE telegram_user_id = ?
               ORDER BY created_at DESC LIMIT 5""",
            (message.from_user.id,),
        )
        rows = await cursor.fetchall()

    if not rows:
        await message.answer("📭 Пока нет распознанных документов.")
        return

    lines = ["📜 <b>Последние распознавания:</b>\n"]
    for row in rows:
        doc_label = DOC_TYPE_LABELS.get(row["document_type"], row["document_type"])
        conf = row["overall_confidence"] or 0
        fields = json.loads(row["fields_json"] or "{}")
        name = fields.get("last_name", {}).get("value", "—")
        lines.append(f"• {doc_label} | {name} | {conf:.0%} | {row['created_at'][:16]}")

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


@router.message(Command("stats"))
async def cmd_stats(message: Message):
    """Show user stats."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """SELECT COUNT(*) as total,
                      AVG(overall_confidence) as avg_conf,
                      AVG(processing_time_ms) as avg_time
               FROM documents WHERE telegram_user_id = ?""",
            (message.from_user.id,),
        )
        row = await cursor.fetchone()

    total = row[0] or 0
    avg_conf = row[1] or 0
    avg_time = row[2] or 0

    await message.answer(
        f"📊 <b>Ваша статистика:</b>\n\n"
        f"Документов распознано: {total}\n"
        f"Средняя точность: {avg_conf:.0%}\n"
        f"Среднее время: {avg_time:.0f}мс",
        parse_mode=ParseMode.HTML,
    )


@router.message(F.photo | F.document)
async def handle_document_photo(message: Message):
    """Handle photo or document upload — recognize via DocLens."""
    # Show "processing" indicator
    processing_msg = await message.answer("⏳ Распознаю документ...")

    try:
        # Download file
        if message.photo:
            # Get highest resolution photo
            photo = message.photo[-1]
            file = await bot.get_file(photo.file_id)
            file_bytes = await bot.download_file(file.file_path)
            filename = "photo.jpg"
        elif message.document:
            file = await bot.get_file(message.document.file_id)
            file_bytes = await bot.download_file(file.file_path)
            filename = message.document.file_name or "document"
        else:
            await processing_msg.edit_text("❌ Отправьте фото или файл документа.")
            return

        # Read bytes
        photo_bytes = file_bytes.read() if hasattr(file_bytes, "read") else file_bytes

        # Recognize via DocLens API
        result = recognize_document(photo_bytes, filename)
        if asyncio.iscoroutine(result):
            result = await result

        # Format and send result
        text = format_result(result)

        # Save to database first (need doc_id for buttons)
        doc_id = 0
        recognition_id = result.get("id", "")
        doc_type = result.get("document_type", "unknown")

        if "error" not in result:
            doc_id = await save_recognition(
                user_id=message.from_user.id,
                username=message.from_user.username or "",
                result=result,
            )
            logger.info(f"Saved recognition #{doc_id} for user {message.from_user.id}")

        # Build inline keyboard (use short doc_id, not long UUID)
        kb = InlineKeyboardBuilder()
        if result.get("fields") and "error" not in result and doc_id:
            # Short callback_data: "c:123:passport_rf" (well under 64 bytes)
            short_type = doc_type[:15]  # Truncate type to be safe
            kb.button(
                text="✏️ Исправить поле",
                callback_data=f"c:{doc_id}:{short_type}",
            )
            kb.button(
                text="✅ Всё верно",
                callback_data=f"ok:{doc_id}",
            )
            kb.adjust(2)

        await processing_msg.edit_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=kb.as_markup() if result.get("fields") and doc_id else None,
        )

        # Send to webhook (CRM/1C)
        if "error" not in result:
            webhook_data = {
                "source": "telegram_bot",
                "telegram_user_id": message.from_user.id,
                "telegram_username": message.from_user.username,
                "document_type": doc_type,
                "fields": result.get("fields", {}),
                "confidence": result.get("overall_confidence", 0),
                "recognition_id": recognition_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await send_to_webhook(webhook_data)

    except Exception as e:
        logger.exception(f"Recognition error: {e}")
        await processing_msg.edit_text(f"❌ Ошибка: {e}")


# ============================================================
# Correction flow (ML feedback)
# ============================================================

# Temporary storage for correction state
_correction_state = {}  # user_id -> {recognition_id, doc_type, field_name, original}


@router.callback_query(F.data.startswith("c:"))
async def on_correct_start(callback: CallbackQuery):
    """Show field selection for correction."""
    parts = callback.data.split(":", 2)
    doc_id = parts[1]
    doc_type = parts[2] if len(parts) > 2 else ""

    # Get fields from DB
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT field_name, field_value FROM document_fields
               WHERE document_id = ?""",
            (doc_id,),
        )
        fields = await cursor.fetchall()

        # Also get recognition_id for ML feedback
        cursor2 = await db.execute(
            "SELECT recognition_id FROM documents WHERE id = ?", (doc_id,)
        )
        row = await cursor2.fetchone()
        rec_id = row["recognition_id"] if row else ""

    if not fields:
        await callback.answer("Поля не найдены", show_alert=True)
        return

    kb = InlineKeyboardBuilder()
    for field in fields:
        fname = field["field_name"]
        label = FIELD_LABELS.get(fname, fname)
        short_val = (field["field_value"] or "—")[:15]
        # Keep callback_data under 64 bytes: "f:123:last_name"
        kb.button(
            text=f"{label}: {short_val}",
            callback_data=f"f:{doc_id}:{fname[:20]}",
        )
    kb.button(text="❌ Отмена", callback_data="x")
    kb.adjust(1)

    # Store doc metadata for correction flow
    _correction_state[f"doc:{callback.from_user.id}"] = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "recognition_id": rec_id,
    }

    await callback.message.edit_reply_markup(reply_markup=kb.as_markup())
    await callback.answer()


@router.callback_query(F.data.startswith("f:"))
async def on_fix_field(callback: CallbackQuery):
    """User selected a field to correct — ask for correct value."""
    parts = callback.data.split(":", 2)
    doc_id = parts[1]
    field_name = parts[2] if len(parts) > 2 else ""

    # Get original value
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """SELECT field_value FROM document_fields
               WHERE document_id = ? AND field_name = ?""",
            (doc_id, field_name),
        )
        row = await cursor.fetchone()

    original = row[0] if row else ""
    label = FIELD_LABELS.get(field_name, field_name)

    # Get doc metadata
    doc_meta = _correction_state.get(f"doc:{callback.from_user.id}", {})

    # Save state for text input handler
    _correction_state[callback.from_user.id] = {
        "recognition_id": doc_meta.get("recognition_id", ""),
        "doc_type": doc_meta.get("doc_type", ""),
        "doc_id": doc_id,
        "field_name": field_name,
        "original": original,
    }

    await callback.message.answer(
        f"✏️ Исправление поля <b>{label}</b>\n"
        f"Текущее значение: <code>{original}</code>\n\n"
        f"Отправьте правильное значение:",
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("ok:"))
async def on_confirm(callback: CallbackQuery):
    """User confirmed all fields are correct."""
    await callback.answer("✅ Спасибо! Данные подтверждены.", show_alert=True)
    await callback.message.edit_reply_markup(reply_markup=None)


@router.callback_query(F.data == "x")
async def on_cancel_correct(callback: CallbackQuery):
    """Cancel correction."""
    _correction_state.pop(callback.from_user.id, None)
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer("Отменено")


@router.message(F.text)
async def handle_text(message: Message):
    """Handle text messages — check if it's a correction response."""
    state = _correction_state.pop(message.from_user.id, None)
    if not state:
        # Not a correction, show hint
        await message.answer(
            "📸 Отправьте фото документа для распознавания.\n"
            "Или используйте /help для справки."
        )
        return

    corrected_value = message.text.strip()
    label = FIELD_LABELS.get(state["field_name"], state["field_name"])

    # Send correction to DocLens ML
    ml_result = await send_correction(
        recognition_id=state["recognition_id"],
        document_type=state["doc_type"],
        field_name=state["field_name"],
        original=state["original"],
        corrected=corrected_value,
    )

    if ml_result and ml_result.get("status") == "ok":
        await message.answer(
            f"✅ <b>{label}</b> исправлено:\n"
            f"<s>{state['original']}</s> → <b>{corrected_value}</b>\n\n"
            f"🤖 Система запомнит это исправление для обучения.",
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.answer(
            f"✅ <b>{label}</b> исправлено на: <b>{corrected_value}</b>\n"
            f"⚠️ ML обучение недоступно (данные сохранены локально).",
            parse_mode=ParseMode.HTML,
        )

    # Update local DB too
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE document_fields SET field_value = ?
               WHERE field_name = ? AND document_id = (
                   SELECT id FROM documents WHERE recognition_id = ?
               )""",
            (corrected_value, state["field_name"], state["recognition_id"]),
        )
        await db.commit()


# ============================================================
# Main
# ============================================================

async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set! Create .env file.")
        return
    if not DOCLENS_API_KEY:
        logger.warning("DOCLENS_API_KEY not set — API calls will fail")

    await init_db()
    dp.include_router(router)

    logger.info("Bot starting...")
    logger.info(f"DocLens API: {DOCLENS_API_URL}")

    # Set menu button to open MiniApp
    try:
        webapp_url = get_webapp_url()
        await bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(
                text="📱 DocLens",
                web_app=WebAppInfo(url=webapp_url),
            )
        )
        logger.info(f"Menu button set: {webapp_url}")
    except Exception as e:
        logger.warning(f"Failed to set menu button: {e}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
