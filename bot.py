import asyncio
import os
from groq import Groq
from aiogram import Bot, Dispatcher, types, F
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from aiogram.filters import Command
from kids_db import init_db, save_child, get_child
from sick_db import init_sick_table, add_symptom, get_symptoms_last_hours, get_symptoms_older
from vaccines import get_vaccines_by_age
from subscriptions import init_subs_table, subscribe, unsubscribe, is_subscribed, get_all_subscribers

import os
TOKEN = os.getenv("TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
import httpx
groq_client = Groq(api_key=GROQ_API_KEY, timeout=httpx.Timeout(120.0, connect=30.0))
# ЗАГРУЗКА ЗНАНИЯ
print("Загружаю знания ...")
embeddings = FastEmbedEmbeddings(
    model_name="intfloat/multilingual-e5-small"
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="medical_docs",
)
print("База знаний готова")
init_db()
init_sick_table()
init_subs_table()

# ПРОМПТ
SAFETY_PROMPT = """
ТЫ — МЕДИЦИНСКИЙ ПОМОЩНИК ДЛЯ МАМ. СТРОГИЕ ПРАВИЛА:

🚫 НИКОГДА НЕ НАЗНАЧАЙ рецептурные препараты (антибиотики, гормоны, противотуберкулёзные, цитостатики).
   Если в источнике упомянуто такое лекарство — пиши: «Это лечится только под контролем врача. Обратитесь к педиатру.»

✅ МОЖНО упоминать безрецептурные препараты (парацетамол, ибупрофен) с пометкой:
   «Дозировку уточните у врача или в инструкции.»

❌ НЕ пиши уровни доказательности (A, B, C, I, IIa) — они непонятны родителям.

🚨 КРАСНЫЕ ФЛАГИ (отвечай немедленно: «Вызывайте скорую 103!»):
   - Температура выше 39°C, не спадает больше часа
   - Судороги
   - Затруднённое дыхание, синюшность
   - Сыпь с температурой
   - Ребёнок не просыпается, вялый
   - Кровь в стуле или рвоте
   - Обезвоживание (не писает 6+ часов)

📌 ВСЕГДА заканчивай ответ одной из фраз:
   - «Обсудите это с педиатром.»
   - «Если сомневаетесь — позвоните в скорую (103).»

📝 Отвечай кратко (до 10 строк), простым языком, со смайликами, без таблиц.
"""


def search_in_base(question: str, k: int = 3) -> str:
    """Ищет релевантные куски в базе знаний."""
    try:
        results = vectorstore.similarity_search(question, k=k)
        context_parts = []
        for doc in results:
            import os
            source = os.path.basename(doc.metadata.get("source", "неизвестно"))
            context_parts.append(f"[Источник: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        print(f"⚠️ Ошибка поиска в базе: {e}")
        return ""
# Храним контекст последнего анализа для каждого пользователя
user_context = {}
def split_message(text, max_length=4000):
    """Разбивает длинный текст на части по 4000 символов."""
    parts = []
    while len(text) > max_length:
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(text[:split_index])
        text = text[split_index:].strip()
    parts.append(text)
    return parts


async def check_facts(text: str) -> str:
    """Проверяет текст на медицинские факты с опорой на базу знаний."""
    # Ищем в базе знаний
    context = search_in_base(text, k=3)

    system_prompt = SAFETY_PROMPT + """

ФОРМАТ ОТВЕТА:
1. Не пиши таблицы, не используй символы | и ---
2. Сначала одна короткая строка: о чём вообще это видео/текст (1 предложение)
3. Потом маркированный список с эмодзи:
   ✅ или ❌ в начале каждой строки
   Коротко: что утверждается и правда это или нет
4. В конце — 1-2 коротких совета с эмодзи 💡
5. Весь ответ — максимум 15 строк

Если в предоставленных источниках нет информации по теме — честно скажи,
что не нашёл в рекомендациях, и дай ответ из общих знаний, но с пометкой.
"""

    user_message = f"""Проверь этот текст: {text}

Вот информация из клинических рекомендаций (используй её как основу):
{context if context else "В базе нет подходящих источников."}
"""

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        model="openai/gpt-oss-120b",
    )

    return chat_completion.choices[0].message.content
async def answer_question(question: str, context: dict) -> str:
    """Отвечает на уточняющий вопрос, опираясь на базу и предыдущий анализ."""
    # Ищем в базе знаний
    base_context = search_in_base(question, k=3)

    system_prompt = SAFETY_PROMPT + """

Ты отвечаешь на уточняющий вопрос. Пользователь ранее прислал материал,
ты его проанализировал. Теперь он задаёт вопрос.
Опирайся на базу знаний, если там есть информация. Если нет — отвечай из общих знаний с пометкой.
"""

    user_message = f"""
Раньше пользователь прислал: {context['type']}
Содержание: {context['transcription']}
Твой вердикт был: {context['verdict']}

Вот что есть в клинических рекомендациях:
{base_context if base_context else "В базе нет подходящих источников."}

Теперь вопрос: {question}
"""

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        model="openai/gpt-oss-120b",
    )

    return chat_completion.choices[0].message.content
def split_message(text, max_length=4000):
    """Разбивает длинный текст на части по 4000 символов."""
    parts = []
    while len(text) > max_length:
        # Ищем последний перенос строки или пробел перед лимитом
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length  # Если нет пробелов, режем прямо так
        parts.append(text[:split_index])
        text = text[split_index:].strip()
    parts.append(text)
    return parts

from aiogram.client.session.aiohttp import AiohttpSession
session = AiohttpSession(timeout=300)  # 5 минут
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    await message.answer("Привет, мама!👋 Ты пришла в место спокойствия и правды, касаемо здоровья твоих детей💓 Всем нам знакомо чувство паники, когда соседка 🗣️говорит одно, бабушка - второе, а интернет совершенно другое… Мы здесь, чтобы помочь найти истину в приводимых утверждениях💪🏼Просто отправь видео, пост или высказывание в сфере здоровья ребенка, в котором ты сомневаешься, и я дам один однозначный ответ, основанный на клинических рекомендациях, ВОЗ.📚🤝С помощью кнопки Help, нужно выбрать интересующую опцию✅")

@dp.message(F.video)
async def handle_video(message: types.Message):
    await message.answer("Видео получил! Обрабатываю, мне нужна минутка и я смогу Вам помочь...")
    os.makedirs("videos", exist_ok=True)
    
    video_path = f"videos/{message.video.file_id}.mp4"
    audio_path = f"videos/{message.video.file_id}.wav"
    
    try:
        # 1. Скачиваем видео
        await bot.download(message.video, video_path)
        await message.answer("Видео уже скачал! Извлекаю звук...")
        
        # 2. Извлекаем звук (используем ffmpeg, который лежит в папке)
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-filter:a", "atempo=0.85", audio_path, "-y"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            await message.answer(f"Ошибка при извлечении звука: {result.stderr[:200]}")
            return

        await message.answer("Звук извлечен! Анализирую, чтобы быстрее Вам помочь...")
        
        # 3. Отправляем звук в Groq
        with open(audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
               file=(audio_path, file.read()),
               model="whisper-large-v3",
               response_format="text",
               language="ru",
               prompt="ОРВИ, ОРЗ, грипп, COVID-19, педиатрия, фитонциды, эфирные масла, антибиотики, противовирусные, ибупрофен, парацетамол, преднизолон, иммунитет, вакцинация, гомеопатия, БАДы, клинические рекомендации, врач-педиатр, температура, кашель, насморк"
            )
    
                # 4. Отправляем текст в Groq для проверки фактов
        await message.answer("Осталось немного и я смогу Вам помочь! Проверяю медицинские утверждения в тексте...")
        
                # Системный промпт — это инструкция для нейросети, как ей себя вести
        system_prompt = """
        Ты — медицинский фактчекер. Твоя задача — проверить медицинские утверждения на достоверность.
        
        ФОРМАТ ОТВЕТА (строго соблюдай):
        1. Не пиши таблицы, не используй символы | и ---
        2. Не повторяй весь текст целиком
        3. Сначала одна короткая строчка: о чём вообще это видео/текст (1 предложение)
        4. Потом маркированный список с эмодзи:
           ✅ или ❌ в начале каждой строчки
           Коротко: что утверждается и правда это или нет
        5. В конце — 1-2 коротких совета с эмодзи 💡
        6. Весь ответ — максимум 15 строк
        
        Пример правильного ответа:
        
        🎬 В этом видео говорится о лечении ОРВИ чесноком.
        
        ❌ Чеснок не убивает вирусы ОРВИ — научных доказательств нет.
        ✅ При первых симптомах нужно обратиться к врачу.
        ❌ Заменять лекарства чесноком опасно, особенно для детей.
        
        💡 При ОРВИ: покой, обильное питьё, жаропонижающее по назначению врача.
        💡 Не занимайтесь самолечением — вызовите педиатра на дом.
        """
        
        # Отправляем запрос нейросети Llama 3
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Проверь этот текст: {transcription}"}
            ],
            model="openai/gpt-oss-120b", # Актуальная модель Groq
        )
        
        # Получаем ответ от нейросети
        ai_response = chat_completion.choices[0].message.content
        # Сохраняем контекст для уточняющих вопросов
        user_context[message.chat.id] = {
            "type": "видео",
            "transcription": transcription,
            "verdict": ai_response
        }
        # Отправляем ответ пользователю по частям
        full_text = ai_response
        parts = split_message(full_text)
        for part in parts:
            await message.answer(part)
        
    except Exception as e:
        print(f"❌ ОШИБКА в handle_video: {type(e).__name__}: {e}")
        await message.answer(f"Произошла ошибка: {e}")

@dp.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: types.Message):
    chat_id = message.chat.id
    text = message.text.strip()
    text_lower = text.lower()

    # === 1. ЕСЛИ ЕСТЬ КОНТЕКСТ И ЭТО ВОПРОС ПРО "ПОЧЕМУ" — ОТВЕЧАЕМ РАЗВЁРНУТО ===
    if (
        chat_id in user_context
        and "sick" in user_context[chat_id].get("type", "")
        and any(w in text_lower for w in ("почему", "зачем", "отчего", "из-за чего", "что не так"))
    ):
        context_data = user_context[chat_id]
        await message.answer("💭 Объясняю...")

        prompt = f"""
Маме только что было сказано: {context_data['verdict']}
Контекст: {context_data['transcription']}

Мама спрашивает: "{text}"

Объясни простым языком (до 8 строк, со смайликами):
1. Почему это опасно для ребёнка.
2. Что может произойти, если не обратиться.
3. Что делать прямо сейчас.

Не назначай лекарства, не пугай лишним.
"""
        try:
            context_base = search_in_base(f"{context_data['transcription']} опасность для ребёнка", k=3)
            chat = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SAFETY_PROMPT},
                    {"role": "user", "content": prompt + f"\n\nВот что есть в рекомендациях:\n{context_base if context_base else 'нет данных'}"}
                ],
                model="openai/gpt-oss-120b",
            )
            answer = chat.choices[0].message.content
            for part in split_message(answer):
                await message.answer(part)
        except Exception as e:
            await message.answer(f"Ошибка: {e}")
        return

    # === 2. ЕСЛИ ЭТО ОБЫЧНЫЙ УТОЧНЯЮЩИЙ ВОПРОС ===
    if chat_id in user_context and text.endswith("?"):
        await message.answer("💭 Думаю над вашим вопросом...")
        try:
            answer = await answer_question(text, user_context[chat_id])
            for part in split_message(answer):
                await message.answer(part)
        except Exception as e:
            await message.answer(f"Произошла ошибка: {e}")
        return

    # === 3. ИНАЧЕ — НОВЫЙ ФАКТ ДЛЯ ПРОВЕРКИ ===
    await message.answer("Текст получен! Анализирую на лжемедицинские факты...")
    try:
        verdict = await check_facts(text)
        user_context[chat_id] = {
            "type": "текст",
            "transcription": text,
            "verdict": verdict
        }
        for part in split_message(verdict):
            await message.answer(part)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    await message.answer("📷 Картинка получена! Читаю текст и проверяю факты исходя из клинических рекомендаций...")

    os.makedirs("images", exist_ok=True)
    photo_path = f"images/{message.photo[-1].file_id}.jpg"

    try:
        # Скачиваем фото
        await bot.download(message.photo[-1], photo_path)

        # Настраиваем путь к Tesseract
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Открываем и распознаём текст (русский + английский)
        from PIL import Image
        img = Image.open(photo_path)
        recognized_text = pytesseract.image_to_string(img, lang='rus+eng').strip()

        # Подпись к фото (если есть)
        caption = (message.caption or "").strip()

        # Собираем общий текст: распознанный с картинки + подпись
        combined_text = ""
        if recognized_text:
            combined_text += recognized_text
        if caption:
            if combined_text:
                combined_text += "\n\n" + caption
            else:
                combined_text = caption

        # Если вообще ничего не нашли
        if not combined_text.strip():
            await message.answer(
                "🤔 Не могу прочитать текст на картинке.\n"
                "Попробуйте прислать её крупнее или напишите текст сообщением."
            )
            return

        # Проверяем факты через базу знаний
        verdict = await check_facts(combined_text)

        # Сохраняем контекст для уточняющих вопросов
        user_context[message.chat.id] = {
            "type": "картинка",
            "transcription": combined_text,
            "verdict": verdict
        }

        parts = split_message(verdict)
        for part in parts:
            await message.answer(part)

    except Exception as e:
        await message.answer(f"Произошла ошибка при обработке картинки: {e}")

@dp.message(F.voice)
async def handle_voice(message: types.Message):
    await message.answer("🎤 Голосовое получено! Слушаю...")

    os.makedirs("voices", exist_ok=True)
    voice_path = f"voices/{message.voice.file_id}.ogg"

    try:
        # Скачиваем голосовое
        await bot.download(message.voice, voice_path)

        # Отправляем в Groq на расшифровку
        with open(voice_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(voice_path, file.read()),
                model="whisper-large-v3",
                response_format="text",
                language="ru",
                prompt="ОРВИ, ОРЗ, грипп, педиатрия, фитонциды, эфирные масла, антибиотики, ибупрофен, врач, лечение, прикорм, прививки"
            )

        await message.answer(f"📝 Я услышал:\n_{transcription}_", parse_mode="Markdown")

        # Проверяем факты через RAG
        verdict = await check_facts(transcription)

        # Сохраняем контекст для уточняющих вопросов
        user_context[message.chat.id] = {
            "type": "голосовое",
            "transcription": transcription,
            "verdict": verdict
        }

        parts = split_message(verdict)
        for part in parts:
            await message.answer(part)

    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")

@dp.message(F.video_note)
async def handle_video_note(message: types.Message):
    await message.answer("🎥 Кружок получен! Слушаю...")

    os.makedirs("circles", exist_ok=True)
    circle_path = f"circles/{message.video_note.file_id}.mp4"
    audio_path = f"circles/{message.video_note.file_id}.wav"

    try:
        # Скачиваем кружок
        await bot.download(message.video_note, circle_path)

        # Извлекаем звук через ffmpeg
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-i", circle_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", "-filter:a", "atempo=0.95",
             audio_path, "-y"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            await message.answer(f"Ошибка при извлечении звука: {result.stderr[:200]}")
            return

        await message.answer("🎧 Звук извлечён! Расшифровываю...")

        # Отправляем в Whisper
        with open(audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="text",
                language="ru",
                prompt="ОРВИ, ОРЗ, грипп, педиатрия, фитонциды, эфирные масла, антибиотики, ибупрофен, врач, лечение, прикорм, прививки"
            )

        await message.answer(f"📝 Я услышал:\n_{transcription}_", parse_mode="Markdown")

        # Проверяем факты
        verdict = await check_facts(transcription)

        # Сохраняем контекст
        user_context[message.chat.id] = {
            "type": "кружок",
            "transcription": transcription,
            "verdict": verdict
        }

        parts = split_message(verdict)
        for part in parts:
            await message.answer(part)

    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")

@dp.message(Command("mykid"))
async def cmd_mykid(message: types.Message):
    args = message.text.replace("/mykid", "").strip()
    user_id = message.from_user.id

    if not args:
        child = get_child(user_id)
        if child:
            age = child['age_months']
            await message.answer(
                f"👶 *Профиль малыша*\n\n"
                f"Возраст: *{age} мес.*\n"
                f"Пол: *{child['gender']}*\n\n"
                f"💡 *Что интересует?* Напишите `/ask` + тему:\n"
                f"💉 `/vaccines` — календарь прививок по возрасту\n\n"
                f"🍼 `/ask что кушать?`\n"
                f"😴 `/ask как наладить сон?`\n"
                f"🧸 `/ask какие игрушки?`\n"
                f"💉 `/ask какие прививки?`\n"
                f"🎯 `/ask что должен уметь?`\n\n"
                f"Чтобы обновить возраст: `/mykid 6 мальчик`",
                parse_mode="Markdown"
            )
        else:
            await message.answer(
                "👶 *Давайте знакомиться с малышом!*\n\n"
                "Я буду учитывать возраст и пол ребёнка, когда ты задаёшь вопросы "
                "о развитии, прикорме, прививках. Так мои ответы станут точнее. 🎯\n\n"
                "*Напишите команду с возрастом и полом:*\n"
                "`/mykid 5 мальчик`\n"
                "`/mykid 8 девочка`\n"
                "`/mykid 1 год мальчик`\n\n"
                "Потом можно спрашивать: `/ask что кушать в 5 месяцев?`",

                parse_mode="Markdown"
            )
        return

       # Разбираем аргументы (гибкий парсинг)
    import re
    text_lower = args.lower()
    
    # Ищем возраст
    age = None
    # Проверяем "1 год", "2 года", "3 лет"
    year_match = re.search(r'(\d+)\s*(год|года|лет|г\.?)', text_lower)
    if year_match:
        age = int(year_match.group(1)) * 12
    else:
        # Проверяем "5 месяцев", "6 мес", или просто "5"
        month_match = re.search(r'(\d+)\s*(месяц|мес|м\.?)', text_lower)
        if month_match:
            age = int(month_match.group(1))
        else:
            # Ищем любое число в тексте
            num_match = re.search(r'\b(\d+)\b', text_lower)
            if num_match:
                age = int(num_match.group(1))
    
    # 2. Ищем пол
    gender = None
    if any(w in text_lower for w in ("мальчик", "мальч", "сын", " boy", "муж")):
        gender = "мальчик"
    elif any(w in text_lower for w in ("девочка", "дев", "дочь", "girl", "жен")):
        gender = "девочка"
    
    # Если ничего не нашли — сообщаем
    if age is None and gender is None:
        await message.answer(
            "🤔 Не понял возраст и пол. Напишите так:\n"
            "`/mykid 5 мальчик` или `/mykid 5 месяцев мальчик` или `/mykid 1 год девочка`",
            parse_mode="Markdown"
        )
        return

    save_child(user_id, age_months=age, gender=gender)
    child = get_child(user_id)

    # Просим нейросеть дать сводку по возрасту
    prompt = (
        f"Мама сообщила, что её ребёнку теперь {child['age_months']} мес., пол: {child['gender']}. "
        f"Поздравь её тепло и кратко (до 10 строк, со смайликами, без таблиц) напиши: "
        f"1) что обычно умеет ребёнок в этом возрасте; "
        f"2) какие прививки по календарю (если актуально); "
        f"3) что нового в прикорме (если актуально); "
        f"4) один интересный факт про этот возраст."
    )

    # Ищем в базе знаний
    context = search_in_base(f"развитие ребёнка {child['age_months']} месяцев", k=3)

    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SAFETY_PROMPT},
                {"role": "user", "content": prompt + f"\n\nВот что есть в рекомендациях:\n{context if context else 'нет данных'}"}
            ],
            model="openai/gpt-oss-120b",
        )
        summary = chat.choices[0].message.content
        for part in split_message(summary):
            await message.answer(part)
        # Меню тем
        age = child['age_months']
        menu = (
            f"\n\n💡 *Что вам сейчас интересно узнать про вашего ребёнка?*\n\n"
            f"Напишите `/ask` и тему. Например:\n\n"
            f"💉 `/vaccines` — календарь прививок по возрасту\n\n"
            f"🍼 `/ask что кушать в {age} месяцев?`\n"
            f"😴 `/ask как наладить сон?`\n"
            f"🧸 `/ask какие игрушки подойдут?`\n"
            f"💉 `/ask какие прививки нужны?`\n"
            f"🎯 `/ask что должен уметь в {age} месяцев?`\n"
            f"🧷 `/ask как ухаживать за кожей?`\n\n"
            f"Или просто задайте свой вопрос."
        )
        await message.answer(menu, parse_mode="Markdown")
    except Exception as e:
        await message.answer(f"✅ Профиль сохранён! (Ошибка сводки: {e})")

@dp.message(Command("ask"))
async def cmd_ask(message: types.Message):
    child = get_child(message.from_user.id)
    if not child:
        await message.answer("Сначала создайте профиль: `/mykid 3 мальчик`", parse_mode="Markdown")
        return

    question = message.text.replace("/ask", "").strip()
    if not child:
        await message.answer(
            "👶 *Сначала познакомимся!*\n\n"
            "Чтобы я отвечал с учётом возраста малыша, создайте профиль:\n"
            "`/mykid 5 мальчик`\n\n"
            "Потом задавайте вопросы через `/ask`.",
            parse_mode="Markdown"
        )
        return

    question = message.text.replace("/ask", "").strip()
    if not question:
        await message.answer(
            "💬 *Задайте вопрос о здоровье малыша*\n\n"
            "Я поищу ответ в клинических рекомендациях и объясню простым языком.\n"
            "Учитываю возраст и пол ребёнка из профиля.\n\n"
            "*Примеры:*\n"
            "`/ask что кушать в 5 месяцев?`\n"
            "`/ask какие игрушки подойдут?`\n"
            "`/ask как наладить сон?`\n"
            "`/ask какие прививки нужны?`",
            parse_mode="Markdown"
        )
        return

    await message.answer("💭 Думаю...")

    # Ищем в базе знаний
    search_query = f"ребёнок {child['age_months']} месяцев: {question}"
    base_context = search_in_base(search_query, k=3)

    prompt = (
        f"Мама спрашивает про ребёнка: возраст {child['age_months']} мес., пол {child['gender']}. "
        f"Вопрос: {question}\n\n"
        f"Вот что есть в клинических рекомендациях:\n"
        f"{base_context if base_context else 'нет данных'}\n\n"
        f"Отвечай кратко (до 10 строк), тепло, со смайликами, без таблиц. "
        f"Опирайся на рекомендации, если есть информация."
    )

    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SAFETY_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-120b",
        )
        answer = chat.choices[0].message.content
        for part in split_message(answer):
            await message.answer(part)
    except Exception as e:
        await message.answer(f"Ошибка: {e}")

@dp.message(Command("checklist"))
async def cmd_checklist(message: types.Message):
    symptoms = message.text.replace("/checklist", "").strip()

    if not symptoms:
        await message.answer(
            "📋 *Команда для чек-листа вопросов врачу*\n\n"
            "Часто бывает так: приходишь к педиатру, выходишь из кабинета — "
            "и понимаешь, что забыла спросить самое важное. 🤦‍♀️\n\n"
            "Я помогу этого избежать! Напишите команду ещё раз и кратко опишите "
            "состояние ребёнка — я составлю список вопросов, которые стоит задать врачу.\n\n"
            "*Пример:*\n"
            "`/checklist температура 37, кашель`\n"
            "`/checklist насморк, отказывается есть, вялый`",
            parse_mode="Markdown"
        )
        return

    await message.answer("📋 Готовлю вопросы для педиатра...")

    # Ищем в базе знаний
    context = search_in_base(f"симптомы: {symptoms}", k=3)

    prompt = f"""
Мама заметила у ребёнка такие симптомы: {symptoms}

Вот что есть в клинических рекомендациях:
{context if context else "нет данных"}

Составь список из 5-7 конкретных вопросов, которые маме стоит задать педиатру на приёме.
Формат ответа:
📋 *Вопросы для педиатра:*
1. ...
2. ...
3. ...

В конце добавь одну строку: "💡 Запишите ответы врача, чтобы не забыть."
Не пиши диагнозы, не назначай лекарства. Только вопросы.
"""

    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SAFETY_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-120b",
        )
        answer = chat.choices[0].message.content
        for part in split_message(answer):
            await message.answer(part)
    except Exception as e:
        await message.answer(f"Ошибка: {e}")

@dp.message(Command("sick"))
async def cmd_sick(message: types.Message):
    import re
    from datetime import datetime as dt

    user_id = message.from_user.id
    args = message.text.replace("/sick", "").strip()

    # === 1. Если без аргументов — показываем сводку ===
    if not args:
        rows = get_symptoms_last_hours(user_id, hours=24)

        if not rows:
            await message.answer(
                "🌡 *Дневник симптомов*\n\n"
                "Помогает отслеживать состояние малыша и замечать опасные изменения.\n\n"
                "*Записывайте так:*\n"
                "`/sick 38.5` — температура\n"
                "`/sick кашель` — симптом\n"
                "`/sick понос 5 раз` — с деталями\n\n"
                "Через сутки напишите `/sick` — покажу сводку и предупрежу, если что-то не так.",
                parse_mode="Markdown"
            )
            return

        # Собираем сводку
        summary = "📊 *Сводка за последние 24 часа:*\n\n"
        temps = []
        for symptom, value, is_temp, ts in rows:
            time_str = dt.fromisoformat(ts).strftime("%H:%M")
            if is_temp:
                summary += f"🌡 `{time_str}` — температура *{value}°C*\n"
                temps.append(float(value))
            else:
                extra = f" — {value}" if value else ""
                summary += f"• `{time_str}` — {symptom}{extra}\n"

        # Красные флаги
        if temps:
            max_temp = max(temps)
            last_temp = temps[-1]
            if last_temp >= 39:
                summary += "\n🚨 *Температура ≥39°C — срочно вызывайте скорую (103)!*"
            elif max_temp >= 39:
                summary += "\n⚠️ За сутки была температура ≥39°C. Если не спадает — вызывайте скорую."

        # Проверяем, держится ли температура больше суток
        older = get_symptoms_older(user_id, hours=24)
        if older > 0 and temps:
            summary += "\n\n⚠️ *Температура держится больше 24 часов.* Это повод обратиться к врачу."

        await message.answer(summary, parse_mode="Markdown")

        # Сохраняем контекст, если была высокая температура
        if temps and max(temps) >= 39:
            user_context[user_id] = {
                "type": "sick_temp_high",
                "transcription": f"за сутки была температура {max(temps)}°C у ребёнка",
                "verdict": "Высокая температура (≥39°C) в течение суток — красный флаг, требует вызова скорой помощи."
            }
        elif temps and max(temps) >= 38:
            user_context[user_id] = {
                "type": "sick_temp_mid",
                "transcription": f"за сутки была температура {max(temps)}°C у ребёнка",
                "verdict": "Умеренная температура за сутки — нужно следить за динамикой."
            }

        # Анализ от нейросети
        await message.answer("💭 Анализирую...")
        try:
            symptoms_text = ", ".join([f"{s} {v}".strip() for s, v, _, _ in rows])
            context = search_in_base(f"симптомы у ребёнка: {symptoms_text}", k=3)

            prompt = f"""
У ребёнка за последние 24 часа зафиксированы такие симптомы: {symptoms_text}

Вот что есть в клинических рекомендациях:
{context if context else "нет данных"}

Дай маме краткий комментарий (до 8 строк, со смайликами):
1. Что это может быть (без диагноза).
2. Что делать сейчас.
3. Когда срочно к врачу.

Не назначай лекарства.
"""
            chat = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SAFETY_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-oss-120b",
            )
            answer = chat.choices[0].message.content
            for part in split_message(answer):
                await message.answer(part)
        except Exception as e:
            await message.answer(f"Ошибка анализа: {e}")
        return

    # === 2. Записываем новый симптом ===
    text_lower = args.lower()

    # Ищем температуру (36-42)
    temp_match = re.search(r'\b(3[6-9]|4[0-2])[.,]?\d?\b', text_lower)

    if temp_match:
        value = temp_match.group(0).replace(",", ".")
        add_symptom(user_id, "температура", value, is_temperature=True)
        num = float(value)

        if num >= 39:
            await message.answer(
                f"🌡 Записал температуру *{value}°C* в {dt.now().strftime('%H:%M')}.\n\n"
                f"🚨 *Температура ≥39°C! Срочно вызывайте скорую (103).*\n\n"
                f"💬 Хотите узнать почему? Напишите «почему?»",
                parse_mode="Markdown"
            )
            # Сохраняем контекст для вопроса "почему?"
            user_context[user_id] = {
                "type": "sick_temp_high",
                "transcription": f"температура {value}°C у ребёнка",
                "verdict": "Высокая температура (≥39°C) — это красный флаг, требующий вызова скорой помощи."
            }
        elif num >= 38:
            await message.answer(
                f"🌡 Записал температуру *{value}°C* в {dt.now().strftime('%H:%M')}.\n\n"
                f"💡 Следите за состоянием. Если не спадает или поднимается — звоните врачу.",
                parse_mode="Markdown"
            )
            user_context[user_id] = {
                "type": "sick_temp_mid",
                "transcription": f"температура {value}°C у ребёнка",
                "verdict": "Умеренная температура — нужно следить за динамикой, при ухудшении обращаться к врачу."
            }
    else:
        # Это не температура — записываем как симптом
        add_symptom(user_id, args, "", is_temperature=False)
        await message.answer(
            f"✅ Записал: *{args}* в {dt.now().strftime('%H:%M')}.\n\n"
            f"💡 Напишите `/sick` через сутки — покажу сводку.",
            parse_mode="Markdown"
        )

@dp.message(Command("vaccines"))
async def cmd_vaccines(message: types.Message):
    child = get_child(message.from_user.id)

    if not child:
        await message.answer(
            "👶 *Сначала создадим профиль!*\n\n"
            "Напишите: `/mykid 5 мальчик`\n\n"
            "После этого `/vaccines` покажет нужные прививки.",
            parse_mode="Markdown"
        )
        return

    age = child["age_months"]
    if age is None:
        await message.answer("⚠️ В профиле не указан возраст. Обновите, пожалуйста: `/mykid 5 мальчик`", parse_mode="Markdown")
        return

    done, soon, future = get_vaccines_by_age(age)

    if age < 12:
        age_str = f"{age} мес."
    else:
        years = age // 12
        months = age % 12
        age_str = f"{years} г." + (f" {months} мес." if months else "")

    text = f"💉 *Календарь прививок для ребёнка {age_str}*\n\n"

    if done:
        text += "✅ *Уже должны быть сделаны:*\n"
        for months, name in done:
            text += f"• {name}\n"
        text += "\n"

    if soon:
        text += "⏳ *Скоро (в ближайшие 2 месяца):*\n"
        for months, name in soon:
            text += f"• {name}\n"
        text += "\n"

    if future:
        text += "📅 *В будущем:*\n"
        for months, name in future[:3]:
            text += f"• {name}\n"
        if len(future) > 3:
            text += f"• ... и ещё {len(future) - 3}\n"
        text += "\n"

    text += "💡 *Важно:* проверьте с педиатром, какие прививки уже сделаны. "
    text += "Если что-то пропущено — врач составит индивидуальный план."

    await message.answer(text, parse_mode="Markdown")

async def generate_myth() -> str:
    """Генерирует случайный миф с опровержением из базы знаний."""
    import random
    topics = [
        "чеснок при ОРВИ", "гомеопатия при простуде", "керосин от вшей",
        "уринотерапия", "чудо-капли в нос", "лечение кашля банками",
        "обливание холодной водой при температуре", "витамин C от простуды",
        "прививки и аутизм", "молоко с мёдом от кашля", "горчичники при бронхите",
        "спиртовые компрессы детям", "травяные настои вместо лекарств",
    ]
    topic = random.choice(topics)
    context = search_in_base(topic, k=3)

    prompt = f"""
Тема для мифа дня: {topic}

Вот что есть в клинических рекомендациях:
{context if context else "нет данных"}

Составь миф дня в таком формате:
🎭 *Миф:* ...
❌ *Правда:* ...
✅ *Что делать:* ...
💡 *Совет:* ...

Кратко (до 8 строк), со смайликами, без таблиц. Не назначай лекарства.
"""
    chat = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": SAFETY_PROMPT},
            {"role": "user", "content": prompt}
        ],
        model="openai/gpt-oss-120b",
    )
    return chat.choices[0].message.content


@dp.message(Command("myth"))
async def cmd_myth(message: types.Message):
    await message.answer("🎭 Готовлю миф дня...")
    try:
        myth = await generate_myth()
        parts = split_message(myth)

        if os.path.exists("assets/myth_banner.jpg"):
            await message.answer_photo(
                photo=types.FSInputFile("assets/myth_banner.jpg"),
                caption=parts[0] if parts else None,
            )
            for part in parts[1:]:
                await message.answer(part)
        else:
            for part in parts:
                await message.answer(part)
    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@dp.message(Command("subscribe"))
async def cmd_subscribe(message: types.Message):
    user_id = message.from_user.id
    if is_subscribed(user_id):
        await message.answer("📬 Вы уже подписаны на «Миф дня». Каждый день в 10:00 буду присылать полезное!")
        return
    subscribe(user_id)
    await message.answer(
        "✅ *Подписка оформлена!*\n\n"
        "📬 Каждый день в 10:00 я буду присылать Вам один миф о детском здоровье "
        "и его научное опровержение.\n\n"
        "Отписаться: `/unsubscribe`",
        parse_mode="Markdown"
    )


@dp.message(Command("unsubscribe"))
async def cmd_unsubscribe(message: types.Message):
    unsubscribe(message.from_user.id)
    await message.answer("🔕 Подписка отменена. Если захотите снова — напишите `/subscribe`.")


async def daily_myth_scheduler():
    """Каждый день в 10:00 рассылает миф дня всем подписчикам."""
    import asyncio
    from datetime import datetime, timedelta

    while True:
        now = datetime.now()
        next_run = now.replace(hour=10, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        wait_seconds = (next_run - now).total_seconds()
        print(f"⏰ Следующая рассылка мифа через {wait_seconds / 3600:.1f} ч")
        await asyncio.sleep(wait_seconds)

        print("📬 Рассылаю миф дня...")
        try:
            myth = await generate_myth()
            parts = split_message(myth)
            subscribers = get_all_subscribers()
            has_image = os.path.exists("assets/myth_banner.jpg")

            for user_id in subscribers:
                try:
                    if has_image and parts:
                        await bot.send_photo(
                            chat_id=user_id,
                            photo=types.FSInputFile("assets/myth_banner.jpg"),
                            caption=parts[0],
                        )
                        for part in parts[1:]:
                            await bot.send_message(user_id, part)
                    else:
                        for part in parts:
                            await bot.send_message(user_id, part)
                except Exception as e:
                    print(f"⚠️ Не отправилось {user_id}: {e}")
            print(f"✅ Миф отправлен {len(subscribers)} подписчикам")
        except Exception as e:
            print(f"⚠️ Ошибка рассылки: {e}")


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    text = (
        "🤖 *Я — Лучик Правды, Ваш помощник для мамочек!*\n\n"
        "*Что я умею:*\n\n"
        "📝 *Проверка фактов*\n"
        "Отправьте мне текст, видео, голосовое, кружок или картинку — я проверю, правда это или миф.\n\n"
        "👶 *Профиль ребёнка*\n"
        "`/mykid 5 мальчик` — сохранить возраст и пол малыша\n"
        "`/mykid` — посмотреть текущий профиль\n\n"
        "💬 *Вопросы о здоровье*\n"
        "`/ask что кушать в 5 месяцев?` — вопрос с учётом возраста\n\n"
        "📋 *Чек-лист к врачу*\n"
        "`/checklist температура 38, кашель` — список вопросов для педиатра\n\n"
        "🌡 *Дневник симптомов*\n"
        "`/sick 38.5` — записать температуру\n"
        "`/sick кашель` — записать симптом\n"
        "`/sick` — сводка за 24 часа\n\n"
        "💉 *Прививки*\n"
        "`/vaccines` — календарь прививок по возрасту\n\n"
        "🎭 *Миф дня*\n"
        "`/myth` — миф дня сейчас\n"
        "`/subscribe` — ежедневная рассылка в 10:00\n"
        "`/unsubscribe` — отписаться\n\n"
        "💡 *Напоминание:* я не заменяю врача. Всегда уточняйте у педиатра!"
    )
    await message.answer(text, parse_mode="Markdown")

async def main():
    print("бот работает!")
    asyncio.create_task(daily_myth_scheduler())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
