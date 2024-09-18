from typing import Any, List

from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Response
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

from app import crud, models, schemas
from app.api import deps

from translate import Translator
from happytransformer import HappyTextToText as HappyTTT
from happytransformer import TTSettings
from textblob import *

router = APIRouter()

@router.post("/translate/",
  summary='Translate any text in other language',
  description='This api call simulates fetching all blogs',
  response_description="Test"
  )
async def translate_text(text: str, target_lang: str):
    try:
        translator = Translator(to_lang=target_lang)
        translated_text = translator.translate(text)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An error occurred during translation.")
    
@router.get('/grammer_fixer/{text}')
async def Grammer_Fixer(text: str):
    Grammer = HappyTTT("T5", "prithivida/grammar_error_correcter_v1")
    config = TTSettings(do_sample=True, top_k=10, max_length=100)
    corrected = Grammer.generate_text(text, args=config)
    return (corrected.text)


@router.get('/fix_paragraph_words/{paragraph}')
async def fix_paragraph_words(paragraph):
    sentence = TextBlob(paragraph)
    correction = sentence.correct()
    return (correction)


@router.get('/fix_word_spell/{word}')
async def fix_word_spell(word):
    word = Word(word)
    correction = word.correct()
    return (correction)