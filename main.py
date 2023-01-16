# coding:utf-8

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from typing import Optional
import shutil
from fastapi import FastAPI, UploadFile, File
from starlette.responses import FileResponse
import uvicorn
from pdf_parse_new import *
from urllib.parse import quote
import re

app = FastAPI()

pdf_extractor, vision_model, pdf_predictor, translate_tool, imath_detector = build_predictors()



@app.post("/translatePaper/")
async def translate_scientific_paper(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"Error": "File Type Error"}
    filename = re.sub(' +', ' ', file.filename)
    with open(os.path.join("api_receive_pdfs", filename), 'wb') as f:
        shutil.copyfileobj(file.file, f)
    pdf_tokens, pdf_images, normal_font = load_tokens_and_images(os.path.join("api_receive_pdfs", filename), pdf_extractor)
    pdf_blocks = generate_blocks(pdf_images, vision_model, imath_detector)
    pdf_all_data, imath_font = predicting_tokens(pdf_blocks, pdf_tokens, pdf_predictor)
    emath_font = mark_emath_token(pdf_all_data, imath_font, normal_font)
    pdf_all_data = get_math_map(pdf_all_data)
    output_dir = os.path.join("api_receive_pdfs", filename[:-4])
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, 'Figures/'))
        os.makedirs(os.path.join(output_dir, 'Tables/'))
        os.makedirs(os.path.join(output_dir, 'Equations/'))
        os.makedirs(os.path.join(output_dir, 'Expressions/'))
    pdf_blocks = generating_text(pdf_all_data, translate_tool, output_dir)
    height = fake_draw(pdf_all_data) + 20
    pdf = draw_pdf(pdf_images, pdf_all_data, height, pdf_tokens[0].page_size[0], output_dir)
    os.rename(os.path.join(output_dir, 'test.pdf'), os.path.join("api_receive_pdfs", 'zh_' + file.filename))
    shutil.rmtree(output_dir)
    return FileResponse(os.path.join("api_receive_pdfs", 'zh_' + file.filename), media_type = "application/pdf", headers={'Content-Disposition': 'attachment; filename="zh_' + quote(file.filename) + '"'})


@app.post("/translateSentence/")
async def translate_single_sentence(sentence: str):
    answer = translate(translate_tool, sentence).replace("\n", ' ')
    return {'result': answer}

