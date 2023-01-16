import os, re, json
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from ditod import MyTrainer
from ditod import add_vit_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import layoutparser as lp
import cv2
import numpy as np
import torch
import easyocr
from PIL import Image
import pdfplumber, pdf2image
import fitz
from vila.utils import union_lp_box
from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor, LayoutIndicatorPDFPredictor
import logging
from collections import Counter
logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
from fpdf import FPDF
from urllib.error import HTTPError
from json import dump
from easynmt import EasyNMT
import unicodedata
import nemo
import nemo.collections.nlp as nemo_nlp
import nltk
import itertools
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Frame
from paragraph import Paragraph   # 修改源码
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import red
import math
from typing import List
from fontTools.ttLib import TTFont as Font
from pdfplumber_extractor import PDFPlumberTokenExtractor    # 修改源码


img_token_rate = 2.5
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file('cascade_layoutlmv3.yaml')
cfg.freeze()
threshold = 0.6

font_tool = Font("fonts/TIMES.ttf")

math_symbols = "±∞≠~×÷∝<≪>≫≤≥∓≅≈≡∀∁∂√∛∜∪∩∅%°℉℃∆∇∃∄∈∋←↑→↓↔∴±¬αβγδεϵθϑμπρστφω*⋮⋯⋰⋱ℵℶαβγδεϵζηθϑικλμνξοπϖρϱσςτυφϕχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ∀∁∂ðℇϜℏ℩ıϰ℘℧Å℮∃∄ℵℶℷℸ+÷×±∓∝*∩∪⊎⊓⊔∧∨≮≰≯≱≃≈≅≢≄≉≇∝≪≫∈∋∉⊂⊃⊆⊇≺≻≼≽⊏⊐⊑⊒∥⊥⊢⊣⋈≍∑∫∬∭∮∯∰∱∲∳∏∐⋂⋃⋀⋁⨀⨂⨁⨄⨃∔∸∖⋒⋓⊟⊠⊡⊞⋇⋉⋊⋋⋌⋏⋎⊝⊺⊕⊖⊗⊘⊙⊛⊚†‡⋆⋄≀△⋀⋁⨀⨂⨁⨅⨆⨄⨃∴∵⋘⋙≦≧≲≳⋖⋗≶⋚≷⋛≑≒≓∽≊⋍≼≽⋞⋟≾≿⋜⋝⊆⊇⊲⊳⊴⊵⊨⋐⋑⊏⊐⊩⊪≖≗≜≏≎∝≬⋔≐⋈↔←→↑↓↕⇐⇒⟺⇑⇓⟹⟸⟷⟶⟵⇕⇔↗↖↘↙↚↛↮⇍⇏⇎⇋⇌⇉⇇≁⊄⊅⊈⊉⊊⊋⋢∌∉⋡≄≉≇≭≨≩⊀⊁⋠∤∦"
# 已经去除了-以及/
# 有争议：+,<,>
math_symbol_list = ['(cid:']
# name MATHEMATICAL ...
# 还要补充函数？
for i in math_symbols:
    if i not in math_symbol_list:
        math_symbol_list.append(i)

def build_predictors():
    # pdfplumber进行抽取
    pdf_extractor = PDFPlumberTokenExtractor()
    # pdf_extractor = PDFExtractor("pdfplumber")
    # layoutlmv3模型
    assert cfg.INPUT.MIN_SIZE_TEST == 800
    vision_model = MyTrainer.build_model(cfg)
    vision_model.eval()
    DetectionCheckpointer(vision_model).resume_or_load(cfg.MODEL.WEIGHTS, resume = False)
    # vila模型
    pdf_predictor = HierarchicalPDFPredictor.from_pretrained("allenai/hvila-block-layoutlm-finetuned-docbank")
    # MFD模型
    imath_detector = lp.Detectron2LayoutModel(config_path = "MFD_config.yaml",
        model_path = "/path/to/model_final-2.pth", 
        label_map = {1: "Equation"}, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.70])
    # 翻译模型
    # translate_tool = EasyNMT('mbart50_en2m', cache_folder = '/home/lizichao/code/unilm/layoutlmv3/examples/object_detection/cache-easynmt')
    # translate_tool = EasyNMT('opus-mt')
    translate_tool = nemo_nlp.models.machine_translation.MTEncDecModel.from_pretrained(model_name="nmt_en_zh_transformer24x6")
    
    return pdf_extractor, vision_model, pdf_predictor, translate_tool, imath_detector


def translate(translate_tool, src_content):
    if src_content == "":
        return ""
    paragraphs = src_content.split("\n")
    translated_content = []
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    for para in paragraphs:
        splitted_sentences = []
        temp_sentence = ''
        for sentence in nltk.sent_tokenize(para.strip()):
            if len(sentence.strip()) > 0:
                if len(temp_sentence) + len(sentence.strip()) < 512:
                    temp_sentence += " " + sentence.strip()
                else:
                    splitted_sentences.append(temp_sentence.strip())
                    temp_sentence = sentence.strip()
        if len(temp_sentence.strip()) > 0:
            splitted_sentences.append(temp_sentence.strip())
        
        mini_splitted_sentences = [splitted_sentences[i * 128 : min((i + 1) * 128, len(splitted_sentences))] for i in range(math.ceil(len(splitted_sentences) / 128))]
        translated_sentences = []
        for sentences in mini_splitted_sentences:
            translated_sentences.extend(translate_tool.translate(splitted_sentences, source_lang = "en", target_lang = 'zh'))
        translated_content.append(" ".join(translated_sentences))
    return "\n".join(translated_content)


def union_Rectangle(blocks: List[lp.TextBlock]):

    x1, y1, x2, y2 = float("inf"), float("inf"), float("-inf"), float("-inf")

    for bbox in blocks:
        _x1, _y1, _x2, _y2 = bbox.coordinates
        x1 = min(x1, _x1)
        y1 = min(y1, _y1)
        x2 = max(x2, _x2)
        y2 = max(y2, _y2)

    return lp.Rectangle(x1, y1, x2, y2)


def load_tokens_and_images(input_pdf, pdf_extractor):
    # 1 inch = 2.54 cm
    # 1 inch = 72 points

    pdf_handler = fitz.open(input_pdf)
    pdf_tokens = pdf_extractor(input_pdf)
    pdf_len = len(pdf_tokens)
    pdf_images = []
    trans = fitz.Matrix(img_token_rate, img_token_rate)
    for i in range(pdf_len):
        page = pdf_handler.load_page(i)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        pdf_images.append(Image.frombytes("RGB", [pm.width, pm.height], pm.samples))

    # 在这里统计正常（行文）字体
    font_words = {}
    normal_font = []
    for page_tokens in pdf_tokens:
        pre_font = ''
        pre_word = ''
        for token in page_tokens.tokens:
            font_size = token.font.split("-")[-1]
            font = token.font.rstrip("-" + font_size)
            token.text = unicodedata.normalize("NFKC", token.text)
            word = token.text
            if font not in font_words:
                font_words[font] = []
            if len(word) > 3 and re.match("^[a-zA-Z\-]+$", word) != None and len(pre_word) > 3 and re.match("^[a-zA-Z\-]+$", pre_word) != None and word not in font_words[font] and pre_font == font:
                font_words[font].append(word)
            elif (word.find("http") != -1 or word.find(".com") != -1) and word not in font_words[font]:
                font_words[font].append(word)
            pre_font = font
            pre_word = word
    for font in font_words:
        if len(font_words[font]) > 8:
            normal_font.append(font)

    return pdf_tokens, pdf_images, normal_font



def generate_blocks(pdf_images, vision_model, imath_detector):
    pdf_blocks = []
    label_map = {0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
    for pdf_image in pdf_images:
        image = utils.convert_PIL_to_numpy(pdf_image, 'RGB')
        tfm_gens = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")]
        img, _ = T.apply_transform_gens(tfm_gens, image)
        img_tensor = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        with torch.no_grad():
            res = vision_model.inference([{'image': img_tensor}])
        instance = res[0]['instances']
        boxes = instance.get('pred_boxes')
        scores = instance.get("scores").cpu().numpy()
        classes = instance.get("pred_classes").cpu().numpy()
        rate = image.shape[1] / 800
        blocks = []
        for i in range(len(classes)):
            if scores[i] < threshold:
                continue
            box = boxes[i].tensor.cpu().numpy()[0]
            rectangle = lp.elements.layout_elements.Rectangle(box[0] * rate / img_token_rate,
                box[1] * rate / img_token_rate, box[2] * rate / img_token_rate, box[3] * rate / img_token_rate)
            block = lp.elements.layout_elements.TextBlock(block = rectangle,
                score = scores[i], type = label_map[classes[i]], )
            blocks.append(block)
        # 这里的这些block之间的关系还需要优化
        imath_blocks = imath_detector.detect(pdf_image.resize((int(image.shape[1] / img_token_rate), int(image.shape[0] / img_token_rate))))
        figure_table_math_blocks = [b for b in blocks if b.type == 'Table' or b.type == 'Figure'] + [block for block in imath_blocks]
        real_blocks = [b for b in blocks if not any(b.is_in(b_fig) for b_fig in figure_table_math_blocks)] + figure_table_math_blocks
        real_blocks.sort(key=lambda block:block.coordinates[1])
        pdf_blocks.append(lp.Layout(real_blocks))
    return pdf_blocks


def predicting_tokens(pdf_blocks, pdf_tokens, pdf_predictor):
    pdf_all_data = []
    imath_font = []
    for i in range(len(pdf_blocks)):
        # page_token = pdf_tokens[i].scale(img_token_rate)
        page_tokens = pdf_tokens[i]
        page_tokens.annotate(blocks = pdf_blocks[i])
        page_data = page_tokens.to_pagedata().to_dict()
        page_data['labels'] = pdf_predictor.predict(page_data, page_tokens.page_size, return_type = "list")
        left_index = [j for j in range(len(page_tokens.tokens))]
        # token_groups = {}
        page_all_data = []
        for block in pdf_blocks[i]:
            # token_index_group = []
            remaining_indexs = []
            block_data = {'tokens': [], 'block': block}
            # preds = []
            for j in left_index:
                if block.id != None and block.id == page_data['block_ids'][j]:
                    page_tokens.tokens[j].label = page_data['labels'][j]
                    block_data['tokens'].append(page_tokens.tokens[j])
                elif page_tokens.tokens[j].is_in(block, soft_margin={"top": 1, "bottom": 1, "left": 1, "right": 1}, center=True):
                    block.id = page_data['block_ids'][j]
                    page_tokens.tokens[j].label = page_data['labels'][j]
                    block_data['tokens'].append(page_tokens.tokens[j])
                else:
                    remaining_indexs.append(j)
            left_index = remaining_indexs
            page_all_data.append(block_data)
            # 对isolated math expression的字体识别
            if block.type == "Equation":
                for token in block_data['tokens']:
                    font_size = token.font.split("-")[-1]
                    font = token.font.rstrip("-" + font_size)
                    if font not in imath_font:
                        imath_font.append(font)
            if block.type == 'Text' or block.type == 'List' or block.type == 'Title':
                if len(block_data['tokens']) > 0:
                    block.block = union_Rectangle(block_data['tokens'])
        block_data = {'block': None, 'tokens': []}
        for j in left_index:
            page_tokens.tokens[j].label = page_data['labels'][j]
            block_data['tokens'].append(page_tokens.tokens[j])

        page_all_data = sorted(page_all_data, key = lambda ele : ele['block'].id if ele['block'].id != None else float('-inf'))
        page_all_data.append(block_data)
        pdf_all_data.append(page_all_data)
        # left_tokens.append([pdf_tokens.tokens[j] for j in left_index])
    return pdf_all_data, imath_font


def mark_emath_token(pdf_all_data, imath_font, normal_font):
    emath_font = []
    # 第一遍循环
    for page_all_data in pdf_all_data:
        for data in page_all_data:
            block = data['block']
            if block != None and (block.type == "Equation" or block.type == 'Figure' or block.type == 'Table'):
                continue
            tokens = data['tokens']
            for i in range(len(tokens)):
                token = tokens[i]
                font_size = token.font.split("-")[-1]
                font = token.font.rstrip("-" + font_size)
                text = token.text

                flag = 0 # 标记是否为数学字符
                if font in imath_font and font not in normal_font:
                    flag = 1
                elif i > 0 and tokens[i - 1].is_math == True and font not in normal_font:
                    flag = 1
                else:
                    for symbol in text:
                        if symbol in math_symbol_list or unicodedata.name(symbol, '0').find("MATHEMATICAL") != -1:
                            flag = 1
                            break
                if flag == 1 and font not in normal_font:
                    token.is_math = True
                    if font not in emath_font:
                        emath_font.append(font)
                    for j in range(i - 1, -1, -1):
                        font_j = tokens[j].font.rstrip("-" + tokens[j].font.split("-")[-1])
                        if tokens[j].is_math == True or font_j in normal_font:
                            break
                        else:
                            tokens[j].is_math = True
                            if font_j not in emath_font:
                                emath_font.append(font_j)
                else:
                    token.is_math = False
    # 第二遍循环
    t = 0
    while t < 2:
      t += 1
      for page_all_data in pdf_all_data:
        for data in page_all_data:
            block = data['block']
            if block != None and (block.type == "Equation" or block.type == 'Figure' or block.type == 'Table'):
                continue
            tokens = data['tokens']
            for i in range(len(tokens)):
                token = tokens[i]
                font_size = token.font.split("-")[-1]
                font = token.font.rstrip("-" + font_size)
                if token.is_math == False and font in emath_font:
                    token.is_math = True
                    for j in range(i - 1, -1, -1):
                        font_j = tokens[j].font.rstrip("-" + tokens[j].font.split("-")[-1])
                        if tokens[j].is_math == True or font_j in normal_font:
                            break
                        else:
                            tokens[j].is_math = True
                            if font_j not in emath_font:
                                emath_font.append(font_j)
                    for j in range(i + 1, len(tokens)):
                        font_j = tokens[j].font.rstrip("-" + tokens[j].font.split("-")[-1])
                        if tokens[j].is_math == True or font_j in normal_font:
                            break
                        else:
                            tokens[j].is_math = True
                            if font_j not in emath_font:
                                emath_font.append(font_j)
    return emath_font          
#

def get_math_map(pdf_all_data):
    new_all_data = []
    
    for page in range(len(pdf_all_data)):
        page_all_data = pdf_all_data[page]
        new_page_data = []
        times = 0
        for block_data in page_all_data:
            block = block_data['block']
            if block != None and (block.type == "Equation" or block.type == 'Figure' or block.type == 'Table'):
                new_page_data.append({'block': block})
                continue
            new_block_data = {'block': block, 'labels': [], 'words': [], 'line_ids': [], 'bbox': [], 'font': []}
            token_list = []
            i = 0
            while i < (len(block_data['tokens'])):
                if block_data['tokens'][i].is_math == False:
                    if "(cid:" in block_data['tokens'][i].text:
                        block_data['tokens'][i].text = "EME_" + str(page) + "_" + str(times)
                        block_data['tokens'][i].is_math = True
                        times += 1
                    elif len(block_data['tokens'][i].text) == 1:
                        if ord(block_data['tokens'][i].text) in font_tool['cmap'].tables[1].cmap:
                            if font_tool['cmap'].tables[1].cmap[ord(block_data['tokens'][i].text)].find(".notdef") != -1:
                                block_data['tokens'][i].text = "EME_" + str(page) + "_" + str(times)
                                block_data['tokens'][i].is_math = True
                                times += 1
                        else:
                            block_data['tokens'][i].text = "EME_" + str(page) + "_" + str(times)
                            block_data['tokens'][i].is_math = True
                            times += 1
                    token_list.append(block_data['tokens'][i])
                    i += 1
                else:
                    j = i + 1
                    while j < len(block_data['tokens']) and block_data['tokens'][j].is_math and block_data['tokens'][j].line_id == block_data['tokens'][i].line_id:
                        j += 1
                    if j > i + 1:
                        math_token = union_lp_box([block_data['tokens'][k] for k in range(i, j)])
                        math_token.text = "EME_" + str(page) + "_" + str(times)
                        math_token.is_math = True
                        math_token.label = block_data['tokens'][i].label
                        math_token.line_id = block_data['tokens'][i].line_id
                        math_token.font = block_data['tokens'][i].font
                        times += 1
                        token_list.append(math_token)
                    else:
                        token = block_data['tokens'][i]
                        if "(cid:" in token.text:
                            token.text = "EME_" + str(page) + "_" + str(times)
                            times += 1
                        elif len(token.text) > 1:
                            token.text = "EME_" + str(page) + "_" + str(times)
                            times += 1
                        elif unicodedata.name(token.text, '0').find("LETTER") != -1:
                            token.text = "EME_" + str(page) + "_" + str(times)
                            times += 1
                        elif ord(token.text) not in font_tool['cmap'].tables[1].cmap:
                            token.text = "EME_" + str(page) + "_" + str(times)
                            times += 1
                        elif font_tool['cmap'].tables[1].cmap[ord(token.text)].find(".notdef") != -1:
                            token.text = "EME_" + str(page) + "_" + str(times)
                            times += 1
                        else:
                            token.is_math = False
                        token_list.append(token)
                    i = j
            new_block_data['words'] = [token.text for token in token_list]
            new_block_data['bbox'] = [token.coordinates for token in token_list]
            new_block_data['labels'] = [token.label for token in token_list]
            new_block_data['line_ids'] = [token.line_id for token in token_list]
            new_block_data['font'] = [token.font for token in token_list]
            new_block_data['tokens'] = token_list  # new added
            new_page_data.append(new_block_data)
        new_all_data.append(new_page_data)
    return new_all_data




# 漏缺：1 上标的考虑 2 block之间的拼接（一起翻译）3 有些行之间不应该由空格区分（例如在block内居中的行等,难判断）
def generating_text(pdf_all_data, translate_tool, output_dir):   # generate text & type & font & align
    json_data = []
    pdf_blocks = []
    for page_all_data in pdf_all_data:
        page_blocks = []
        for data in page_all_data:
            block = data['block']
            if block == None:
                continue

            if block.type == 'Text':
                preds = Counter(data['labels'])
                typ = ''
                if len(preds) > 1:
                    typ = preds.most_common()[0][0]
                elif len(preds) == 1:
                    typ = data['labels'][0]
                if typ == '':
                    block.type == 'None'
                else:
                    block.type = typ[0].upper() + typ[1:]
            # 以后只关注类别：None, Title, Section, Figure, Table, List, Paragraph, 等等等等
            elif block.type == 'Title':
                preds = Counter(data['labels'])
                if len(preds) > 1 and preds.most_common()[0][0] != 'title':
                    block.type = 'Section'
                elif len(preds) == 1 and data['labels'][0] != 'title':
                    block.type = 'Section'
            elif block.type == 'Figure' or block.type == 'Table' or block.type == 'Equation':
                block.text = block.translated_text = block.align = block.font_size = ''
                page_blocks.append({'block_id': block.id, 'type': block.type, 'text': "", 'translated_text': "", 'bbox': list(block.coordinates), 'font': "", 'align': ""})  # 这里改成字典形式
                continue
            # 处理字体   
            fonts = Counter([i.split('-')[-1] for i in data['font']])
            if len(fonts) > 1:
                font = fonts.most_common()[0][0]
            elif len(fonts) == 1:
                font = data['font'][0].split('-')[-1]
            else:
                font = ''
            
            # 开始处理文字和对齐方式
            line_id = -1
            begin_ids = []
            text = ''
            line_text = ''
            avg_indent = 0
            list_mark = ''
            list_indent = 0
            for i in range(len(data['labels'])):
                if data['line_ids'][i] != line_id:
                    line_id = data['line_ids'][i]
                    begin_ids.append(i)
                    avg_indent += data['bbox'][i][2]
            # begin_ids: 记录每行的起始id
            if len(begin_ids) > 0:
                avg_indent /= len(begin_ids)
            for i in range(len(begin_ids)):
                end_index = begin_ids[i + 1] if i < len(begin_ids) - 1 else len(data['line_ids'])
                line_text = " ".join(data['words'][begin_ids[i] : end_index])
                # align
                if block.type == 'Author':
                    text += line_text + '\n'
                elif block.type == 'List':
                    if i == 0:
                        list_indent = data['bbox'][begin_ids[0]][0]
                        text += line_text
                    else:
                        if abs(data['bbox'][begin_ids[i]][0] - list_indent) < 2:
                            text += "\n" + line_text
                        elif text.endswith('-') and len(text) > 1 and text[-2] != ' ':
                            text = text[:-1] + line_text
                        else:
                            text += ' ' + line_text
            
                else:
                    if text.endswith('-') and len(text) > 1 and text[-2] != ' ':
                        text = text[:-1] + line_text
                    else:
                        text += ' ' + line_text
                
            align = 'center' if block.type == 'Author' or block.type == 'Title' else 'normal'
            block.text = unicodedata.normalize("NFKC", text.strip())
            block.translated_text = translate(translate_tool, block.text)
            block.align = align
            block.font_size = font
            # translated_text = translate_tool.translate(block.text, target_lang = 'zh', source_lang = 'en')
            page_blocks.append({'block_id': block.id, 'type': block.type, 'text': block.text, 'translated_text': block.translated_text, 'bbox': list(block.coordinates), 'font': font, 'align': align})  # 这里改成字典形式
        pdf_blocks.append(page_blocks)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(pdf_blocks, f)
    return pdf_blocks


def fake_draw(pdf_all_data):
    need_high = 0
    pdfmetrics.registerFont(TTFont('simhei', 'fonts/simhei.ttf', subfontIndex = 0))
    for page_data in pdf_all_data:
        page_blocks = []
        page_high = 0
        for block_data in page_data:
            if block_data['block'] == None:
                for line_id, token_seq in itertools.groupby(block_data['tokens'], lambda ele : ele.line_id):
                    tokens = list(token_seq)
                    block = union_lp_box(tokens)
                    block.type = 'Tokens'
                    block.token_list = tokens
                    page_blocks.append(block)
            elif block_data['block'].type != 'None':
                page_blocks.append(block_data['block'])
        page_blocks.sort(key=lambda block : block.coordinates[1])

        for i in range(len(page_blocks)):
            block = page_blocks[i]
            new_y0 = block.coordinates[1]
            for j in range(i):
                upper_block = page_blocks[j]
                if max(upper_block.coordinates[0], block.coordinates[0]) < min(upper_block.coordinates[2], block.coordinates[2]):
                    # 有交集
                    offset = block.coordinates[1] - upper_block.coordinates[3]
                    if offset + upper_block.new_coordinates[3] > new_y0:
                        new_y0 = offset + upper_block.new_coordinates[3]
                else:
                    offset = block.coordinates[1] - upper_block.coordinates[1]
                    if offset + upper_block.new_coordinates[1] > new_y0:
                        new_y0 = offset + upper_block.new_coordinates[1]
            if block.type == 'Figure' or block.type == 'Table' or block.type =='Equation':
                block.new_coordinates = (block.coordinates[0], new_y0, block.coordinates[2], block.coordinates[3] + new_y0 - block.coordinates[1])
            elif block.type == 'Tokens':
                block.new_coordinates = (block.coordinates[0], new_y0, block.coordinates[2], block.coordinates[3] + new_y0 - block.coordinates[1])
                for token in block.token_list:
                    token.offset = new_y0 - block.coordinates[1]
            else:
                try:
                    font_size = int(block.font_size)
                except ValueError:
                    font_size = 7
                text = block.translated_text.replace("<", "&lt;").replace(">", "&gt;").replace("_ ", "_").replace(" _", "_").replace(" ", '&nbsp;').replace("\n", "<br/>")
                p = Paragraph(text, ParagraphStyle(name='Normal', fontName="simhei", fontSize=font_size, leading=1.2*font_size, textColor=red))
                need_w, need_h = p.wrap(block.coordinates[2] - block.coordinates[0], block.coordinates[3] - block.coordinates[1])

                block.new_coordinates = (block.coordinates[0], new_y0, block.coordinates[2], new_y0 + need_h + font_size * 2 + (block.coordinates[3] - block.coordinates[1]))
            if block.new_coordinates[3] > page_high:
                page_high = block.new_coordinates[3]
        if page_high > need_high:
            need_high = page_high
    return need_high



def draw_pdf(pdf_images, pdf_all_data, height, width, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, 'Figures/'))
        os.makedirs(os.path.join(output_dir, 'Tables/'))
        os.makedirs(os.path.join(output_dir, 'Equations/'))
        os.makedirs(os.path.join(output_dir, 'Expressions/'))
    
    pdf = Canvas(os.path.join(output_dir, 'test.pdf'), pagesize = (width, height))
    pdfmetrics.registerFont(TTFont('simhei', 'fonts/simhei.ttf', subfontIndex = 0))
    pdfmetrics.registerFont(TTFont('times', 'fonts/TIMES.ttf'))
    for page in range(len(pdf_all_data)):
        page_image = pdf_images[page]
        page_data = pdf_all_data[page]
        for block_data in page_data:
            if block_data['block'] != None and (block_data['block'].type == 'Figure' or block_data['block'].type == 'Table' or block_data['block'].type == 'Equation'):
                continue
            for token_index in range(len(block_data['labels'])):
                if block_data['block'] != None:
                    offset = block_data['block'].new_coordinates[1] - block_data['block'].coordinates[1]
                else:
                    offset = block_data['tokens'][token_index].offset
                token_coord = block_data['bbox'][token_index]
                if re.match("EME_[0-9]+_[0-9]+", block_data['words'][token_index]) != None:
                    img = page_image.crop(tuple([k * img_token_rate for k in token_coord]))
                    pdf.drawInlineImage(img, token_coord[0], height - token_coord[3] - offset - 1, token_coord[2] - token_coord[0], token_coord[3] - token_coord[1])
                    for i in range(img.size[0]):
                        for j in range(img.size[1]):
                            data = img.getpixel((i,j))
                            img.putpixel((i,j), (255, data[1], data[2]))
                    img.save(os.path.join(output_dir, 'Expressions/', block_data['words'][token_index].casefold() + '.jpg'), quality = 95)
            for token_index in range(len(block_data['labels'])):
                if block_data['block'] != None:
                    offset = block_data['block'].new_coordinates[1] - block_data['block'].coordinates[1]
                else:
                    offset = block_data['tokens'][token_index].offset
                token_coord = block_data['bbox'][token_index]
                if re.match("EME_[0-9]+_[0-9]+", block_data['words'][token_index]) == None:
                    pdf.setFont('times', int(block_data['font'][token_index].split("-")[-1]))
                    pdf.drawString(token_coord[0], height - token_coord[3] - offset, unicodedata.normalize("NFKC", block_data['words'][token_index]))

        for block_index in range(len(page_data)):
            # block = page_blocks[block_index]
            block = page_data[block_index]['block']
            if block == None:
                continue
            coord = block.coordinates
            if block.type == 'Figure' or block.type == 'Table' or block.type == 'Equation':
                img = page_image.crop(tuple([k * img_token_rate for k in coord]))
                img.save(os.path.join(output_dir, block.type + 's/', str(page) + '_' + str(block_index) + '.jpg'), quality = 95)
                pdf.drawImage(os.path.join(output_dir, block.type + 's/', str(page) + '_' + str(block_index) + '.jpg'), coord[0], height - block.new_coordinates[3], coord[2] - coord[0], coord[3] - coord[1])
                continue
            elif block.type != 'None':
                try:
                    font_size = int(block.font_size)
                except ValueError:
                    font_size = 7
                text = block.translated_text.replace("<", "&lt;").replace(">", "&gt;").replace("_ ", "_").replace(" _", "_").replace(" ", '&nbsp;').replace("\n", "<br/>")
                all_eme = re.findall("EME_[0-9]+_[0-9]+", text)
                for eme in reversed(sorted(list(set(all_eme)))):
                    img = Image.open(os.path.join(output_dir, "Expressions/", eme.casefold() + '.jpg'))
                    # img.size   width height
                    text = text.replace(eme, "<img src=\"" + os.path.join(output_dir, "Expressions/", eme.casefold() + '.jpg') + "\" width=\"" + str(int(math.floor(font_size) * img.size[0] / img.size[1])) + "\" height=\"" + str(math.floor(font_size)) + "\"/>")
                if text.startswith("<img src=\""):
                    text = "&nbsp;" + text
                p = Paragraph(text, ParagraphStyle(name='Normal', fontName="simhei", fontSize=font_size, leading=1.2*font_size, textColor=red))
                
                need_w, need_h = p.wrap(coord[2] - coord[0], coord[3] - coord[1])
                while need_h > coord[3] - coord[1]:
                    font_size -= 0.5
                    p = Paragraph(text, ParagraphStyle(name='Normal', fontName="simhei", fontSize=font_size, leading=1.2*font_size, textColor=red))
                    need_w, need_h = p.wrap(coord[2] - coord[0], coord[3] - coord[1])
                f = Frame(coord[0], height - block.new_coordinates[3] - 4, coord[2] - coord[0], block.new_coordinates[3] - block.new_coordinates[1] - coord[3] + coord[1], leftPadding=0, rightPadding=0, bottomPadding=0, topPadding=0, showBoundary=0)
                f.addFromList([p], pdf)

        pdf.showPage()
    pdf.save()
    return pdf

            


            

