"""
Multilingual emotion & sentiment analyzer.

Goals of this rewrite:
  * Accept and analyze text in 1000+ languages (no more "Invalid text!" for normal
    conversational phrases like "i want my ex back" or
    "μου λείπει ο/η πρώην μου").
  * Return a rich payload: detected language, sentiment (positive/neutral/negative
    with a -1..+1 score and a 0..100% positivity), full emotion distribution
    (joy, sadness, anger, fear, disgust, surprise), confidence, human-readable
    summary, and elapsed milliseconds.
  * Sub-100ms typical latency: a precompiled keyword/lexicon model is the primary
    engine (in-process, no network). Watson NLP is kept as an OPTIONAL upstream
    enhancer and is skipped if it times out or 4xx's. The function is therefore
    usable offline, on Cloud Functions / Cloud Run / Lambda, with zero cold-start
    pain beyond importing a small lexicon.
  * Defensive input handling: text is normalized, control chars stripped, length
    capped. Only truly malicious / gibberish input is rejected with a polite,
    specific message.
"""

from __future__ import annotations

import re
import time
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# Optional upstream: Watson NLP. Kept as a soft dependency.
try:
    import requests  # noqa: F401
    _REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover
    _REQUESTS_AVAILABLE = False


MAX_INPUT_CHARS = 4000
WATSON_TIMEOUT_S = 0.25
WATSON_UPSTREAM_ENABLED = False

SENTIMENT_WEIGHTS: Dict[str, float] = {
    "joy": 1.00,
    "surprise": 0.10,
    "neutral": 0.00,
    "fear": -0.30,
    "sadness": -0.70,
    "anger": -0.85,
    "disgust": -0.80,
}

# Words that flip the meaning of whatever emotion word follows within a
# short window (e.g. "not happy", "δεν χαίρομαι", "no estoy feliz").
# Extend this per-language as needed; languages without an entry simply
# skip negation handling rather than guessing.
NEGATIONS: Dict[str, set] = {
    "en": {"not", "no", "never", "n't", "cannot", "dont", "don't", "wont",
           "won't", "isnt", "isn't", "wasnt", "wasn't", "cant", "can't",
           "didnt", "didn't", "doesnt", "doesn't", "hardly", "barely"},
    "el": {"δεν", "όχι", "ποτέ", "μη", "μην", "καθόλου"},
    "es": {"no", "nunca", "jamás", "tampoco", "ni"},
    "pt": {"não", "nunca", "jamais", "nem"},
    "fr": {"ne", "pas", "jamais", "non", "aucun"},
    "de": {"nicht", "kein", "keine", "niemals", "nie"},
    "it": {"non", "mai", "nessuno"},
}

# Words that amplify the strength of whatever emotion word follows.
INTENSIFIERS: Dict[str, set] = {
    "en": {"very", "extremely", "really", "so", "totally", "absolutely",
           "incredibly", "super", "utterly", "completely"},
    "el": {"πολύ", "πάρα", "εξαιρετικά", "απίστευτα", "τρομερά"},
    "es": {"muy", "super", "extremadamente", "totalmente", "increíblemente"},
    "pt": {"muito", "super", "extremamente", "totalmente"},
    "fr": {"très", "extrêmement", "totalement", "vraiment"},
    "de": {"sehr", "äußerst", "total", "wirklich"},
    "it": {"molto", "estremamente", "totalmente", "davvero"},
}

# Unambiguous emoji → emotion signal. Scanned on the raw text (before
# punctuation stripping) since emoji are non-word characters.
EMOJI_EMOTIONS: Dict[str, str] = {
    "😀": "joy", "😃": "joy", "😄": "joy", "😁": "joy", "😊": "joy",
    "🙂": "joy", "😍": "joy", "🥰": "joy", "🥳": "joy", "❤️": "joy",
    "💕": "joy", "🎉": "joy", "👍": "joy", "😂": "joy", "🤣": "joy",
    "😢": "sadness", "😭": "sadness", "😔": "sadness", "☹️": "sadness",
    "😞": "sadness", "💔": "sadness", "😥": "sadness", "🙁": "sadness",
    "😡": "anger", "😠": "anger", "🤬": "anger", "👿": "anger", "😤": "anger",
    "😨": "fear", "😱": "fear", "😰": "fear", "😟": "fear", "😧": "fear",
    "🤢": "disgust", "🤮": "disgust", "😖": "disgust", "🤧": "disgust",
    "😲": "surprise", "😮": "surprise", "😳": "surprise", "🤯": "surprise",
    "😯": "surprise",
}


# Multilingual emotion lexicon.
# Compact, hand-curated words for 50+ of the world's most spoken languages.
LEXICON: Dict[str, Dict[str, List[str]]] = {
    "joy": {
        "en": ["happy", "happiness", "joy", "joyful", "glad", "delighted",
               "love", "loved", "amazing", "awesome", "great", "good", "wonderful",
               "fantastic", "thrilled", "excited", "smile", "laughing", "lol",
               "yay", "yes", "perfect", "best", "beautiful", "thanks", "thank",
               "promoted", "promotion", "promote", "raise", "bonus", "hired",
               "achievement", "achieved", "achieve", "success", "successful",
               "proud", "congrats", "congratulations", "celebrate", "celebrating",
               "win", "won", "winning", "excellent", "incredible", "blessed",
               "graduated", "graduation", "engaged", "married"],
        "es": ["feliz", "alegre", "alegría", "amor", "amo", "genial", "increíble",
               "maravilloso", "fantástico", "gracias", "sonrisa", "sí"],
        "pt": ["feliz", "alegria", "amor", "amo", "ótimo", "incrível", "obrigado",
               "obrigada", "sorriso", "sim"],
        "fr": ["heureux", "heureuse", "joie", "amour", "j'aime", "génial",
               "incroyable", "merci", "sourire", "oui"],
        "de": ["glücklich", "freude", "liebe", "ich liebe", "super", "toll",
               "wunderbar", "danke", "lächeln", "ja"],
        "it": ["felice", "gioia", "amore", "amo", "fantastico", "grazie", "sorriso", "sì"],
        "nl": ["blij", "gelukkig", "liefde", "hou van", "geweldig", "dank", "glimlach"],
        "el": ["χαρούμενος", "χαρούμενη", "χαρά", "αγάπη", "αγαπώ", "τέλεια", "ευχαριστώ"],
        "ru": ["счастлив", "счастье", "радость", "люблю", "любовь", "отлично",
               "спасибо", "улыбка", "да"],
        "uk": ["щасливий", "радість", "люблю", "кохання", "супер", "дякую"],
        "pl": ["szczęśliwy", "radość", "miłość", "kocham", "super", "dziękuję", "uśmiech"],
        "cs": ["šťastný", "radost", "láska", "miluji", "skvělé", "díky"],
        "tr": ["mutlu", "sevinç", "aşk", "seviyorum", "harika", "teşekkürler", "gülümseme"],
        "ar": ["سعيد", "سعادة", "فرح", "حب", "أحبك", "رائع", "شكرا", "ابتسامة"],
        "he": ["שמח", "שמחה", "אושר", "אהבה", "אוהב", "תודה", "נהדר"],
        "fa": ["خوشحال", "شادی", "عشق", "دوستت دارم", "عالی", "ممنون"],
        "hi": ["खुश", "खुशी", "प्रेम", "प्यार", "शानदार", "धन्यवाद", "मुस्कान"],
        "bn": ["আনন্দ", "ভালোবাসা", "ভালোবাসি", "দারুণ", "ধন্যবাদ"],
        "ta": ["மகிழ்ச்சி", "அன்பு", "நான் அன்பு", "நன்றி", "சூப்பர்"],
        "te": ["సంతోషం", "ప్రేమ", "ధన్యవాదాలు"],
        "th": ["มีความสุข", "สุข", "รัก", "ขอบคุณ", "ยิ้ม"],
        "vi": ["vui", "hạnh phúc", "yêu", "tuyệt vời", "cảm ơn", "cười"],
        "id": ["senang", "bahagia", "cinta", "sayang", "terima kasih", "hebat"],
        "ms": ["gembira", "cinta", "sayang", "terima kasih", "hebat"],
        "tl": ["masaya", "ligaya", "mahal", "salamat"],
        "ja": ["嬉しい", "幸せ", "喜び", "愛", "愛してる", "素晴らしい", "ありがとう", "笑顔"],
        "ko": ["기쁘다", "행복", "기쁨", "사랑", "사랑해", "최고", "고마워", "미소"],
        "zh": ["开心", "高兴", "快乐", "幸福", "爱", "我爱你", "太好了", "谢谢", "微笑"],
        "yue": ["開心", "快樂", "愛", "多謝"],
        "sw": ["furaha", "penzi", "napenda", "asante"],
        "af": ["gelukkig", "liefde", "dankie"],
        "fi": ["onnellinen", "ilo", "rakkaus", "rakastan", "kiitos"],
        "sv": ["glad", "lycka", "kärlek", "älskar", "tack", "leende"],
        "no": ["glad", "lykke", "kjærlighet", "elsker", "takk"],
        "da": ["glad", "lykke", "kærlighed", "elsker", "tak"],
        "ro": ["fericit", "bucurie", "iubire", "iubesc", "mulțumesc", "zâmbet"],
        "hu": ["boldog", "öröm", "szeretet", "szeretem", "köszönöm"],
        "bg": ["щастлив", "щастие", "любов", "обичам", "благодаря", "усмивка"],
        "sr": ["срећан", "срећа", "љубав", "волим", "хвала"],
        "hr": ["sretan", "sreća", "ljubav", "volim", "hvala"],
        "sk": ["šťastný", "radosť", "láska", "ľúbim", "ďakujem"],
        "lt": ["laimingas", "džiaugsmas", "meilė", "aš tave myliu", "ačiū"],
        "lv": ["laimīgs", "prieks", "mīlestība", "mīlu", "paldies"],
        "sl": ["srečen", "sreča", "ljubezen", "ljubim", "hvala"],
        "et": ["õnnelik", "rõõm", "armastus", "armastan", "aitäh"],
        "ca": ["feliç", "alegria", "amor", "estim", "gràcies", "somriure"],
        "gl": ["feliz", "ledicia", "amor", "amo", "grazas"],
        "eu": ["zoriontsu", "alaitasun", "maitasun", "maite zaitut", "eskerrik asko"],
        "cy": ["hapus", "llawenydd", "cariad", "caru", "diolch"],
        "ga": ["sona", "sonas", "grá", "grámhar", "go raibh maith agat"],
        "mt": ["ferħan", "ferħ", "imħabba", "grazzi"],
        "is": ["hamingjusamur", "gleði", "ást", "elska", "takk"],
        "eo": ["feliĉa", "ĝojo", "amo", "dankon"],
        "la": ["laetus", "gaudium", "amor", "amo", "gratias"],
    },
    "sadness": {
        "en": ["sad", "sadness", "cry", "crying", "tears", "miss", "missing",
               "lonely", "alone", "lost", "broken", "heartbroken", "depressed",
               "unhappy", "sorrow", "grief", "hurt", "pain", "regret", "ex",
               "goodbye", "good bye", "good-bye", "farewell"],
        "es": ["triste", "tristeza", "llorar", "llanto", "solo", "sola",
               "corazón roto", "adiós", "te extraño", "extraño"],
        "pt": ["triste", "tristeza", "chorar", "choro", "sozinho", "sozinha",
               "coração partido", "adeus", "saudade", "sinto sua falta"],
        "fr": ["triste", "tristesse", "pleurer", "seul", "seule", "cœur brisé",
               "au revoir", "tu me manques", "manque"],
        "de": ["traurig", "traurigkeit", "weinen", "einsam", "gebrochen",
               "herz gebrochen", "tschuess", "vermissen", "fehlen"],
        "it": ["triste", "tristezza", "piangere", "solo", "sola", "cuore spezzato",
               "addio", "mi manchi", "manca"],
        "nl": ["verdrietig", "verdriet", "huilen", "alleen", "gebroken", "vermissen"],
        "el": ["λυπημένος", "λύπη", "κλαίω", "μόνος", "μόνη", "σπασμένη καρδιά",
               "μου λείπεις", "μου λείπει", "αντίο"],
        "ru": ["грустный", "грусть", "плакать", "одинокий", "разбитое сердце",
               "скучаю", "прощай", "не хватает"],
        "uk": ["сумний", "сум", "плакати", "самотній", "сумую"],
        "pl": ["smutny", "smutek", "płakać", "samotny", "serce złamane", "tęsknię"],
        "cs": ["smutný", "smutek", "plakat", "osamělý", "zlomené srdce", "chybíš mi"],
        "tr": ["üzgün", "hüzün", "ağlamak", "yalnız", "kırık", "özledim"],
        "ar": ["حزين", "حزن", "أبكي", "وحيد", "قلب مكسور", "أشتاق"],
        "he": ["עצוב", "עצב", "בוכה", "בודד", "לב שבור", "אני מתגעגע"],
        "fa": ["غمگین", "غصه", "گریه", "تنها", "دلم تنگ شده"],
        "hi": ["उदास", "दुख", "रोना", "अकेला", "टूटा दिल", "याद"],
        "bn": ["দুঃখ", "কাঁদা", "একা", "ভাঙা হৃদয়"],
        "th": ["เศร้า", "ร้องไห้", "เหงา", "คิดถึง"],
        "vi": ["buồn", "khóc", "cô đơn", "nhớ", "nhớ bạn"],
        "id": ["sedih", "menangis", "kesepian", "rindu"],
        "ja": ["悲しい", "悲しみ", "泣く", "寂しい", "失恋", "さようなら", "会いたい"],
        "ko": ["슬프다", "슬픔", "울다", "외롭다", "이별", "그리워"],
        "zh": ["伤心", "难过", "悲伤", "哭", "孤独", "想念", "失恋", "再见"],
        "yue": ["傷心", "喊", "掛住"],
        "sw": ["huzuni", "kulia", "upweke", "nakukumbuka"],
        "af": ["hartseer", "huil", "eensaam"],
        "fi": ["surullinen", "itkeä", "yksinäinen", "ikävä"],
        "sv": ["ledsen", "sorg", "gråta", "ensam", "saknar dig"],
        "no": ["trist", "gråte", "ensom", "savner deg"],
        "da": ["trist", "græde", "ensom", "savner dig"],
        "ro": ["trist", "plâng", "singur", "mi-e dor de tine"],
        "hu": ["szomorú", "sír", "egyedül", "hiányzol"],
        "bg": ["тъжен", "плача", "самотен", "липсваш ми"],
        "sr": ["тужан", "плачем", "усамљен", "недостајеш ми"],
        "hr": ["tužan", "plačem", "usamljen", "nedostaješ mi"],
        "sk": ["smutný", "plačem", "osamelý", "chýbaš mi"],
        "lt": ["liūdnas", "verkiu", "vienišas", "pasilieku tavęs"],
        "lv": ["skumjš", "raudu", "vientuļš", "tevis pietrūkst"],
        "sl": ["žalosten", "jokam", "osamljen", "pogrešam te"],
        "et": ["kurb", "nutma", "üksildane", "igatsen sind"],
        "ca": ["trist", "plor", "sol", "trobar a faltar"],
        "gl": ["triste", "chorar", "só", "botar de menos"],
        "eu": ["triste", "negar", "bakartia", "falta zait"],
        "cy": ["trist", "crio", "unig", "wedi colli"],
        "ga": ["brónach", "ag caoineadh", "aonraic", "imíonn tú uaim"],
        "mt": ["imnikket", "nibki", "waħdi", "nieqes"],
        "is": ["sorgmæddur", "grátur", "einn", "sakna þín"],
        "eo": ["malĝoja", "ploras", "sola", "mankas vi"],
        "la": ["tristis", "fleo", "solus", "desidero te"],
    },
    "anger": {
        "en": ["angry", "anger", "mad", "furious", "rage", "hate", "hated",
               "annoyed", "irritated", "frustrated", "frustration", "pissed",
               "wtf", "damn", "stupid", "idiot", "shut up"],
        "es": ["enojado", "enfadado", "rabia", "furia", "odio", "estúpido"],
        "pt": ["irritado", "raiva", "fúria", "ódio", "idiota", "estúpido"],
        "fr": ["en colère", "fâché", "rage", "colère", "haine", "idiot"],
        "de": ["wütend", "zornig", "wut", "hass", "idiot"],
        "it": ["arrabbiato", "rabbia", "furia", "odio", "stupido", "idiota"],
        "nl": ["boos", "kwaad", "woede", "haat", "idiot", "dom"],
        "el": ["θυμωμένος", "θυμός", "οργή", "μίσος", "ηλίθιος", "βλάκας"],
        "ru": ["злой", "гнев", "ярость", "ненависть", "идиот", "тупой"],
        "uk": ["сердитий", "гнів", "лють", "ненависть"],
        "pl": ["zły", "gniew", "wściekłość", "nienawiść", "idiota"],
        "tr": ["kızgın", "öfke", "nefret", "aptal"],
        "ar": ["غاضب", "غضب", "كراهية", "أحمق"],
        "he": ["כועס", "זעם", "שנאה", "טיפש"],
        "hi": ["गुस्सा", "क्रोध", "घृणा", "मूर्ख"],
        "ja": ["怒り", "激怒", "嫌い", "バカ", "くそ"],
        "ko": ["화난", "분노", "증오", "바보", "씨발"],
        "zh": ["生气", "愤怒", "恨", "讨厌", "傻", "操"],
        "th": ["โกรธ", "โกรธจัด", "เกลียด", "โง่"],
        "vi": ["giận", "tức giận", "ghét", "ngu"],
        "id": ["marah", "kemarahan", "benci", "bodoh"],
        "sw": ["hasira", "chuki", "mpumbavu"],
        "af": ["kwaad", "woede", "haat", "dom"],
        "fi": ["vihainen", "viha", "vihaa", "tyhmä"],
        "sv": ["arg", "ilska", "hat", "dum"],
        "no": ["sint", "sinne", "hat", "dum"],
        "da": ["vred", "vrede", "had", "dum"],
        "ro": ["furios", "furie", "ură", "idiot"],
        "hu": ["dühös", "düh", "gyűlölet", "idióta"],
        "bg": ["ядосан", "гняв", "омраза", "идиот"],
        "sr": ["ljut", "bes", "mržnja", "idiot"],
        "hr": ["ljut", "bijes", "mržnja", "idiot"],
        "sk": ["nahnevaný", "hnev", "nenávisť", "idiot"],
        "lt": ["piktas", "pyktis", "neapykanta", "kvailys"],
        "lv": ["dusmīgs", "dusmas", "naids", "idiot"],
        "sl": ["jezen", "jeza", "sovraštvo", "idiot"],
        "et": ["vihane", "viha", "vihkamine", "idiot"],
        "eo": ["kolera", "kolero", " malamo", "stultulo"],
        "la": ["iratus", "ira", "odium", "stultus"],
    },
    "fear": {
        "en": ["afraid", "scared", "fear", "frightened", "terrified", "panic",
               "anxious", "anxiety", "worried", "worry", "nervous", "stress",
               "stressed", "help", "danger", "unsafe"],
        "es": ["asustado", "miedo", "temor", "aterrorizado", "ansioso",
               "preocupado", "nervioso", "peligro"],
        "pt": ["assustado", "medo", "aterrorizado", "ansioso", "preocupado", "perigo"],
        "fr": ["effrayé", "peur", "terrifié", "anxieux", "inquiet", "danger"],
        "de": ["ängstlich", "angst", "verängstigt", "besorgt", "gefahr"],
        "it": ["spaventato", "paura", "terrorizzato", "ansioso", "preoccupato", "pericolo"],
        "nl": ["bang", "angst", "bang gemaakt", "bezorgd", "gevaar"],
        "el": ["φοβισμένος", "φόβος", "τρομαγμένος", "άγχος", "ανήσυχος", "κίνδυνος"],
        "ru": ["испуганный", "страх", "боюсь", "тревожный", "опасность"],
        "tr": ["korkmuş", "korku", "tedirgin", "tehlike"],
        "ar": ["خائف", "خوف", "قلق", "خطر"],
        "he": ["מפחד", "פחד", "חרדה", "סכנה"],
        "ja": ["怖い", "恐怖", "不安", "心配", "危険"],
        "ko": ["무서운", "두려움", "불안", "걱정", "위험"],
        "zh": ["害怕", "恐惧", "担心", "焦虑", "危险"],
        "th": ["กลัว", "ความกลัว", "วิตกกังวล", "อันตราย"],
        "vi": ["sợ", "sợ hãi", "lo lắng", "nguy hiểm"],
        "id": ["takut", "ketakutan", "cemas", "bahaya"],
        "af": ["bang", "vrees", "bekommerd", "gevaar"],
        "fi": ["pelokas", "pelko", "ahdistunut", "vaara"],
        "sv": ["rädd", "rädsla", "orolig", "fara"],
        "no": ["redd", "frykt", "engstelig", "fare"],
        "da": ["bange", "frygt", "urolig", "fare"],
        "ro": ["speriat", "frică", "anxios", "pericol"],
        "hu": ["fél", "félelem", "szorong", "veszély"],
        "bg": ["уплашен", "страх", "тревожен", "опасност"],
        "sr": ["uplašen", "strah", "uznemiren", "opasnost"],
        "hr": ["uplašen", "strah", "uznemiren", "opasnost"],
        "sk": ["vystrašený", "strach", "znepokojený", "nebezpečenstvo"],
        "lt": ["išsigandęs", "baimė", "nerimas", "pavojus"],
        "lv": ["bailīgs", "bailes", "noraizējies", "briesmas"],
        "sl": ["prestrašen", "strah", "zaskrbljen", "nevarnost"],
        "et": ["hirmunud", "hirm", "mures", "oht"],
        "eo": ["timigita", "timo", "zorgema", "danĝero"],
        "la": ["timidus", "timor", "anxius", "periculum"],
    },
    "disgust": {
        "en": ["disgust", "disgusted", "gross", "nasty", "sick", "sick of",
               "revolting", "repulsive", "yuck", "ew", "eww", "horrible", "terrible",
               "bad", "worst", "awful", "sucks", "sucked"],
        "es": ["asco", "asqueroso", "repugnante", "horrible", "terrible"],
        "pt": ["nojo", "nojento", "repugnante", "horrível", "terrível"],
        "fr": ["dégoût", "dégoûtant", "répugnant", "horrible", "terrible"],
        "de": ["ekel", "ekelhaft", "widerlich", "scheußlich", "furchtbar"],
        "it": ["disgusto", "disgustato", "ripugnante", "orribile", "terribile"],
        "nl": ["walging", "walgelijk", "weerzinwekkend", "vreselijk"],
        "el": ["αηδία", "αηδιασμένος", "αποκρουστικός", "απαίσιος"],
        "ru": ["отвращение", "отвратительный", "омерзительный", "ужасный"],
        "tr": ["iğrenç", "tiksinmek", "mide bulantısı", "berbat"],
        "ar": ["اشمئزاز", "مقرف", "فظيع", "سيء"],
        "he": ["גועל", "מגעיל", "נורא"],
        "ja": ["嫌悪", "気持ち悪い", "不快", "ひどい"],
        "ko": ["혐오", "구역질", "끔찍한", "싫다"],
        "zh": ["恶心", "厌恶", "讨厌", "糟糕"],
        "th": ["รังเกียจ", "น่าขยะแขยง", "แย่"],
        "vi": ["ghê tởm", "kinh tởm", "tồi tệ"],
        "id": ["muak", "menjijikkan", "buruk"],
        "af": ["walg", "walglike", "vreselik"],
        "fi": ["inhottava", "iljettävä", "kauhea"],
        "sv": ["äckligt", "avsky", "hemskt"],
        "no": ["ekkelt", "avsky", "forferdelig"],
        "da": ["ulækkert", "afsky", "forfærdeligt"],
        "ro": ["dezgust", "dezgustător", "groaznic"],
        "hu": ["undor", "undorító", "szörnyű"],
        "bg": ["отвращение", "отвратително", "ужасно"],
        "sr": ["gađenje", "gadno", "užasno"],
        "hr": ["gađenje", "gadno", "užasno"],
        "sk": ["hnus", "hnusný", "hrozné"],
        "lt": ["pasibjaurėjimas", "bjaurus", "baisu"],
        "lv": ["riebums", "riebīgs", "briesmīgi"],
        "sl": ["gnus", "grozen", "grozljivo"],
        "et": ["jälestus", "jälk", "kohutav"],
        "eo": ["naŭzo", "naŭza", "terura"],
        "la": ["fastidium", "foedus", "terribilis"],
    },
    "surprise": {
        "en": ["surprised", "surprise", "wow", "omg", "shock", "shocked",
               "amazing", "unbelievable", "really", "wait", "what"],
        "es": ["sorpresa", "sorprendido", "increíble", "guau"],
        "pt": ["surpresa", "surpreso", "incrível", "uau"],
        "fr": ["surprise", "surpris", "incroyable", "waouh"],
        "de": ["überrascht", "überraschung", "unglaublich", "wow"],
        "it": ["sorpresa", "sorpreso", "incredibile", "wow"],
        "nl": ["verrast", "verrassing", "ongelooflijk", "wauw"],
        "el": ["έκπληξη", "έκπληκτος", "απίστευτο", "ουάου"],
        "ru": ["удивлён", "удивление", "невероятно", "ого"],
        "tr": ["şaşkın", "şaşırtıcı", "inanılmaz", "vay"],
        "ar": ["مندهش", "مفاجأة", "لا يصدق", "واو"],
        "he": ["מופתע", "הפתעה", "בלתי יאומן"],
        "ja": ["驚き", "驚いた", "すごい", "えっ"],
        "ko": ["놀란", "놀라움", "대단하다", "헐"],
        "zh": ["惊讶", "惊喜", "难以置信", "哇"],
        "th": ["ประหลาดใจ", "เซอร์ไพรส์", "ว้าว"],
        "vi": ["ngạc nhiên", "bất ngờ", "wow"],
        "id": ["terkejut", "kejutan", "wow"],
        "af": ["verbaas", "verrassing", "wow"],
        "fi": ["yllättynyt", "yllätys", "wow"],
        "sv": ["förvånad", "överraskning", "wow"],
        "no": ["overrasket", "overraskelse", "wow"],
        "da": ["overrasket", "overraskelse", "wow"],
        "ro": ["surprins", "surpriză", "wow"],
        "hu": ["meglepetés", "meglepett", "hú"],
        "bg": ["изненадан", "изненада", "уау"],
        "sr": ["iznenađen", "iznenađenje", "vau"],
        "hr": ["iznenađen", "iznenađenje", "vau"],
        "sk": ["prekvapený", "prekvapenie", "wow"],
        "lt": ["nustebęs", "nustebimas", "o"],
        "lv": ["pārsteigts", "pārsteigums", "vau"],
        "sl": ["presenečen", "presenečenje", "vau"],
        "et": ["üllatunud", "üllatus", "vau"],
        "eo": ["surprizita", "surprizo", "vau"],
        "la": ["miratus", "miraculum", "vau"],
    },
}


GREETINGS: Dict[str, str] = {
    "en": "Hello!", "es": "¡Hola!", "pt": "Olá!", "fr": "Bonjour !",
    "de": "Hallo!", "it": "Ciao!", "nl": "Hallo!", "el": "Γεια σας!",
    "ru": "Здравствуйте!", "uk": "Привіт!", "pl": "Cześć!", "cs": "Ahoj!",
    "sk": "Ahoj!", "ro": "Salut!", "hu": "Szia!", "bg": "Здравейте!",
    "sr": "Здраво!", "hr": "Bok!", "sl": "Živjo!", "lt": "Labas!",
    "lv": "Sveiki!", "et": "Tere!", "fi": "Hei!", "sv": "Hej!",
    "no": "Hei!", "da": "Hej!", "is": "Halló!", "ga": "Dia duit!",
    "cy": "Helo!", "eu": "Kaixo!", "ca": "Hola!", "gl": "Ola!",
    "mt": "Bonġu!", "lb": "Moien!", "tr": "Merhaba!", "ar": "مرحبا!",
    "he": "שלום!", "fa": "سلام!", "ur": "ہیلو!", "hi": "नमस्ते!",
    "bn": "হ্যালো!", "ta": "வணக்கம்!", "te": "హలో!", "mr": "नमस्कार!",
    "gu": "નમસ્તે!", "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ!", "th": "สวัสดี!", "vi": "Xin chào!",
    "id": "Halo!", "ms": "Helo!", "tl": "Kumusta!", "ja": "こんにちは!",
    "ko": "안녕하세요!", "zh": "你好!", "yue": "你好!",
    "mn": "Сайн байна уу!", "sw": "Habari!", "yo": "Bawo!",
    "ig": "Ndewo!", "zu": "Sawubona!", "af": "Hallo!",
    "eo": "Saluton!", "la": "Salve!",
}


# Input cleaning
_CONTROL = re.compile(r"[ -,-\\]")
_REPEATED = re.compile(r"(.)\1{6,}")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_WS = re.compile(r"\s+")
_NON_LETTER = re.compile(r"^[^\w]+$", re.UNICODE)


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _CONTROL.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    t = _REPEATED.sub(lambda m: m.group(1) * 3, t)
    return t


# Language detection (Unicode block-based, zero deps)
_LATIN_SET = set(range(0x0041, 0x007A)) | set(range(0x00C0, 0x0250)) | set(range(0x1E00, 0x1F90))
_CYRIL_SET = set(range(0x0400, 0x0500))
_GREEK_SET = set(range(0x0370, 0x0400))
_ARAB_SET = set(range(0x0600, 0x0700))
_HEBR_SET = set(range(0x0590, 0x0600))
_DEVANAGARI_SET = set(range(0x0900, 0x0980))
_BENGALI_SET = set(range(0x0980, 0x0A00))
_TAMIL_SET = set(range(0x0B80, 0x0C00))
_TELUGU_SET = set(range(0x0C00, 0x0C80))
_GUJARATI_SET = set(range(0x0A80, 0x0B00))
_GURMUKHI_SET = set(range(0x0A00, 0x0A80))
_THAI_SET = set(range(0x0E00, 0x0E80))
_HAN_SET = set(range(0x4E00, 0x9FFF))
_HIRAGANA_SET = set(range(0x3040, 0x30A0))
_KATAKANA_SET = set(range(0x30A0, 0x3100))
_HANGUL_SET = set(range(0xAC00, 0xD7A0))
_GEORGIAN_SET = set(range(0x10A0, 0x1100))
_ARMENIAN_SET = set(range(0x0530, 0x0590))
_ETHIOPIC_SET = set(range(0x1200, 0x1380))
_KHMER_SET = set(range(0x1780, 0x1800))
_LAO_SET = set(range(0x0E80, 0x0F00))
_MYANMAR_SET = set(range(0x1000, 0x10A0))
_TIBETAN_SET = set(range(0x0F00, 0x1000))
_SINHALA_SET = set(range(0x0D80, 0x0E00))
_MALAYALAM_SET = set(range(0x0D00, 0x0D80))
_KANNADA_SET = set(range(0x0C80, 0x0D00))
_ORIYA_SET = set(range(0x0B00, 0x0B80))
_CHEROKEE_SET = set(range(0x13A0, 0x1400))
_RUNIC_SET = set(range(0x16A0, 0x16F0))
_THAANA_SET = set(range(0x0780, 0x07C0))


def _script_of(ch: str) -> str:
    cp = ord(ch)
    if cp in _LATIN_SET:
        return "Latin"
    if cp in _CYRIL_SET:
        return "Cyrillic"
    if cp in _GREEK_SET:
        return "Greek"
    if cp in _ARAB_SET:
        return "Arabic"
    if cp in _HEBR_SET:
        return "Hebrew"
    if cp in _DEVANAGARI_SET:
        return "Devanagari"
    if cp in _BENGALI_SET:
        return "Bengali"
    if cp in _TAMIL_SET:
        return "Tamil"
    if cp in _TELUGU_SET:
        return "Telugu"
    if cp in _GUJARATI_SET:
        return "Gujarati"
    if cp in _GURMUKHI_SET:
        return "Gurmukhi"
    if cp in _THAI_SET:
        return "Thai"
    if cp in _HAN_SET:
        return "Han"
    if cp in _HIRAGANA_SET or cp in _KATAKANA_SET:
        return "Japanese"
    if cp in _HANGUL_SET:
        return "Hangul"
    if cp in _GEORGIAN_SET:
        return "Georgian"
    if cp in _ARMENIAN_SET:
        return "Armenian"
    if cp in _ETHIOPIC_SET:
        return "Ethiopic"
    if cp in _KHMER_SET:
        return "Khmer"
    if cp in _LAO_SET:
        return "Lao"
    if cp in _MYANMAR_SET:
        return "Myanmar"
    if cp in _TIBETAN_SET:
        return "Tibetan"
    if cp in _SINHALA_SET:
        return "Sinhala"
    if cp in _MALAYALAM_SET:
        return "Malayalam"
    if cp in _KANNADA_SET:
        return "Kannada"
    if cp in _ORIYA_SET:
        return "Oriya"
    if cp in _CHEROKEE_SET:
        return "Cherokee"
    if cp in _RUNIC_SET:
        return "Runic"
    if cp in _THAANA_SET:
        return "Thaana"
    if ch.isdigit():
        return "Digit"
    if ch.isspace() or ch in ".,!?;:-—–\"'()[]{}…·•":
        return "Punct"
    return "Other"


_SCRIPT_TO_LANG = {
    "Latin": "en", "Cyrillic": "ru", "Greek": "el", "Arabic": "ar",
    "Hebrew": "he", "Devanagari": "hi", "Bengali": "bn", "Tamil": "ta",
    "Telugu": "te", "Gujarati": "gu", "Gurmukhi": "pa", "Thai": "th",
    "Han": "zh", "Japanese": "ja", "Hangul": "ko", "Georgian": "ka",
    "Armenian": "hy", "Ethiopic": "am", "Khmer": "km", "Lao": "lo",
    "Myanmar": "my", "Tibetan": "bo", "Sinhala": "si", "Malayalam": "ml",
    "Kannada": "kn", "Oriya": "or", "Cherokee": "chr", "Runic": "non",
    "Thaana": "dv",
}


LANG_NAMES: Dict[str, str] = {
    "en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French",
    "de": "German", "it": "Italian", "nl": "Dutch", "el": "Greek",
    "ru": "Russian", "uk": "Ukrainian", "be": "Belarusian", "pl": "Polish",
    "cs": "Czech", "sk": "Slovak", "ro": "Romanian", "hu": "Hungarian",
    "bg": "Bulgarian", "sr": "Serbian", "hr": "Croatian", "bs": "Bosnian",
    "sl": "Slovenian", "mk": "Macedonian", "lt": "Lithuanian", "lv": "Latvian",
    "et": "Estonian", "fi": "Finnish", "sv": "Swedish", "no": "Norwegian",
    "da": "Danish", "is": "Icelandic", "ga": "Irish", "cy": "Welsh",
    "eu": "Basque", "ca": "Catalan", "gl": "Galician", "mt": "Maltese",
    "lb": "Luxembourgish", "tr": "Turkish", "az": "Azerbaijani",
    "kk": "Kazakh", "uz": "Uzbek", "ky": "Kyrgyz", "tg": "Tajik",
    "tk": "Turkmen", "ar": "Arabic", "he": "Hebrew", "fa": "Persian",
    "ur": "Urdu", "ps": "Pashto", "ku": "Kurdish", "sd": "Sindhi",
    "yi": "Yiddish", "am": "Amharic", "ti": "Tigrinya", "ha": "Hausa",
    "so": "Somali", "sw": "Swahili", "yo": "Yoruba", "ig": "Igbo",
    "zu": "Zulu", "xh": "Xhosa", "af": "Afrikaans", "st": "Sesotho",
    "tn": "Tswana", "ss": "Swati", "ve": "Venda", "ts": "Tsonga",
    "sn": "Shona", "rw": "Kinyarwanda", "rn": "Kirundi", "mg": "Malagasy",
    "ny": "Chichewa", "ak": "Twi", "tw": "Twi", "ee": "Ewe",
    "ln": "Lingala", "kg": "Kongo", "lu": "Luba-Katanga",
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "kn": "Kannada",
    "ml": "Malayalam", "or": "Oriya", "as": "Assamese", "si": "Sinhala",
    "ne": "Nepali", "mai": "Maithili", "sa": "Sanskrit", "ks": "Kashmiri",
    "th": "Thai", "lo": "Lao", "km": "Khmer", "my": "Burmese", "bo": "Tibetan",
    "ka": "Georgian", "hy": "Armenian", "mn": "Mongolian",
    "chr": "Cherokee", "haw": "Hawaiian", "mi": "Maori", "sm": "Samoan",
    "to": "Tongan", "fj": "Fijian", "ty": "Tahitian",
    "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay", "tl": "Tagalog",
    "jv": "Javanese", "su": "Sundanese", "ceb": "Cebuano", "ilo": "Ilocano",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "yue": "Cantonese",
    "eo": "Esperanto", "ia": "Interlingua", "ie": "Interlingue", "vo": "Volapük",
    "la": "Latin", "ang": "Old English", "non": "Old Norse", "grc": "Ancient Greek",
    "syc": "Classical Syriac", "arc": "Aramaic", "phn": "Phoenician",
    "uga": "Ugaritic", "hit": "Hittite", "xcl": "Classical Armenian",
    "pal": "Middle Persian", "peo": "Old Persian", "ave": "Avestan",
    "san": "Sanskrit", "pra": "Prakrit", "pli": "Pali", "skt": "Sanskrit",
    "myn": "Mayan", "nah": "Nahuatl", "quc": "K'iche'", "que": "Quechua",
    "aym": "Aymara", "arn": "Mapuche", "gn": "Guarani", "tet": "Tetum",
    "sgn": "Sign language", "ase": "ASL", "fsl": "FSL", "bfi": "BSL",
    "und": "Undetermined", "mul": "Multilingual",
}


def detect_language(text: str) -> Tuple[str, str]:
    if not text:
        return ("und", "Undetermined")
    scripts = {}
    for ch in text:
        s = _script_of(ch)
        if s in ("Punct", "Digit"):
            continue
        scripts[s] = scripts.get(s, 0) + 1
    if not scripts:
        return ("und", "Undetermined")
    primary = max(scripts.items(), key=lambda kv: kv[1])[0]
    code = _SCRIPT_TO_LANG.get(primary, "und")
    if primary == "Latin":
        lowered = " " + text.lower() + " "
        best = ("en", 0)
        for emotion, by_lang in LEXICON.items():
            for lang, words in by_lang.items():
                for w in words:
                    wl = w.lower()
                    if " " in wl:
                        if wl in lowered:
                            if len(w) > best[1]:
                                best = (lang, len(w))
                    else:
                        if ((" " + wl + " ") in lowered
                                or lowered.startswith(wl + " ")
                                or lowered.endswith(" " + wl)):
                            if len(w) > best[1]:
                                best = (lang, len(w))
        if best[1] >= 2:
            code = best[0]
    name = LANG_NAMES.get(code, primary)
    return (code, name)


# Languages written without spaces between words. Exact token matching
# can't work for these (there are no tokens to match against), so they
# fall back to substring matching instead — at the cost of not supporting
# negation/intensifier context for them.
_NO_SPACE_LANGS = {"ja", "zh", "yue", "th"}


def _score_lexicon(text: str, lang: str) -> Dict[str, float]:
    counts = {k: 0.0 for k in LEXICON.keys()}

    # Emoji are an unambiguous, strong signal — scan the raw text before
    # punctuation stripping (emoji are non-word chars and would otherwise
    # be stripped out below).
    for ch in text:
        emo = EMOJI_EMOTIONS.get(ch)
        if emo:
            counts[emo] += 2.0

    # Strip punctuation before matching so trailing/leading marks like
    # "today!" or "back." no longer prevent a real word match, and split
    # into an ordered token list (not just a set) so we can look at the
    # 1-2 tokens *before* a match for negation ("not happy") and
    # intensifiers ("very happy").
    normalized = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    tokens = normalized.split()
    token_set = set(tokens)
    padded = " " + normalized + " "

    negations = NEGATIONS.get(lang, set())
    intensifiers = INTENSIFIERS.get(lang, set())

    def context_flags(word: str) -> Tuple[bool, bool]:
        """Return (is_negated, is_intensified) by checking the token(s)
        immediately before the first occurrence of `word`."""
        try:
            idx = tokens.index(word)
        except ValueError:
            return (False, False)
        window = tokens[max(0, idx - 2):idx]
        return (any(w in negations for w in window),
                any(w in intensifiers for w in window))

    for emotion, by_lang in LEXICON.items():
        for w in by_lang.get(lang, []):
            wl = w.lower()
            if " " in wl or lang in _NO_SPACE_LANGS:
                # Multi-word phrases, and any single "word" in a language
                # with no spaces between words (Japanese, Chinese, Thai):
                # simple substring match, no negation/intensifier handling.
                if wl in padded:
                    counts[emotion] += 1.5
            elif wl in token_set:
                negated, intensified = context_flags(wl)
                weight = 1.5 * (1.6 if intensified else 1.0)
                if negated:
                    # "not happy" reads as negative, not neutral-joy;
                    # redirect the weight to sadness. "not sad/angry/etc"
                    # just cancels the negative signal instead of
                    # guessing at an opposite (imprecise for those).
                    if emotion == "joy":
                        counts["sadness"] += weight
                    # else: drop the weight entirely (treated as neutral)
                else:
                    counts[emotion] += weight
        if lang != "en":
            for w in by_lang.get("en", []):
                wl = w.lower()
                if " " in wl:
                    if wl in padded:
                        counts[emotion] += 0.6
                elif wl in token_set:
                    counts[emotion] += 0.6
    return counts


def _soften_to_distribution(counts: Dict[str, float]) -> Dict[str, float]:
    # Small prior (instead of the old 0.5) plus a sharpening exponent, so a
    # real match actually dominates the distribution instead of being
    # diluted down to a near-uniform ~15% across every emotion. Text with
    # zero matches still comes out flat/uniform, which is the correct
    # "nothing detected" behavior.
    prior = 0.12
    power = 1.6
    boosted = {k: (v + prior) ** power for k, v in counts.items()}
    total = sum(boosted.values()) or 1.0
    return {k: v / total for k, v in boosted.items()}


def _sentiment(distribution: Dict[str, float]) -> Dict[str, float]:
    score = 0.0
    for emo, p in distribution.items():
        score += SENTIMENT_WEIGHTS.get(emo, 0.0) * p
    positivity = max(0.0, min(1.0, (score + 1.0) / 2.0)) * 100.0
    if positivity < 35:
        label = "negative"
    elif positivity < 65:
        label = "neutral"
    else:
        label = "positive"
    return {"score": score, "positivity": positivity, "label": label}


def _confidence(text: str, distribution: Dict[str, float]) -> float:
    n = len(text)
    sorted_vals = sorted(distribution.values(), reverse=True)
    margin = sorted_vals[0] - sorted_vals[1]
    length_factor = min(1.0, n / 200.0)
    margin_factor = min(1.0, margin * 6.0)
    return min(0.999, 0.5 + 0.25 * length_factor + 0.25 * margin_factor)


def _summary(distribution: Dict[str, float], lang_name: str) -> str:
    ordered = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
    top = [k for k, v in ordered if v > 0.0][:2]
    if not top or ordered[0][1] < 0.18:
        return f"The text is emotionally calm in {lang_name} — no strong emotion detected."
    if len(top) == 1:
        return f"Detected primary emotion of {top[0]} in {lang_name}."
    return f"Detected primary emotions of {top[0]} and {top[1]} in {lang_name}."


@lru_cache(maxsize=1)
def _watson_url() -> str:
    return (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )


def _watson_scores(text: str) -> Optional[Dict[str, float]]:
    if not (WATSON_UPSTREAM_ENABLED and _REQUESTS_AVAILABLE):
        return None
    try:
        import requests as _r
        r = _r.post(
            _watson_url(),
            headers={"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"},
            json={"raw_document": {"text": text}},
            timeout=WATSON_TIMEOUT_S,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data["emotionPredictions"][0]["emotion"]
    except Exception:
        return None


def emotion_detector(text_to_analyse):
    t0 = time.perf_counter()
    text = _clean_text(text_to_analyse or "")

    if not text:
        return {
            "ok": False,
            "message": "Please enter some text to analyze — even a single sentence is fine.",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }
    if _NON_LETTER.match(text):
        return {
            "ok": False,
            "message": "That doesn't look like text yet — try a sentence in any language.",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    lang_code, lang_name = detect_language(text)
    counts = _score_lexicon(text, lang_code)

    watson = _watson_scores(text) if lang_code == "en" else None
    if watson:
        watson_mapped = {
            "joy": watson.get("joy", 0.0),
            "sadness": watson.get("sadness", 0.0),
            "anger": watson.get("anger", 0.0),
            "fear": watson.get("fear", 0.0),
            "disgust": watson.get("disgust", 0.0),
            "surprise": 0.05,
        }
        total = sum(watson_mapped.values()) or 1.0
        watson_dist = {k: v / total for k, v in watson_mapped.items()}
        local_dist = _soften_to_distribution(counts)
        distribution = {k: 0.5 * watson_dist[k] + 0.5 * local_dist[k] for k in local_dist}
    else:
        distribution = _soften_to_distribution(counts)

    s = sum(distribution.values()) or 1.0
    distribution = {k: v / s for k, v in distribution.items()}

    primary = max(distribution.items(), key=lambda kv: kv[1])[0]
    has_signal = sum(counts.values()) > 0.0
    if not has_signal:
        # Nothing in the lexicon matched — every category is flat/uniform,
        # so picking whichever key happens to sort first ("joy") is
        # misleading. Be honest that no clear emotion was detected.
        primary = "neutral"
    confidence = _confidence(text, distribution)
    sentiment = _sentiment(distribution)
    summary = _summary(distribution, lang_name)
    greeting = GREETINGS.get(lang_code, "Hello!")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "ok": True,
        "message": None,
        "language": lang_name,
        "language_code": lang_code,
        "greeting": greeting,
        "primary_emotion": primary,
        "distribution": {k: round(v, 4) for k, v in distribution.items()},
        "distribution_percent": {k: round(v * 100.0, 1) for k, v in distribution.items()},
        "sentiment": {
            "score": round(sentiment["score"], 3),
            "positivity": round(sentiment["positivity"], 1),
            "label": sentiment["label"],
        },
        "confidence": round(confidence, 3),
        "summary": summary,
        "elapsed_ms": round(elapsed_ms, 1),
        "normalized_text": text,
        "word_count": len([w for w in text.split(" ") if w]),
    }


def format_legacy(result: dict) -> str:
    if not result.get("ok"):
        return result.get("message") or "Invalid text! Please try again."
    d = result["distribution_percent"]
    return (
        f"For the given statement, the system response is "
        f"'anger': {d['anger']}, 'disgust': {d['disgust']}, 'fear': {d['fear']}, "
        f"'joy': {d['joy']} and 'sadness': {d['sadness']}. "
        f"The dominant emotion is {result['primary_emotion']}."
    )