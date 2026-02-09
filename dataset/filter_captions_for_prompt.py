#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
from pathlib import Path

# -------- paths --------
IN_PATH  = "/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"   # .csv / .xlsx / .xls
OUT_PATH = "/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain_min20_formatted.xlsx"
MIN_WORDS = 20

# -------- lexicons (augmented) --------
COLOR_MAP = {
    r"\berythematous\b": "red",
    r"\bred\b": "red",
    r"\bpink\b": "pink",
    r"\bbrown\b|\bbrown\s*\(hyperpigment\w*\)": "brown",
    r"\bblack\b": "black",
    r"\bblue\b|\bbluish\b|\bblue[- ]white\w*\b": "blue",
    r"\bviolaceous\b|\bpurple\b": "purple",
    r"\byellow\b|\byellowish\b|xanthomatous": "yellow",
    r"\bwhite\b|\bwhitish\b|\bwhite\s*\(hypopigment\w*\)": "white",
    r"\btan\b": "tan",
    r"\bsalmon\b": "salmon",
    r"\bgray\b|\bgrey\b|\bgr[ae]yish\b": "gray",
    r"\bskin[- ]colou?red\b|flesh[- ]?colored": "skin-colored",
    r"\bhyperpigment\w*\b": "hyperpigmented",
    r"\bhypopigment\w*\b": "hypopigmented",
    r"\bpigment\w*\b": "pigmented",
    r"(?:cafe|caf[é])[- ]?au[- ]?lait": "café-au-lait"
}

TEXTURE_WORDS = [
    "smooth","rough","scaly","flaky","greasy","dry","oily",
    r"keratotic|hyperkerat\w*",
    r"lichenif\w*","indurat\w*","atroph\w*","wrinkled",
    r"ulcerat\w*","crust\w*","weeping","oozing","friabl\w*","exudat\w*",
    "verrucous","warty","thickened","thin",
    r"xerosis|xerotic","boggy","sclerotic","macerat\w*","desquamation","translucent","sclerosis","poikiloderma"
]

SHAPE_WORDS = [
    "round","oval","annular","linear","irregular","stellate","polygonal",
    "arciform","arcuate","targetoid","serpiginous","confluent",
    "dome-shaped","flat-topped","polycyclic","gyrate","reticular","reticulated","geographic",
    "leaf-shaped","wedge-shaped","angulated","acuminate","geometric","regular","symmetric","asymmetric"
]

ARRANGEMENT_WORDS = [
    "clustered","grouped","satellite","scattered","generalized","localized","disseminated","discrete","diffuse",
    "linear arrangement","reticulated pattern","dermatomal","zosteriform","herpetiform",
    "follicular-centered","perifollicular","dermatomal distribution"
]

REGION_WORDS = [
    "scalp","face","forehead","temple","cheek","nose","nasal","lip","ear","eyelid","periorbital","chin",
    "neck","chest","breast","back","abdomen","belly","stomach","groin","inguinal","axilla","armpit",
    "buttock","gluteal","arm","forearm","elbow","wrist","hand","palm","finger","thumb","nail","cuticle",
    "leg","thigh","knee","shin","calf","ankle","foot","heel","sole","toe","trunk","flank",
    "oral","tongue","buccal","genital","penis","scrotum","vulva","labia","perineum","mucosa","perioral",
    "palmar","plantar","dorsal","ventral","dorsum","subungual","periungual","interdigital","intertriginous",
    "auricular","preauricular","postauricular","tragus","conchal","alar","nasolabial","vermillion","vermilion",
    "malleolus","parotid","mandibular","occipital","temporal","suprapubic","perianal","periareolar","areola",
    "scapular","clavicular","thenar","hypothenar","webspace"
]

BORDER_PHRASES_GOOD = [
    r"well[- ]defined", r"well[- ]demarcated", r"sharply[- ]defined", r"sharply[- ]demarcated",
    r"clear margin", r"\bcircumscribed\b", r"\bsharp\b"
]
BORDER_PHRASES_POOR = [r"ill[- ]defined", r"poorly defined", r"indistinct", r"blurred(?: margin)?", r"poorly[- ]?defined"]
BORDER_PHRASES_OTHER = [r"rolled border", r"\bpearl\w*\b", r"collarette\w*"]

ELEVATION_ADJ = [
    "raised","elevated","flat","depressed","umbilicated","pedunculated","sessile","exophytic","fungating","vegetating","translucent"
]
ELEVATION_NOUN = [
    "macule","patch","papule","plaque","nodule","tumor","vesicle","bulla","pustule","wheal",
    "cyst","abscess","ulcer","erosion","fissure","comedo","burrow",
    r"maculopapule|maculo[- ]?papule", r"maculopatch|maculo[- ]?patch",
    r"papulopustule|papulo[- ]?pustule", r"papulovesicle|papulo[- ]?vesicle", r"papulonodule|papulo[- ]?nodule",
    "molluscoid","keloid","scar","necrosis","ecchymotic","hemorrhage"
]

SURROUNDING_HINT = [
    "surrounding","perilesional","around","adjacent","background","peripheral","bordering","rim","halo"
]
SURROUNDING_FINDINGS = [
    r"erythema\w*","indurat\w*","edema|oedema","atroph\w*","hyperpigment\w*","hypopigment\w*",
    r"scale\w*","scaly","crust\w*","bleed\w*","purpura\w*","petech\w*","ecchym\w*","excoriat\w*","lichenif\w*",
    r"telangiect\w*","angioma\w*","hemangi\w*","capillar\w*","spider","friabl\w*","exudat\w*"
]

DX_MAP = {
    r"\bbcc\b|basal cell carcinoma": "basal cell carcinoma",
    r"\bscc\b|squamous cell carcinoma": "squamous cell carcinoma",
    r"\bmelanoma\b": "melanoma",
    r"\bnev(us|i)|naev(us|i)|melanocytic": "melanocytic nevus",
    r"seborrheic keratosis|\bsk\b": "seborrheic keratosis",
    r"actinic keratosis|\bak\b": "actinic keratosis",
    r"\bpsoriasis\b|psoriasiform": "psoriasis",
    r"\beczema\b|dermatitis|eczematous": "dermatitis",
    r"\btinea\b|dermatophyte|ringworm": "tinea",
    r"\bwart\b|verruca|hpv|warty|papillomatous": "viral wart",
    r"hemangioma|haemangioma|angioma": "hemangioma",
    r"\bimpetigo\b": "impetigo",
    r"\bfolliculitis\b|perifollicular|follicular-centered": "folliculitis",
    r"\bacne\b|comedo\w*": "acne vulgaris",
    r"\brosacea\b": "rosacea",
    r"\bmolluscum\b|molluscoid": "molluscum contagiosum",
    r"onychomycosis": "onychomycosis",
    r"paronychia": "paronychia",
    r"cellulitis": "cellulitis",
    r"\babscess\b": "abscess",
    r"\bscabies\b|burrow": "scabies",
    r"\burticaria\b|hive|wheal": "urticaria",
    r"lichen planus|lichenoid": "lichen planus",
    r"\bvitiligo\b": "vitiligo",
    r"pityriasis rosea": "pityriasis rosea",
    r"xanthoma|xanthomatous": "xanthoma"
}

NEXTSTEP_RULES = {
    # dermoscopy features
    "dermoscopy": re.compile(
        r"pigment network|dots and globules|streaks|regression structures|blue[- ]white\w* veil|vascular structures|telangiect\w*",
        re.I
    ),
    # biopsy flags
    "biopsy": re.compile(
        r"irregular|ulcerat\w*|necros\w*|bleed\w*|hemorrhag\w*|variegat\w*|asymmetr\w*|rapid(ly)?\s*grow|very dark|black\b|blue\b",
        re.I
    ),
}

# -------- helpers --------
def first_match(text: str, patterns):
    if not text:
        return None
    if isinstance(patterns, (list, tuple)):
        for p in patterns:
            m = re.search(rf"\b{p}\b" if p.isalpha() else p, text, re.I)
            if m:
                return m.group(0).lower()
        return None
    if isinstance(patterns, dict):
        for p, val in patterns.items():
            if re.search(p, text, re.I):
                return val
        return None
    m = re.search(patterns, text, re.I)
    return m.group(0).lower() if m else None

def border_phrase(text: str):
    if any(re.search(p, text, re.I) for p in BORDER_PHRASES_GOOD):
        return "well-defined margins"
    if any(re.search(p, text, re.I) for p in BORDER_PHRASES_POOR):
        return "ill-defined margins"
    for p in BORDER_PHRASES_OTHER:
        m = re.search(p, text, re.I)
        if m:
            return m.group(0).lower()
    return None

def elevation_phrase(text: str):
    adj = first_match(text, ELEVATION_ADJ)
    noun = first_match(text, ELEVATION_NOUN)
    if adj and noun: return f"{adj}, {noun}"
    if noun:         return noun
    if adj:          return adj
    return None

def color_texture_pair(text: str):
    col = first_match(text, COLOR_MAP)
    tex = None
    for w in TEXTURE_WORDS:
        m = re.search(w if not w.isalpha() else rf"\b{w}\b", text, re.I)
        if m:
            tex = m.group(0).lower()
            break
    if col and tex: return f"{col}/{tex}"
    if col:         return col
    if tex:         return tex
    return None

def pick_region(text: str):
    for w in REGION_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", text, re.I):
            return w
    return None

def pick_shape(text: str):
    s = first_match(text, SHAPE_WORDS)
    if s:
        return s
    return first_match(text, ARRANGEMENT_WORDS)

def guess_dx(text: str, row):
    for k in ("disease_label","diagnosis","dx","label"):
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    for p, name in DX_MAP.items():
        if re.search(p, text, re.I):
            return name
    return "an undetermined dermatologic process"

def guess_next_step(text: str):
    if NEXTSTEP_RULES["biopsy"].search(text):
        return "a biopsy is advised for confirmation"
    if NEXTSTEP_RULES["dermoscopy"].search(text):
        return "dermoscopic evaluation and clinical correlation are recommended"
    return None

def build_sentence(row) -> str:
    t = (row.get("caption") or "").strip()
    low = t.lower()

    ct   = color_texture_pair(low)
    shp  = pick_shape(low)
    reg  = pick_region(low)
    bdr  = border_phrase(low)
    elev = elevation_phrase(low)
    dx   = guess_dx(low, row)
    ns   = guess_next_step(low)

    lead_parts = []
    if ct:  lead_parts.append(ct)
    if shp: lead_parts.append(shp)
    lead = ", ".join(lead_parts).strip()

    if lead and reg:
        head = f"The {lead} lesion on the {reg}"
    elif lead:
        head = f"The {lead} lesion"
    elif reg:
        head = f"The lesion on the {reg}"
    else:
        head = "The lesion"

    with_parts = [p for p in (bdr, elev) if p]
    with_str = f" with {' and '.join(with_parts)}" if with_parts else ""

    sent = f"{head}{with_str} is most consistent with {dx}"
    if ns:
        sent += f"; {ns}"
    sent = re.sub(r"\s+", " ", sent).strip()
    if not re.search(r"[.!?]$", sent):
        sent += "."
    return sent

# -------- io --------
def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str)
    raise ValueError("Unsupported file type")

def main():
    df = load_table(IN_PATH)
    if "caption" not in df.columns:
        raise RuntimeError("Missing 'caption' column.")
    df["caption"] = df["caption"].astype(str).fillna("").str.strip()
    df["word_count"] = df["caption"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS].copy()
    df["caption_format"] = df.apply(build_sentence, axis=1)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT_PATH, index=False)

if __name__ == "__main__":
    main()
