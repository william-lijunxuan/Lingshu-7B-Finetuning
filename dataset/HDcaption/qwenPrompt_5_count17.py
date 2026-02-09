#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
from pathlib import Path

# -------- paths --------
IN_PATH  = "/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"   # .csv / .xlsx / .xls
OUT_PATH = "/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_v2_pretrain_min20_structured_5ok.xlsx"
MIN_WORDS = 20

# -------- lexicons (augmented) --------
COLOR_MAP = {
    r"\berythematous\b": "red",
    r"\bred\b": "red",
    r"\bpink\b": "pink",
    r"\bbrown\b|\bbrown\s*\(hyperpigment\w*\)": "brown",
    r"\bblack\b": "black",
    r"\bblue\b|\bbluish\b|\bblue[- ]white\w*\b|\bblue whitish veil\b": "blue",
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

HAIR_WORDS = [
    "hair","hairy","hairless","shaven","shaved","stubble","beard","eyebrow","eyelash","hirsute","vellus","terminal hair"
]

DERMOSCOPY_FEATURES = [
    "pigment network","dots and globules","streaks","regression structures","vascular structures",
    "blue-white veil","blue whitish veil","telangiectasia","telangiectatic"
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

SIZE_PAT = re.compile(
    r"\b\d+(\.\d+)?\s?(mm|millimeter|cm|centimeter)s?\b|\b\d+\s?x\s?\d+\s?(mm|cm)\b|\bmeasures?\s+\d+(\.\d+)?\s?(mm|cm)\b",
    re.I
)

NEAR_SURROUND_PAT = re.compile(
    r"(surrounding|perilesional|around|adjacent|background|peripheral|bordering|rim|halo)[^\n,.]{0,25}"
    r"(erythema|induration|edema|oedema|atroph|hyperpigment|hypopigment|scale|scaly|crust|bleed|purpura|petech|ecchym|excoriat|lichenif|telangiect|angioma|hemangi|capillar|spider|friabl|exudat)",
    re.I
)

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

def find_all(text: str, patterns, limit=5):
    hits = []
    for p in patterns:
        m = re.search(rf"\b{p}\b" if isinstance(p, str) and p.isalpha() else p, text, re.I)
        if m:
            val = m.group(0).lower()
            if val not in hits:
                hits.append(val)
        if len(hits) >= limit:
            break
    return hits

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

def color_term(text: str):
    return first_match(text, COLOR_MAP)

def texture_terms(text: str):
    return find_all(text, TEXTURE_WORDS, limit=3)

def shape_terms(text: str):
    hits = find_all(text, SHAPE_WORDS, limit=3)
    if not hits:
        hits = find_all(text, ARRANGEMENT_WORDS, limit=3)
    return hits

def pick_region(text: str):
    for w in REGION_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", text, re.I):
            return w
    return None

def size_term(text: str):
    m = SIZE_PAT.search(text)
    return m.group(0) if m else None

def hair_terms(text: str):
    return find_all(text, HAIR_WORDS, limit=2)

def surrounding_phrase(text: str):
    if NEAR_SURROUND_PAT.search(text):
        findings = find_all(text, SURROUNDING_FINDINGS, limit=3)
        if findings:
            return " ".join(sorted(set(findings)))
        return "present"
    return None

def dermoscopy_terms(text: str):
    return find_all(text, DERMOSCOPY_FEATURES, limit=3)

def guess_dx(text: str, row):
    for k in ("disease_label","diagnosis","dx","label"):
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    for p, name in DX_MAP.items():
        if re.search(p, text, re.I):
            return name
    return None

def build_record(row) -> dict:
    t = (row.get("caption") or "").strip()
    low = t.lower()

    reg   = pick_region(low)
    gen_tex = texture_terms(low)
    hair   = hair_terms(low)

    sz   = size_term(low)
    shp  = shape_terms(low)
    bdr  = border_phrase(low)
    col  = color_term(low)
    ltex = texture_terms(low)
    elev = elevation_phrase(low)
    surr = surrounding_phrase(low)
    dmo  = dermoscopy_terms(low)

    region_ok = reg is not None
    gen_ok = bool(gen_tex or hair)
    lesion_sub = int(sz is not None) + int(bool(shp)) + int(bdr is not None) + int(col is not None) + int(bool(ltex))
    lesions_ok = lesion_sub >= 3
    elevation_ok = elev is not None
    surrounding_ok = surr is not None

    region_line = f"The image suggests involvement of the {reg}." if reg else "Not clearly discernible."
    gen_parts = []
    if gen_tex: gen_parts.append(", ".join(gen_tex))
    if hair:    gen_parts.append(", ".join(hair))
    gen_line = f"The surrounding skin shows {', '.join(gen_parts)}." if gen_parts else "Not clearly discernible."

    lesion_bits = []
    if sz:   lesion_bits.append(f"size {sz}")
    if shp:  lesion_bits.append(f"shape {', '.join(shp)}")
    if bdr:  lesion_bits.append(f"margins {bdr}")
    if col:  lesion_bits.append(f"color {col}")
    if ltex: lesion_bits.append(f"texture {', '.join(ltex)}")
    if dmo:  lesion_bits.append(f"dermoscopy {', '.join(dmo)}")
    lesion_line = f"A lesion with " + "; ".join(lesion_bits) + "." if lesion_bits else "Not clearly discernible."

    elev_line = f"Elevation relative to the skin surface: {elev}." if elev else "Not clearly discernible."
    surr_line = f"The surrounding skin shows {surr}." if surr else "Not clearly discernible."

    block = (
        "Region: " + region_line + "\n\n"
        "General skin texture and hair growth: " + gen_line + "\n\n"
        "Lesions: " + lesion_line + "\n\n"
        "Elevation: " + elev_line + "\n\n"
        "Skin texture surrounding the lesion: " + surr_line
    ).strip()

    return {
        "caption_format": block,
        "region_ok": region_ok,
        "gen_ok": gen_ok,
        "lesions_ok": lesions_ok,
        "elevation_ok": elevation_ok,
        "surrounding_ok": surrounding_ok
    }

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

    recs = df.apply(build_record, axis=1).apply(pd.Series)
    mask = recs["region_ok"] & recs["gen_ok"] & recs["lesions_ok"] & recs["elevation_ok"] & recs["surrounding_ok"]

    out = df.loc[mask].copy()
    out["caption_format"] = recs.loc[mask, "caption_format"].values

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUT_PATH, index=False)

if __name__ == "__main__":
    main()
