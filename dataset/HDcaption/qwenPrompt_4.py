#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
from pathlib import Path

# -------- paths --------
IN_PATH  = "/root/dataset/skin/Derm1M/Derm1M_v2_pretrain.csv"
OUT_PATH = "/root/dataset/skin/Derm1M/Derm1M_v2_pretrain_min20_structured_4ok_multiRegion.xlsx"
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

SIDE_WORDS = ["left","right","bilateral","both"]
POS_WORDS  = ["upper","lower","mid","middle","proximal","distal","medial","lateral","anterior","posterior","superior","inferior"]

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
    r"\b(?:about|around|approx(?:\.|imately)?\s*)?\d+(\.\d+)?\s?(mm|millimeter|cm|centimeter)s?\b"
    r"|\b\d+\s?x\s?\d+\s?(mm|cm)\b"
    r"|\b(?:about|around|approx(?:\.|imately)?\s*)?\d+(\.\d+)?\s?(?:-|–|to)\s?\d+(\.\d+)?\s?(mm|cm)\b",
    re.I
)

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

def normalize_positional_adverbs(text: str) -> str:
    repl = {
        r"\bposteriorly\b": "posterior",
        r"\banteriorly\b": "anterior",
        r"\bmedially\b": "medial",
        r"\blaterally\b": "lateral",
        r"\bproximally\b": "proximal",
        r"\bdistally\b": "distal",
        r"\bsuperiorly\b": "superior",
        r"\binferiorly\b": "inferior",
    }
    out = text
    for pat, sub in repl.items():
        out = re.sub(pat, sub, out, flags=re.I)
    return out

def regions_from_body_location(val: str):
    if not isinstance(val, str):
        return []
    s = normalize_positional_adverbs(val.strip().lower())
    s = re.sub(r"\b(involving|involvement of|covering|affecting|over|area of|region of|from)\b", "", s, flags=re.I)
    parts = re.split(r"\s*(?:,|;|/|\band\b|\b&\b|\bto\b)\s*", s)
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]
    # keep phrases that contain a known region token to avoid noise
    region_alt = r"(?:%s)" % "|".join(sorted(set(REGION_WORDS), key=len, reverse=True))
    keep = []
    seen = set()
    for p in parts:
        if re.search(rf"\b{region_alt}\b", p, re.I):
            if p not in seen:
                seen.add(p)
                keep.append(p)
    return keep

def join_list_english(items):
    if not items: return ""
    if len(items) == 1: return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]

def size_term(text: str):
    m = SIZE_PAT.search(text)
    return m.group(0) if m else None

def hair_terms(text: str):
    return find_all(text, HAIR_WORDS, limit=2)

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
    cap = (row.get("caption") or "").strip()
    low = cap.lower()

    # Region comes from body_location column
    body_loc = (row.get("body_location") or "").strip()
    regions = regions_from_body_location(body_loc)

    gen_tex = texture_terms(low)
    hair    = hair_terms(low)

    sz   = size_term(low)
    shp  = shape_terms(low)
    bdr  = border_phrase(low)
    col  = color_term(low)
    ltex = texture_terms(low)
    elev = elevation_phrase(low)
    dmo  = dermoscopy_terms(low)

    region_ok   = len(regions) > 0
    gen_ok      = bool(gen_tex or hair)
    lesions_ok  = (int(sz is not None) + int(bool(shp)) + int(bdr is not None) + int(col is not None) + int(bool(ltex))) >= 3
    elevation_ok= elev is not None

    region_line = (
        f"The image suggests involvement of the {join_list_english(regions)}."
        if regions else
        "Not clearly discernible."
    )

    parts_gen = []
    if gen_tex: parts_gen.append(", ".join(gen_tex))
    if hair:    parts_gen.append(", ".join(hair))
    gen_line = (
        f"The surrounding skin shows {', '.join(parts_gen)}."
        if parts_gen else
        "Not clearly discernible."
    )

    lesion_bits = []
    if sz:   lesion_bits.append(f"size {sz}")
    if shp:  lesion_bits.append(f"shape {', '.join(shp)}")
    if bdr:  lesion_bits.append(f"margins {bdr}")
    if col:  lesion_bits.append(f"color {col}")
    if ltex: lesion_bits.append(f"texture {', '.join(ltex)}")
    if dmo:  lesion_bits.append(f"dermoscopy {', '.join(dmo)}")
    lesion_line = (
        f"A lesion with " + "; ".join(lesion_bits) + "."
        if lesion_bits else
        "Not clearly discernible."
    )

    elev_line = (
        f" {elev}."
        if elev else
        "Not clearly discernible."
    )

    block = (
        "Region: " + region_line + "\n\n"
        "General skin texture and hair growth: " + gen_line + "\n\n"
        "Lesions: " + lesion_line + "\n\n"
        "Elevation: " + elev_line + "\n\n"
    ).strip()

    return {
        "caption_format": block,
        "region_ok": region_ok,
        "gen_ok": gen_ok,
        "lesions_ok": lesions_ok,
        "elevation_ok": elevation_ok
    }

def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str)
    raise ValueError("Unsupported file type")

def main():
    df = load_table(IN_PATH)
    for c in ("caption", "body_location"):
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")
    df["caption"] = df["caption"].astype(str).fillna("").str.strip()
    df["body_location"] = df["body_location"].astype(str).fillna("").str.strip()

    df["word_count"] = df["caption"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS].copy()

    recs = df.apply(build_record, axis=1).apply(pd.Series)
    mask = recs["region_ok"] & recs["gen_ok"] & recs["lesions_ok"] & recs["elevation_ok"]

    out = df.loc[mask].copy()
    out["caption_format"] = recs.loc[mask, "caption_format"].values

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUT_PATH, index=False)

if __name__ == "__main__":
    main()
