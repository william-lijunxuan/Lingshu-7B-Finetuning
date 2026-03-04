import re
import json
import unicodedata




#####################################################################

def _norm(s: str) -> str:
    """Lightweight medical-term normalization."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    # strip diacritics (e.g., ç -> c)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # British ↔ American variants
    s = s.replace("naevus", "nevus").replace("naevi", "nevi").replace("haemangioma", "hemangioma")
    # Common abbreviations
    s = s.replace(" pih", " post-inflammatory hyperpigmentation").replace("pih", "post-inflammatory hyperpigmentation")
    s = s.replace("bcc", "basal cell carcinoma").replace("scc", "squamous cell carcinoma").replace(" sk", " seborrheic keratosis")
    s = s.replace(" ak", " actinic keratosis")
    # Hyphens → spaces, collapse non-alnum
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _alias_map():
    """Alias → canonical string map."""
    groups = [
        # exact pairs requested
        {"scar", "scarring", "cicatrix", "scar condition"},
        {"melanocytic nevus", "melanocytic nevi", "nevus", "nevi", "mole", "moles", "naevus", "naevi"},
        {"epidermal cyst", "epidermoid cyst", "sebaceous cyst", "infundibular cyst"},
        {"angioma", "hemangioma", "cherry angioma", "senile hemangioma", "haemangioma"},
        {"acrochordon", "skin tag", "skin tags"},

        {
            "behcet", "behcets", "behçet", "behçets",
            "behcet disease", "behcets disease",
            "behcet syndrome", "behcets syndrome",
            "behcet s disease", "behcets s disease",
            "behcet s syndrome", "behcets s syndrome",
            "behcet's disease", "behcets' disease",
            "behcet's syndrome", "behcets' syndrome",
            "behçet's disease", "behçets' disease",
            "behçet's syndrome", "behçets' syndrome",
            "beh et s disease", "beh et s syndrome"
        },

        {"eruptive xanthoma", "eruptive xanthomatosis"},
        # useful dermatology aliases
        {"basal cell carcinoma", "bcc"},
        {"squamous cell carcinoma", "scc"},
        {"seborrheic keratosis", "sk"},
        {"actinic keratosis", "ak"},
        {"post inflammatory hyperpigmentation", "post-inflammatory hyperpigmentation", "pih"},
        {"melasma", "chloasma"},
        {"tinea", "tinea capitis", "dermatophytosis", "ringworm"},
        {"molluscum contagiosum", "molluscum"},
        {"urticaria", "hives"},
        {"acne vulgaris", "acne"},
        {"psoriasis", "psoriasis vulgaris"},
        {"vitiligo"},
        {"lichen planus"},
        {"impetigo"},
        {"folliculitis"},
        {"paronychia"},
        {"onychomycosis"},
        {"cellulitis"},
        {"abscess"},
        {"rosacea"},
        {"viral wart", "wart", "verruca", "hpv wart"},
        {"Behçet's syndrome", "behcets disease", "behcets", "Behçet syndrome"},

        # hypertrichosis family
        {"hypertrichosis", "localized hypertrichosis", "faun tail nevus", "faun tail"},

        # amyloid / mucin
        {"amyloidosis", "cutaneous amyloidosis"},
        {"mucinosis", "follicular mucinosis"},

        # purpura variants
        {"pigmented progressive purpuric dermatosis", "pigmentary purpuric dermatosis", "purpura pigmentosa chronica"},

        # angiofibroma variants
        {"angiofibroma", "facial angiofibromas"},

        # drug eruption variants (CSV specific)
        {"drug eruption", "drug eruptions & reactions"},

        # grover disease
        {"grover's disease", "transient acantholytic dermatosis"},

        # puppp
        {"puppp", "pruritic urticarial papules and plaques of pregnancy"},

        # lichen simplex
        {"lichen simplex chronicus", "neurodermatitis"},

        # keratoderma
        {"keratoderma", "hyperkeratosis palmaris et plantaris"},

        # wound infection descriptive leaf
        {"local infection of wound", "abrasion, local infection of wound"},

        # cafe spelling
        {"café au lait macule", "cafe au lait macule"},

        # Behçet split-token variant seen in preds
        {"beh et s syndrome", "Behçet's syndrome", "behcets disease", "behcets", "Behçet syndrome"},

        # plural/singular
        {"freckle", "freckles"},

        # Kaposi punctuation/apostrophe variants
        {"kaposi sarcoma", "kaposi's sarcoma", "kaposi s sarcoma"},

        # striae synonyms
        {"striae", "stretch mark", "stretch marks"},

        # mucocele synonyms
        {"mucocele", "mucous cyst", "mucous gland cyst"},

        # HSV wording
        {"herpes simplex", "herpes simplex virus", "hsv"},

        # AD/eczema wording
        {"atopic dermatitis", "eczema", "atopic eczema"},

        # slapped cheek / parvovirus B19
        {"parvovirus b19 infection", "slapped cheek syndrome", "fifth disease"},

        # angioma common name variant
        {"angioma", "strawberry nevus", "strawberry haemangioma", "strawberry hemangioma"},

        # seborrheic keratosis plural
        {"seborrheic keratosis", "seborrheic keratoses"},

        # cutaneous larva migrans naming
        {"cutaneous larva migrans", "creeping eruption", "sand worm eruption"},

        {"varicella", "chickenpox"},
        {"sweet syndrome", "sweet s syndrome", "sweet's syndrome", "sweets syndrome", "sweet s disease"},
        {"granuloma faciale", "granuloma facialis"},
        {"polymorphous light eruption", "polymorphic light eruption"},
        {"morphea", "morphoea"},
        {"tinea pedis", "athlete s foot", "athlete's foot"},
        {"onychomycosis", "tinea unguium"},
        {"hand foot and mouth disease", "hand foot mouth disease", "hand foot & mouth disease"},
        {"poroma", "eccrine poroma"},
        {"furuncle", "furuncles"},
        {"condyloma", "condyloma acuminatum", "condyloma acuminata"},
        {"subungual hematoma", "subungual haemorrhage", "subungual hemorrhage"},
        {"pigmentary purpuric dermatosis", "pigmented purpuric dermatosis"},
        {"flat wart", "verruca plana"},
        {"chondrodermatitis nodularis helicis", "cnh"},
        {"dilated pore of winer", "pore of winer", "dilated pore"},
        {"darier disease", "darier's disease", "dariers disease", "darier s disease"},
        {"beh et s syndrome", "beh et s disease"},
        {"acute generalized exanthematous pustulosis", "agep"},
        {"riehl melanosis", "riehl's melanosis", "riehl s melanosis"},
        {"majocchi granuloma", "majocchi's granuloma", "majocchi s granuloma"},
        {"mucocele", "mucocoele", "mucocoeles", "mucoceles"},
        {"squamous cell carcinoma", "squamocellular carcinoma"},
        {"erythema annulare centrifugum", "centrifugal erythema annulare"},
        {"varicose vein", "varicose veins", "varicose veins of lower extremity"},
        {"eruptive xanthoma", "eruptive xanthomas", "eruptive xanthomatosis"},
        {"xanthoma", "xanthomas"},
        {"filiform wart", "warts", "viral wart", "wart", "verruca"},
        {"plantar warts", "plantar wart"},
        {"genital warts", "genital wart"},
    ]
    alias2canon = {}
    for g in groups:
        # choose a stable canonical: first sorted normalized term
        norm_group = sorted(_norm(x) for x in g if _norm(x))
        if not norm_group:
            continue
        canon = norm_group[0]
        for x in norm_group:
            alias2canon[x] = canon
    return alias2canon

_ALIAS2CANON = _alias_map()

def _canonical(term: str) -> str:
    n = _norm(term)
    return _ALIAS2CANON.get(n, n)

# Put this near your alias definitions
PARENT_MAP = {
    "dermatitis": {
        "seborrheic dermatitis",
        "atopic dermatitis",
        "allergic contact dermatitis",
        "irritant contact dermatitis",
        "follicular mite dermatitis",
    },
    "contact dermatitis": {
        "allergic contact dermatitis",
        "irritant contact dermatitis",
    },
    "angioma": {
        "infantile hemangioma",
        "strawberry hemangioma",
        "strawberry haemangioma",
        "hemangioma",
    },
    "hemangioma": {
        "infantile hemangioma",
        "strawberry hemangioma",
        "strawberry haemangioma",
    },
    "psoriasis": {
        "oral psoriasis",
        "psoriasis vulgaris",
    },
    "alopecia": {
        "alopecia areata",
        "androgenetic alopecia",
    },
    "ichthyosis": {
        "lamellar ichthyosis",
        "autosomal recessive congenital ichthyosis",
        "x linked ichthyosis",
    },
    "scar": {
        "atrophic scar",
        "hypertrophic scar",
        "cicatrix",
        "scarring",
    },
    "cicatrix": {
        "atrophic scar",
        "hypertrophic scar",
        "scar",
        "scarring",
    },
    "nail disease": {
        "onychomycosis",
    },
    "tinea": {
        "tinea capitis",
        "tinea corporis",
        "tinea cruris",
        "tinea pedis",
        "tinea unguium",
    },
    "mucocele": {
        "mucous cyst",
        "mucous gland cyst",
    },
    "erythema dyschromicum perstans": {
        "follicular erythema dyschromicum perstans",
    },
    "abscess": {
        "deep seated abscess",
    },
    "basal cell carcinoma": {
        "superficial basal cell carcinoma",
        "nodular basal cell carcinoma",
        "pigmented basal cell carcinoma",
    },
    "urticaria": {
        "papular urticaria",
    },
    "lupus erythematosus": {
        "systemic lupus erythematosus",
        "subacute cutaneous lupus erythematosus",
        "sle",
    },
    "elephantiasis nostras": {
        "elephantiasis nostras verrucosa",
    },
    "epidermolysis bullosa": {
        "epidermolysis bullosa simplex",
    },
    "follicular mucinosis": {
        "benign follicular mucinosis",
    },
    "bullous disease": {
        "bullous pemphigoid",
        "pemphigus vulgaris",
    },
    "viral exanthem": {
        "roseola",
    },
    "cellulitis": {
        "erysipelas",
    },
    "skin cancer": {
        "melanoma",
    },

    "dermatophytosis": {
        "tinea",
        "tinea corporis",
        "tinea cruris",
        "tinea pedis",
        "tinea capitis",
        "tinea unguium",
        "athlete s foot",
        "ringworm",
    },

    "skin diseases caused by warts": {
        "warts",
        "viral wart",
        "verruca",
        "verruca vulgaris",
        "verruca plana",
        "flat wart",
        "filiform wart",
        "genital wart",
        "genital warts",
        "plantar wart",
        "plantar warts",
    },

    "warts": {
        "viral wart",
        "wart",
        "verruca",
        "verruca vulgaris",
        "verruca plana",
        "flat wart",
        "filiform wart",
        "genital wart",
        "genital warts",
        "plantar wart",
        "plantar warts",
    },

    "morphea": {
        "pansclerotic morphea",
        "generalized morphea",
        "linear morphea",
        "circumscribed morphea",
        "morphoea",
        "pansclerotic morphoea",
    },

    "cyst": {
        "steatocystoma",
        "steatocystoma multiplex",
        "epidermal cyst",
        "epidermoid cyst",
        "pilar cyst",
        "steatocystoma simplex",
    },
    "stretch mark": {
        "striae alba",
        "striae distensae",
        "linea nigra",
        "striae",
    },
    "porokeratosis": {
        "disseminated superficial actinic porokeratosis",
        "dsap",
    },
    "neurofibromatosis": {
        "generalized neurofibromatosis",
    },
    "poikiloderma": {
        "civatte poikiloderma",
    },
    "elephantiasis": {
        "elephantiasis nostras",
        "elephantiasis nostras verrucosa",
    },
    "varicose vein": {
        "varicose veins",
        "varicose veins of lower extremity",
    },
}


# Normalize keys once for safety
PARENT_MAP = {_norm(k): {_norm(x) for x in v} for k, v in PARENT_MAP.items()}



def _extract_pred_answer(response):
    if response is None:
        return None

    if isinstance(response, dict):
        return response.get("answer", None)

    if isinstance(response, list):
        return None

    if isinstance(response, (bytes, bytearray)):
        response = response.decode("utf-8", errors="ignore")

    if not isinstance(response, str):
        response = str(response)

    s = response.strip()

    # remove markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj.get("answer", None)
        return None
    except Exception:
        fixed = s

        # fix numeric ranges like 1-3
        fixed = re.sub(r'"size_mm"\s*:\s*(\d+)\s*-\s*(\d+)', r'"size_mm": [\1, \2]', fixed)

        # remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

        # drop obvious "bare lines" inside objects/arrays that are not valid JSON
        # e.g. Gottron's papules",
        fixed_lines = []
        for line in fixed.splitlines():
            t = line.strip()
            if not t:
                continue
            # keep lines that look like JSON syntax fragments
            if t.startswith(("{", "}", "[", "]")):
                fixed_lines.append(line)
                continue
            # keep proper key/value lines or list items
            if re.match(r'^\s*".+"\s*:\s*', line) or re.match(r'^\s*".+"\s*,\s*$', line):
                fixed_lines.append(line)
                continue
            # otherwise drop (this removes the bad Gottron's papules line)
            continue
        fixed = "\n".join(fixed_lines)

        # cautious single-quote dict fix (only if no double quotes exist)
        if "{" in fixed and "}" in fixed and "'" in fixed and '"' not in fixed:
            fixed = fixed.replace("'", '"')

        try:
            obj = json.loads(fixed)
            if isinstance(obj, dict):
                return obj.get("answer", None)
        except Exception:
            pass

    # try to decode the first JSON object/array from the first "{" or "["
    m = re.search(r"[\{\[]", s)
    if not m:
        # regex fallback
        m_ans = re.search(r'"answer"\s*:\s*"([^"]+)"', s)
        return m_ans.group(1).strip() if m_ans else None

    s2 = s[m.start():]

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s2)
        if isinstance(obj, dict):
            return obj.get("answer", None)
        return None
    except Exception:
        # last resort: try substring between first { and last }
        start = s2.find("{")
        end = s2.rfind("}")
        if start != -1 and end != -1 and end > start:
            chunk = s2[start:end + 1]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj.get("answer", None)
            except Exception:
                pass

        # final regex fallback (works even if JSON is broken)
        m_ans = re.search(r'"answer"\s*:\s*"([^"]+)"', s)
        return m_ans.group(1).strip() if m_ans else None



def judge_close_end_vqa_json(answer: str, response: str) -> bool:
    pred = _extract_pred_answer(response)
    if pred is None:
        # print("answer:", answer, "pred:", pred,"response",response)
        return False

    a_norm = _norm(answer)
    p_norm = _norm(pred)

    # exact match after normalization
    if a_norm == p_norm:
        return True

    # alias canonical match
    a_can = _canonical(a_norm)
    p_can = _canonical(p_norm)
    if a_can == p_can:
        return True

    # hierarchical match: answer is parent, pred is child (after canonicalization)
    # we apply canonical here to reduce variant issues
    a_h = _canonical(a_can)
    p_h = _canonical(p_can)

    children = PARENT_MAP.get(a_h)
    if children and p_h in children:
        return True

    print("answer:", a_h, "pre:", p_h, False)
    # print("answer:",answer,"pred:",pred)
    return False


