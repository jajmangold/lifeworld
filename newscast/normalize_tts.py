"""TTS text normalization (backstop). The local Qwen3-TTS base clone has NO normalization frontend,
so it mispronounces digits, currency, %, and acronyms. The DeepSeek script writer is asked to emit a
TTS-ready 'spoken_script', but this catches anything it misses. Deterministic, no hard deps (uses
num2words if present, else a built-in fallback). Importable: normalize(text) -> str.
  echo "NNS reports 25% in 2026, $5M, Dr. Smith on St. Paul" | python3 newscast/normalize_tts.py
"""
import re, sys

try:
    from num2words import num2words as _n2w
    def _num(n): return _n2w(int(n))
    def _ord(n): return _n2w(int(n), to="ordinal")
    def _year(n):
        n = int(n)
        if 1100 <= n <= 1999 and n % 100 != 0:
            return _n2w(n // 100) + " " + (_n2w(n % 100) if n % 100 >= 10 else "oh " + _n2w(n % 100))
        if 2000 <= n <= 2099:
            return _n2w(n) if n % 100 < 10 else "twenty " + _n2w(n % 100)
        return _n2w(n)
except Exception:
    _ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    _TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
    def _u(n):
        n = int(n)
        if n < 20: return _ONES[n]
        if n < 100: return _TENS[n//10] + (("-"+_ONES[n%10]) if n%10 else "")
        if n < 1000: return _ONES[n//100]+" hundred"+((" "+_u(n%100)) if n%100 else "")
        if n < 1_000_000: return _u(n//1000)+" thousand"+((" "+_u(n%1000)) if n%1000 else "")
        if n < 1_000_000_000: return _u(n//1_000_000)+" million"+((" "+_u(n%1_000_000)) if n%1_000_000 else "")
        return _u(n//1_000_000_000)+" billion"+((" "+_u(n%1_000_000_000)) if n%1_000_000_000 else "")
    def _num(n): return _u(n)
    _ORD_IRR = {"one":"first","two":"second","three":"third","five":"fifth","eight":"eighth",
                "nine":"ninth","twelve":"twelfth"}
    def _ord(n):
        w = _u(int(n)); parts = w.replace("-", " ").rsplit(" ", 1); last = parts[-1]
        if last in _ORD_IRR: last = _ORD_IRR[last]
        elif last.endswith("y"): last = last[:-1] + "ieth"
        else: last += "th"
        head = (parts[0] + ("-" if "-" in w else " ")) if len(parts) > 1 else ""
        return head + last
    def _year(n):
        n = int(n)
        if 2000 <= n <= 2009: return "two thousand"+((" "+_u(n%100)) if n%100 else "")
        if 2010 <= n <= 2099: return "twenty "+_u(n%100)
        if 1100 <= n <= 1999 and n % 100: return _u(n//100)+" "+(_u(n%100) if n%100>=10 else "oh "+_u(n%100))
        return _u(n)

# pronounceable acronyms a TTS says as a word -> leave alone; everything else all-caps -> spell out
_WORD_ACRONYMS = {"NASA","NATO","OPEC","UNESCO","UNICEF","SCOTUS","POTUS","FEMA","NAFTA","ASEAN","FIFA","COVID","AIDS","LASER","RADAR","SWAT","DACA","START"}
_ABBR = {r"\bDr\.": "Doctor", r"\bMr\.": "Mister", r"\bMrs\.": "Missus", r"\bMs\.": "Miss",
         r"\bSt\.": "Saint", r"\bMt\.": "Mount", r"\bAve\.": "Avenue", r"\bBlvd\.": "Boulevard",
         r"\bGov\.": "Governor", r"\bSen\.": "Senator", r"\bRep\.": "Representative",
         r"\bGen\.": "General", r"\bSgt\.": "Sergeant", r"\bvs\.?": "versus", r"\bU\.S\.": "U S",
         r"\bU\.K\.": "U K", r"\ba\.m\.": "A M", r"\bp\.m\.": "P M", r"\bmph\b": "miles per hour",
         r"\bkm\b": "kilometers", r"\bkg\b": "kilograms", r"\bNo\.": "number", r"\b&": " and "}

def _spell(m):
    w = m.group(0)
    if w in _WORD_ACRONYMS or len(w) < 2: return w
    return " ".join(list(w))                      # FBI -> "F B I"

def normalize(t: str) -> str:
    t = (t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
           .replace("—", ", ").replace("–", "-").replace("…", "..."))
    for pat, rep in _ABBR.items():
        t = re.sub(pat, rep, t)
    # currency: $5M, $5 million, $1,200 -> words + " dollars"
    def _money(m):
        num = m.group(1).replace(",", ""); suf = (m.group(2) or "").lower()
        scale = {"k":" thousand","m":" million","b":" billion","": ""}.get(suf[:1] if suf in ("k","m","b") else "", "")
        word = "million" if suf.startswith("mill") else "billion" if suf.startswith("bill") else "thousand" if suf.startswith("thou") else scale.strip()
        base = _num(float(num)) if "." not in num else _num(num.split(".")[0])
        return f"{base}{(' '+word) if word else ''} dollars"
    t = re.sub(r"\$([0-9][0-9,\.]*)\s*(million|billion|thousand|[KkMmBb])?\b", _money, t)
    # percent
    t = re.sub(r"([0-9][0-9,\.]*)\s*%", lambda m: _num(m.group(1).replace(",","").split(".")[0]) + " percent", t)
    # years (4-digit 1100-2099)
    t = re.sub(r"\b(1[1-9]\d{2}|20\d{2})\b", lambda m: _year(m.group(1)), t)
    # ordinals 1st/2nd/3rd/4th
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _ord(m.group(1)), t)
    # plain integers (with thousands separators)
    t = re.sub(r"\b\d[\d,]*\b", lambda m: _num(m.group(0).replace(",", "")), t)
    # acronyms: 2+ uppercase letters (optionally with internal dots) -> spell unless a known word-acronym
    t = re.sub(r"\b[A-Z]{2,5}\b", _spell, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

if __name__ == "__main__":
    src = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[1:])
    print(normalize(src))
