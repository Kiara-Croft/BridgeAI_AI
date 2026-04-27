"""PASUL 5"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import score_products, format_results, CATEGORY_MAP, INTENT_TO_PROFILE

# Translation maps for user-facing display 
CATEGORY_EN = {
    "laptop":           "laptop",
    "smartphone":       "smartphone",
    "monitor":          "monitor",
    "tv":               "TV",
    "tableta":          "tablet",
    "casti":            "headphones",
    "casti gaming":     "gaming headphones",
}

SCOP_EN = {
    "casual":           "everyday use",
    "gaming":           "gaming",
    "gaming entry":     "entry-level gaming",
    "business":         "business",
    "programare":       "programming",
    "design":           "design",
    "video editing":    "video editing",
    "streaming":        "streaming",
    "student":          "student use",
    "student premium":  "student premium",
    "fotografie":       "photography",
    "muzica":           "music",
    "buget mic":        "budget",
    "social media":     "social media",
    "transport":        "commuting",
    "sport":            "sport",
    "anc":              "noise cancelling",
    "voip":             "calls / VoIP",
    "portabil":         "portability",
    "general":          "general use",
}

def _cat_en(categorie: str) -> str:
    return CATEGORY_EN.get(categorie, categorie)

def _scop_en(scop: str) -> str:
    return SCOP_EN.get(scop, scop)
 
def _simple_nlp_fallback(user_text: str) -> str:
    text = user_text.lower()
 
    categorie = "laptop"  
    if any(w in text for w in ["monitor", "ecran", "display", "screen"]):
        categorie = "monitor"
    elif any(w in text for w in ["casti gaming", "gaming headphone", "gaming headphones", "gaming headset"]):
        categorie = "casti gaming"
    elif any(w in text for w in ["headphone", "headphones", "earphone", "earbuds", "airpods", "casti"]):
        categorie = "casti"
    elif any(w in text for w in ["telefon", "smartphone", "mobil", "iphone", "android", "galaxy"]) or \
         (any(w in text for w in ["phone"]) and not any(w in text for w in ["headphone", "earphone"])):
        categorie = "smartphone"
    elif any(w in text for w in ["pixel"]) and not any(w in text for w in ["headphone"]):
        categorie = "smartphone"
    elif any(w in text for w in ["tv", "televizor", "televiziune"]):
        categorie = "tv"
    elif any(w in text for w in ["tableta", "tablet", "ipad"]):
        categorie = "tableta"

    # Detectare brand  
    brand = ""
    brand_keywords = {
        "apple":    ["apple", "macbook", "mac book", "imac", "mac mini", "iphone", "ipad", "airpods"],
        "iphone":   ["apple", "iphone"],
        "samsung":  ["samsung", "galaxy"],
        "asus":     ["asus", "rog", "zenbook", "vivobook", "tuf"],
        "dell":     ["dell", "xps", "inspiron", "latitude", "alienware"],
        "lenovo":   ["lenovo", "thinkpad", "ideapad", "legion", "yoga"],
        "hp":       ["hp", "hewlett", "pavilion", "spectre", "envy", "omen"],
        "acer":     ["acer", "predator", "aspire", "nitro", "swift"],
        "msi":      ["msi", "stealth", "raider", "creator"],
        "lg":       ["lg"],
        "sony":     ["sony", "xperia"],
        "microsoft":["microsoft", "surface"],
        "razer":    ["razer", "blade"],
        "huawei":   ["huawei", "matebook"],
        "xiaomi":   ["xiaomi", "redmi", "poco"],
        "oneplus":  ["oneplus", "one plus"],
        "google":   ["google", "pixel"],
        "realme":   ["realme"],
        "motorola": ["motorola", "moto"],
        "nokia":    ["nokia"],
    }
    for b, keywords in brand_keywords.items():
        if any(kw in text for kw in keywords):
            brand = b
            break
 
    if brand == "iphone":
        brand = "apple"

    # Detectare scop  
    scop = "casual"  # default

    if any(w in text for w in ["programare", "programator", "coding", "code", "development", "developer",
                                "python", "java", "javascript", "c++", "rust", "ide", "vscode",
                                "terminal", "linux", "ubuntu", "backend", "frontend", "web dev"]):
        scop = "programare"
    elif any(w in text for w in ["gaming", "jocuri", "games", "cs", "fps", "fortnite", "dota",
                                  "gamer", "pe gaming", "valorant", "lol"]):
        scop = "gaming" if not any(x in text for x in ["entry", "budget", "ieftin", "mic"]) else "gaming entry"
    elif any(w in text for w in ["business", "munca", "lucru", "office", "birou",
                                  "profesional", "lucru de birou", "excel", "word", "powerpoint"]):
        scop = "business"
    elif any(w in text for w in ["design", "grafica", "photoshop", "illustrator", "cad",
                                  "autocad", "solidworks", "figma", "sketch", "indesign"]):
        scop = "design"
    elif any(w in text for w in ["video editing", "premiere", "davinci", "final cut",
                                  "content creator", "youtube", "twitch", "montaj video"]):
        scop = "video editing"
    elif any(w in text for w in ["streaming", "film", "filme", "multimedia", "netflix",
                                  "hbo", "disney", "divertisment"]):
        scop = "streaming"
    elif any(w in text for w in ["student", "scoala", "facultate", "universitate", "cursuri", "liceu"]):
        if any(x in text for x in ["programare", "coding", "inginerie", "cad", "autocad", "performant"]):
            scop = "programare"
        elif any(x in text for x in ["apple", "macbook", "mac", "premium"]):
            scop = "student premium"
        else:
            scop = "student"
    elif any(w in text for w in ["foto", "fotografie", "camera", "poze", "photography", "lightroom"]):
        scop = "fotografie"
    elif any(w in text for w in ["muzica", "music", "podcast", "audiophile", "ascultare", "audio"]):
        scop = "muzica"
    elif any(w in text for w in ["ieftin", "buget mic", "budget", "economic", "ieftina", "second"]):
        scop = "buget mic"
    elif any(w in text for w in ["social media", "instagram", "tiktok", "facebook"]):
        scop = "social media"
    elif any(w in text for w in ["metrou", "naveta", "transport", "calatorii", "calatorie"]):
        scop = "transport"
    elif any(w in text for w in ["sport", "antrenament", "sala", "alergare", "workout", "gym"]):
        scop = "sport"
    elif any(w in text for w in ["anc", "zgomot", "noise canceling", "noise cancelling", "anulare zgomot"]):
        scop = "anc"
    elif any(w in text for w in ["voip", "apel", "call", "conferinte", "zoom", "teams", "meet"]):
        scop = "voip"
    elif any(w in text for w in ["portabil", "usor", "lightweight", "mobilitate", "slim", "subtire"]):
        scop = "portabil"
    elif brand == "apple" and categorie == "laptop":
        scop = "casual"
    elif any(w in text for w in ["general", "uz general", "mixt", "mix", "echilibrat", "orice"]):
        scop = "general"
    elif any(w in text for w in ["casual", "navigare", "browsing", "web", "internet", "acasa"]):
        scop = "casual"

    # Detectare buget  
    buget = 0
    patterns = [
        r'(\d[\d.]*)\s*(?:lei|ron)',
        r'~\s*(\d[\d.]*)',
        r'circa\s+(\d[\d.]*)',
        r'pana\s+la\s+(\d[\d.]*)',
        r'maxim\s+(\d[\d.]*)',
        r'(\d{3,5})\s*(?:de\s+)?(?:lei|ron)?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num_str = match.group(1).replace('.', '').replace(',', '.')
            try:
                val = float(num_str)
                if val > 100:
                    buget = val
                    break
            except Exception:
                pass

    if buget < 100:
        buget = 0

    # Clarificare  
    clarificare = ""
    if buget == 0:
        clarificare = f"What budget do you have in mind for {_cat_en(categorie)} (in lei)?"

    return json.dumps({
        "categorie": categorie,
        "scop": scop,
        "buget": buget,
        "brand": brand,
        "clarificare": clarificare
    }, ensure_ascii=False)


def extract_intent(user_message: str, conversation_history: list) -> dict:
    """
    Extrage intenția structurată din mesajul utilizatorului.
    Returnează un dict cu: categorie, scop, buget, brand, clarificare
    """
    fallback_json = _simple_nlp_fallback(user_message)
    intent = json.loads(fallback_json)
    if "brand" not in intent:
        intent["brand"] = ""
    return intent


def _simple_presentation(products: list, user_request: dict) -> str:
    """Simple English product presentation."""
    scop = _scop_en(user_request.get("scop", "your needs"))
    categorie = _cat_en(user_request.get("categorie", "product"))
    brand = user_request.get("brand", "")

    brand_str = f" {brand.upper()}" if brand else ""
    lines = [f"\n🎯 Found {len(products)} {categorie}{brand_str} for {scop}:\n"]

    for p in products[:3]:
        lines.append(f"  #{p['rank']} {p['product_name'][:55]}")
        lines.append(f"     💰 {p['price']} | 🎯 Match: {p['score_pct']}")
        specs = p.get("specs", {})
        if specs:
            spec_str = " | ".join([f"{k}: {v}" for k, v in specs.items()
                                    if str(v) not in ('N/A', 'nan', 'None', 'nan GB', 'nan Hz', 'nan ms', 'nan ore', 'nan W', 'nan"')])
            if spec_str:
                lines.append(f"     🔧 {spec_str}")
        lines.append("")

    lines.append("Would you like more details about any of these, or should I refine the search?")
    return "\n".join(lines)


def generate_presentation(products: list, user_request: dict, conversation_history: list) -> str:
    """
    Generates the product recommendation presentation.
    """
    if not products or "error" in products[0]:
        return f"Sorry, {products[0].get('error', 'no matching products found')}. Could you rephrase your request?"

    return _simple_presentation(products, user_request)


# CONVERSATIONAL INTERFACE  

class BridgeAI:
    def __init__(self):
        self.conversation_history = []
        self.last_request = {}
        self.last_results = []

        print("\n" + "="*60)
        print("  🤖 BridgeAI — Your personal tech advisor")
        print("="*60)
        print("Hi! I'm BridgeAI, here to help you find the perfect device.")
        print("Tell me what you're looking for (e.g. 'gaming laptop ~4000 lei')")
        print("─"*60)
        print("Special commands: 'exit' to quit, 'reset' for a new conversation")
        print("─"*60 + "\n")

    def chat(self, user_message: str) -> str:
        if user_message.lower() in ("exit", "quit", "bye", "pa"):
            return "Goodbye! Hope I was helpful. 👋"

        if user_message.lower() == "reset":
            self.conversation_history = []
            self.last_request = {}
            self.last_results = []
            return "Conversation reset. How can I help you?"

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Step 1: Extract intent (NLP)  
        print("🔍 Analysing request...")
        intent = extract_intent(user_message, self.conversation_history[:-1])

        print(f"   Detected: {intent}")

        if intent.get("clarificare"):
            response = intent["clarificare"]
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        if intent.get("buget", 0) == 0:
            clarify_q = f"What budget do you have in mind for {_cat_en(intent.get('categorie', 'device'))} (in lei)?"
            self.conversation_history.append({"role": "assistant", "content": clarify_q})
            return clarify_q

        self.last_request = intent

        # Step 2: RF scoring  
        print("🌲 Random Forest scoring products...")
        products = score_products(intent, top_n=5)
        self.last_results = products

        # Step 3: Presentation  
        print("✍️  Generating recommendations...\n")
        response = generate_presentation(products, intent, self.conversation_history[:-1])

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def run(self):
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue

                response = self.chat(user_input)
                print(f"\n🤖 BridgeAI: {response}")

                if user_input.lower() in ("exit", "quit", "bye", "pa"):
                    break

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    ai = BridgeAI()
    ai.run()