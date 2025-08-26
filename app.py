import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Zwende Nameplates Chat", page_icon="🛍️", layout="centered")

st.markdown(
    """
    <style>
    .small-note { font-size: 0.86rem; opacity: 0.7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load KB (CSV)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_kb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize helper cols
    df["__product_norm"] = df["Product Name"].str.lower().str.strip()
    df["__variant_norm"] = df["Variant Option"].fillna("Default").astype(str).str.lower().str.strip()
    df["Customization Options"] = df["Customization Options"].fillna("")
    # Nice display name for variants (hide 'Default')
    df["Variant Display"] = df["Variant Option"].apply(lambda x: "" if str(x).lower()=="default" else str(x))
    return df

# Try local CSV; if not present, allow upload
DEFAULT_CSV = "zwende_nameplates_flat.csv"
try:
    kb_df = load_kb(DEFAULT_CSV)
    st.session_state["_kb_loaded_from"] = "repo file"
except Exception:
    st.warning("Upload your `zwende_nameplates_flat.csv` to start.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        kb_df = load_kb(up)
        st.session_state["_kb_loaded_from"] = "uploaded"
    else:
        st.stop()

# -----------------------------
# Intent & utils
# -----------------------------
INTENTS = {
    "pricing": ["price", "cost", "how much", "rate", "₹", "rs", "budget"],
    "availability": ["available", "in stock", "out of stock", "stock", "have this"],
    "customization": ["customize", "customisation", "personalize", "personalise", "add name", "font", "color", "colour", "image upload", "text", "engrave"],
    "shipping": ["ship", "shipping", "delivery time", "lead time", "arrive", "how long"],
    "similar": ["similar", "alternatives", "more like this", "recommend"],
    "order_support": ["order id", "tracking", "where is my order", "shipped", "delivered", "parcel"],
    "discounts": ["discount", "coupon", "offer", "deal"],
}

SIZE_RE = re.compile(r"\b\d+(\.\d+)?\s*x\s*\d+(\.\d+)?\s*(in|inch|inches)?\b")
MONEY_RE = re.compile(r"(?:₹|rs\.?\s*|inr\s*)?(\d{3,6})(?!\d)", re.IGNORECASE)

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()

def detect_intent(msg: str) -> str:
    m = normalize(msg)
    best_intent = "fallback"
    best_hits = 0
    for intent, kws in INTENTS.items():
        hits = sum(1 for k in kws if k in m)
        if hits > best_hits:
            best_hits = hits
            best_intent = intent
    return best_intent if best_hits > 0 else "fallback"

def extract_variant(msg: str) -> Optional[str]:
    m = SIZE_RE.search(msg.lower())
    if m:
        # Return normalized pattern like "8x15 inches" if present
        return m.group(0).replace("inch", "inches").replace("in", "in").strip()
    return None

def extract_budget(msg: str) -> Optional[int]:
    # Returns single budget number; supports "under 2000", "budget 1800", "rs 1500"
    m = MONEY_RE.findall(msg)
    if not m:
        return None
    # pick the first numeric capture
    try:
        return int(m[0] if isinstance(m[0], str) else m[0][0])
    except Exception:
        return None

def fuzzy_match_product(query: str, choices: List[str], limit: int = 5) -> List[Tuple[str, int]]:
    # Returns [(choice, score), ...] by token sort ratio
    return process.extract(
        query, choices, scorer=fuzz.WRatio, limit=limit
    )

def list_variants_with_prices(df: pd.DataFrame, product_name: str) -> List[str]:
    sub = df[df["Product Name"] == product_name]
    lines = []
    for _, r in sub.iterrows():
        v = r["Variant Display"] or "Default"
        price = r["Price"]
        lines.append(f"- {v} → ₹{int(price) if float(price).is_integer() else price}")
    # collapse duplicates
    return sorted(list(set(lines)))

def product_has_multiple_variants(df: pd.DataFrame, product_name: str) -> bool:
    sub = df[df["Product Name"] == product_name]
    return sub["Variant Display"].nunique() > 1

def get_product_row(df: pd.DataFrame, product_name: str, variant_norm: Optional[str]) -> Optional[pd.Series]:
    sub = df[df["Product Name"] == product_name]
    if sub.empty:
        return None
    if variant_norm:
        # try exact variant
        cand = sub[sub["__variant_norm"].str.contains(variant_norm)]
        if cand.empty:
            # try contains numeric pattern only
            cand = sub[sub["__variant_norm"].str.contains(re.escape(variant_norm.split()[0]), regex=True)]
        if not cand.empty:
            return cand.iloc[0]
    # fallback: single-variant or first row
    return sub.iloc[0]

def suggest_by_budget_and_size(df: pd.DataFrame, budget: Optional[int], size_token: Optional[str], k: int = 5) -> pd.DataFrame:
    sub = df.copy()
    if budget:
        sub = sub[sub["Price"] <= budget]
    if size_token:
        size_t = size_token.replace("inches", "in")
        sub = sub[sub["__variant_norm"].str.contains(size_t.split()[0], regex=True, na=False)]
    # deduplicate by product+variant row
    return sub.sort_values(by="Price").head(k)

def recommend_similar(df: pd.DataFrame, base_row: pd.Series, k: int = 3) -> pd.DataFrame:
    price = base_row["Price"]
    lo, hi = price * 0.8, price * 1.2
    sub = df[(df["Price"] >= lo) & (df["Price"] <= hi)]
    # simple keyword overlap from product name
    base_words = set(normalize(base_row["Product Name"]).split())
    def score(row):
        words = set(normalize(row["Product Name"]).split())
        return len(base_words & words)
    sub["__sim"] = sub.apply(score, axis=1)
    sub = sub[sub["Product Name"] != base_row["Product Name"]]
    return sub.sort_values(by=["__sim","Price"], ascending=[False,True]).head(k).drop(columns="__sim")

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ctx" not in st.session_state:
    st.session_state.ctx = {"product": None, "variant": None, "budget": None, "last_intent": None}

# -----------------------------
# Sidebar (helper)
# -----------------------------
with st.sidebar:
    st.header("Zwende Agent (Nameplates)")
    st.caption(f"KB loaded from: **{st.session_state.get('_kb_loaded_from','file')}**")
    st.write("**Products loaded:**", int(kb_df["Product Name"].nunique()))
    st.write("**Variants loaded:**", len(kb_df))
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.ctx = {"product": None, "variant": None, "budget": None, "last_intent": None}
        st.experimental_rerun()
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="small-note">Tip: ask “price of pebble art nameboard”, “available sizes?”, “shipping time for acrylic designer”, “show similar”, “budget 1800 10x15”.</div>', unsafe_allow_html=True)

# -----------------------------
# Chat UI
# -----------------------------
st.title("🛍️ Zwende Nameplates — Sales & Support Agent")

# System greet (only once)
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({"role": "assistant",
                                      "content": "Hi! I’m your Zwende Nameplates assistant. Ask me about price, availability, customization, shipping, similar items, or order support. 😊"})

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user = st.chat_input("Type your message…")
if not user:
    st.stop()

# Append user msg
st.session_state.messages.append({"role":"user","content":user})

# -----------------------------
# Agent Logic
# -----------------------------
msg = user
norm_msg = normalize(msg)
ctx = st.session_state.ctx
intent = detect_intent(norm_msg)
ctx["last_intent"] = intent

# Try entity extraction
variant_token = extract_variant(norm_msg)
budget = extract_budget(norm_msg)
if budget: ctx["budget"] = budget
if variant_token: ctx["variant"] = variant_token

# Fuzzy match product from message words (> 85 score ideal)
product_choices = kb_df["Product Name"].unique().tolist()
best_matches = fuzzy_match_product(norm_msg, product_choices, limit=3)
product_match = None
for name, score, _ in best_matches:
    if score >= 85:
        product_match = name
        break
# If none high-confidence, keep top suggestions for later
suggestions = [name for (name, score, _) in best_matches if score >= 60]

# Update ctx if we found a confident product
if product_match:
    ctx["product"] = product_match

def reply(text: str):
    st.session_state.messages.append({"role":"assistant","content":text})
    with st.chat_message("assistant"):
        st.markdown(text)

# -----------------------------
# Intent Handling
# -----------------------------
if intent == "pricing":
    if not ctx["product"]:
        # ask budget + show suggestions if any
        sug_text = ""
        if suggestions:
            sug_text = "\n\nHere are possible matches:\n- " + "\n- ".join(suggestions)
        ask = "Do you have a budget in mind (e.g., ₹1800)? I can suggest the best options."
        if ctx["budget"]:
            # Suggest by budget immediately
            rec = suggest_by_budget_and_size(kb_df, ctx["budget"], ctx["variant"])
            if rec.empty:
                reply(f"I didn’t find options within ₹{ctx['budget']}. Could you try a different budget or share a product name/URL?{sug_text}")
            else:
                lines = []
                for _, r in rec.iterrows():
                    vn = r["Variant Display"] or "Default"
                    lines.append(f"- **{r['Product Name']}** ({vn}) → ₹{int(r['Price']) if float(r['Price']).is_integer() else r['Price']}  \n  {r['Product URL']}")
                reply(f"Here are options within your budget ₹{ctx['budget']}" + (f" and size `{ctx['variant']}`" if ctx["variant"] else "") + ":\n" + "\n".join(lines) + "\n\nWould you like similar options or a different size?")
        else:
            reply(ask + sug_text)
    else:
        # Have a product
        has_multi = product_has_multiple_variants(kb_df, ctx["product"])
        if has_multi and not ctx["variant"]:
            variants = list_variants_with_prices(kb_df, ctx["product"])
            reply(f"**{ctx['product']}** has these options:\n" + "\n".join(variants) + "\n\nWhich one would you like?")
        else:
            row = get_product_row(kb_df, ctx["product"], normalize(ctx["variant"]) if ctx["variant"] else None)
            if row is None:
                reply("I couldn’t find that exact variant. Would you like to pick from available options?")
            else:
                vn = row["Variant Display"] or "Default"
                price = row["Price"]
                reply(f"**Price:** ₹{int(price) if float(price).is_integer() else price} ({vn})  \n{row['Product URL']}\n\nWant me to suggest similar items or add this to your shortlist?")

elif intent == "availability":
    if not ctx["product"]:
        if suggestions:
            reply("Which product do you mean?\n- " + "\n- ".join(suggestions))
        else:
            reply("Please share the product name or URL, and I’ll check availability.")
    else:
        has_multi = product_has_multiple_variants(kb_df, ctx["product"])
        if has_multi and not ctx["variant"]:
            variants = list_variants_with_prices(kb_df, ctx["product"])
            reply(f"**{ctx['product']}** has multiple options. Please pick one:\n" + "\n".join(variants))
        else:
            row = get_product_row(kb_df, ctx["product"], normalize(ctx["variant"]) if ctx["variant"] else None)
            if row is None:
                reply("I couldn’t find that exact variant. Want to see available options?")
            else:
                avail = row["Availability"]
                vn = row["Variant Display"] or "Default"
                if str(avail).lower() == "instock":
                    reply(f"**Availability:** In Stock ({vn})  \n{row['Product URL']}\n\nWould you like similar options?")
                else:
                    # suggest nearest alternatives
                    alts = kb_df[kb_df["Product Name"] == ctx["product"]]
                    alts = alts[alts["Availability"].str.lower() == "instock"]
                    if alts.empty:
                        reply("That variant is currently out of stock. Would you like me to suggest similar products?")
                    else:
                        lines = []
                        for _, r in alts.iterrows():
                            v = r["Variant Display"] or "Default"
                            lines.append(f"- {v} → ₹{int(r['Price']) if float(r['Price']).is_integer() else r['Price']}")
                        reply("That variant is out of stock. Available options:\n" + "\n".join(lines))

elif intent == "customization":
    if not ctx["product"]:
        reply("You can typically personalize nameplates with **Text, Font choices, and Image Upload**. Share the product name/URL for exact options.")
    else:
        sub = kb_df[kb_df["Product Name"] == ctx["product"]]
        opts = ", ".join(sorted(set(x for x in sub["Customization Options"].astype(str).tolist() if x)))
        reply(f"**Customization options for {ctx['product']}:** {opts or 'Text, Font, Image Upload'}\n\nWant me to configure one for you?")

elif intent == "shipping":
    if not ctx["product"]:
        # generic
        mins = int(kb_df["Shipping Min (days)"].min())
        maxs = int(kb_df["Shipping Max (days)"].max())
        reply(f"Most nameplates ship in **{mins}–{maxs} days** depending on design & location. Share a product for exact estimate.")
    else:
        row = get_product_row(kb_df, ctx["product"], normalize(ctx["variant"]) if ctx["variant"] else None)
        if row is None:
            reply("Share a variant or let me show the available options to estimate shipping.")
        else:
            mn = int(row["Shipping Min (days)"])
            mx = int(row["Shipping Max (days)"])
            reply(f"**Estimated shipping:** {mn}–{mx} days.  \n{row['Product URL']}\n\nWould you like help placing an order or seeing similar products?")

elif intent == "similar":
    if not ctx["product"]:
        # use budget if present
        if ctx["budget"]:
            rec = suggest_by_budget_and_size(kb_df, ctx["budget"], ctx["variant"], k=5)
            if rec.empty:
                reply(f"I couldn’t find similar items within ₹{ctx['budget']}. Try changing budget or share a product name.")
            else:
                lines = []
                for _, r in rec.head(3).iterrows():
                    vn = r["Variant Display"] or "Default"
                    lines.append(f"- **{r['Product Name']}** ({vn}) → ₹{int(r['Price']) if float(r['Price']).is_integer() else r['Price']}  \n  {r['Product URL']}")
                reply("Here are some options you might like:\n" + "\n".join(lines))
        else:
            reply("Share a product you like, and I’ll recommend similar items.")
    else:
        base = get_product_row(kb_df, ctx["product"], normalize(ctx["variant"]) if ctx["variant"] else None)
        if base is None:
            reply("Tell me the variant (if any), and I’ll suggest similar items.")
        else:
            sim = recommend_similar(kb_df, base, k=3)
            if sim.empty:
                reply("I couldn’t find close matches. Want me to search by budget?")
            else:
                lines = []
                for _, r in sim.iterrows():
                    vn = r["Variant Display"] or "Default"
                    lines.append(f"- **{r['Product Name']}** ({vn}) → ₹{int(r['Price']) if float(r['Price']).is_integer() else r['Price']}  \n  {r['Product URL']}")
                reply("You may also like:\n" + "\n".join(lines))

elif intent == "order_support":
    reply("To track your order, please share your **Order ID**. You can also use the tracking link sent to your email/SMS. If it still doesn’t resolve, I can connect you to a human agent. 🙌")

elif intent == "discounts":
    reply("I don’t see an active discount code right now. Would you like me to show options **under a specific budget** (e.g., ₹2000)?")

else:
    # fallback / clarify
    maybe = ("pricing", "availability", "customization", "shipping", "similar", "order_support", "discounts")
    reply("I didn’t quite get that. You can ask about **price, availability, customization, shipping, similar items, discounts, or order support**.\nTry: *“price of pebble art nameboard 8x15”*")

