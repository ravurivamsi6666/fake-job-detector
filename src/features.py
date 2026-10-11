import html
import re

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    text = re.sub(r"https?://\S+|www\.\S+|#URL_\w+#", " urltoken ", text)
    text = re.sub(r"\S+@\S+\.\S+|#EMAIL_\w+#", " emailtoken ", text)
    text = re.sub(r"#PHONE_\w+#", " phonetoken ", text)
    text = re.sub(r"[$₹£€]\s?\d[\d,]*", " moneytoken ", text)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_text(posting):
    raw = " ".join(str(posting.get(field) or "") for field in TEXT_FIELDS)
    return clean_text(raw)


if __name__ == "__main__":
    sample = {
        "title": "Data Entry Clerk",
        "description": "Earn $500 per week!! Email jobs@quickcash.com NOW",
        "company_profile": None,
    }
    print(build_text(sample))