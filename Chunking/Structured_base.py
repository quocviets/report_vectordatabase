import re
import fitz
import json

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
def structure_based_chunking(text):
    pattern = r"\d+\.\s*Bệnh"
    matches = list(re.finditer(pattern, text))

    chunks = []

    for i, match in enumerate(matches):
        start = match.start()

        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(text)

        chunk = text[start:end].strip()

        chunks.append({
            "id": i,
            "text": chunk
        })

    return chunks


def save_chunks(chunks, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=4
        )


if __name__ == "__main__":
    pdf_path = "Data/Bệnh cây và cách chữa.pdf"

    text = extract_text_from_pdf(pdf_path)
    text = re.sub(r"[\u200b-\u200d\uFEFF]", "", text)
    print("===== Preview =====")
    print(text[:1000])

    chunks = structure_based_chunking(text)

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print("=" * 50)
        print(chunk["text"][:200])

    save_chunks(
        chunks,
        "Data/structure_chunks.json"
    )