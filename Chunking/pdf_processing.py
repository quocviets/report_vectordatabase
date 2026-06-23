import fitz
import json


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()
    return text


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def save_chunks(chunks, output_path):
    data = []

    for i, chunk in enumerate(chunks):
        data.append({
            "id": i,
            "text": chunk
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    pdf_path = "Data/Bệnh cây và cách chữa.pdf"

    text = extract_text_from_pdf(pdf_path)

    chunks = split_text(text)

    save_chunks(
        chunks,
        "Output/chunks.json"
    )

    print(f"Total chunks: {len(chunks)}")