import re
import json
import fitz

def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text
def clean_text(text):
    text = text.replace("\u200b", "")

    lines = text.split("\n")
    cleaned_lines = []

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Nếu mở ngoặc nhưng chưa đóng ngoặc
        if "(" in line and ")" not in line:
            while i + 1 < len(lines) and ")" not in line:
                i += 1
                line += " " + lines[i].strip()

        cleaned_lines.append(line)
        i += 1

    return "\n".join(cleaned_lines)
def save_chunk(chunks, disease, section, buffer):
    if disease and section and buffer:
        chunks.append({
            "disease": disease,
            "section": section,
            "content": " ".join(buffer)
        })
if __name__ == "__main__":
    text = extract_text_from_pdf("Data\\Bệnh cây và cách chữa.pdf")
    text = clean_text(text)

    section_pattern = r"^●\s*(.*?):\s*(.*)"
    disease_pattern = r"^\d+\.|^Bệnh"
    text = re.sub(r"\.\s*(\d+\.)", ".\n\\1", text)

    lines = text.split("\n")

    chunks = []
    current_disease = None
    current_section = None
    buffer = []

    for line in lines:
        line = line.strip()

        if not line:
            continue


        # Trường hợp 1: Gặp Disease mới
        if re.match(disease_pattern, line):
            save_chunk(
                chunks,
                current_disease,
                current_section,
                buffer
            )

            current_disease = line
            current_section = None
            buffer = []

            continue


        # Trường hợp 2: Gặp Section mới
        section_match = re.match(section_pattern, line)

        if section_match:
            save_chunk(
                chunks,
                current_disease,
                current_section,
                buffer
            )

            current_section = section_match.group(1)

            buffer = []

            first_content = section_match.group(2)

            if first_content:
                buffer.append(first_content)

            continue


        # Trường hợp 3: Nội dung bình thường
        if current_section:
            buffer.append(line)


    # Lưu chunk cuối cùng sau khi kết thúc vòng for
    save_chunk(
        chunks,
        current_disease,
        current_section,
        buffer
    )


    # Test kết quả
    for chunk in chunks:
        print(chunk)
        print("-" * 50)
with open("Data/section_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)
print("Save succesfully")