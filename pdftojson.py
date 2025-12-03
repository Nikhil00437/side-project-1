import json
import pdfplumber

pdf_path = "D:\projects\FreeGround\exam_prep\ilovepdf_merged(2).pdf"
json_path = "D:\projects\FreeGround\exam_prep\json\exam_material.json"
def pdf_to_json(pdf_path, json_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            data = {"pages": []}

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                data["pages"].append({
                    "page_number": i + 1,
                    "text": text.strip()
                })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Converted PDF to JSON successfully: {json_path}")

    except Exception as e:
        print("Bro… seriously? Something went wrong.")
        print(e)


# Example usage
if __name__ == "__main__":
    pdf_to_json(pdf_path, json_path)
