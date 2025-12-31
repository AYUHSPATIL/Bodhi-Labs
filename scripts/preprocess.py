import json
import os
import pandas as pd
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


#Utility functions
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_correct_option(options):
    """
    Returns (correct_option_id, correct_option_text)
    """
    for opt in options:
        if opt.get("answerstatus") == "Correct":
            return opt.get("answerid"), opt.get("optiontext")
    return None, ""  # safe fallback


def extract_tag_fields(tags):
    """
    Extracts stable semantic fields + all tag IDs.
    """
    result = {
        "domain": None,
        "competency_name": None,
        "competency_area": None,
        "competency_definition": None,
        "type": None,
        "tag_ids": []
    }

    for t in tags:
        tag_id = t.get("tagid")
        tag_name = t.get("tagname", "")
        tag_name_lower = tag_name.lower()

        if tag_id:
            result["tag_ids"].append(tag_id)

        if "domain:" in tag_name_lower:
            result["domain"] = tag_name.split(":", 1)[1].strip()

        elif "competency name:" in tag_name_lower:
            result["competency_name"] = tag_name.split(":", 1)[1].strip()

        elif "competency area:" in tag_name_lower:
            result["competency_area"] = tag_name.split(":", 1)[1].strip()

        elif "competency definition:" in tag_name_lower:
            result["competency_definition"] = tag_name.split(":",1)[1].strip()

        elif "type:" in tag_name_lower:
            result["type"] = tag_name.split(":", 1)[1].strip()

    return result


def build_content(question_text, correct_ans, tags):
    """
    Builds a clean, single-string natural language content
    for embeddings. 
    """
    tag_text = ", ".join([t.get("tagname", "") for t in tags])

    return (
        f"Question: {question_text}. "
        f"Correct answer: {correct_ans}. "
        f"Tags: {tag_text}."
    ).strip()



# Main Preprocessing
def preprocess(json_path, output_path):
    logger.info("Starting data preprocessing pipeline.")
    raw = load_json(json_path)

    if not isinstance(raw, list):
        raise ValueError("Expected JSON root to be a list of questions")

    rows = []

    for item in raw:
        question_block = item.get("question", {})
        options = item.get("option", [])
        tags = item.get("tag", [])

        question_id = question_block.get("questionid")
        question_text = question_block.get("questiontext", "")

        if not question_id or not question_text:
            logger.warning("Skipping question due to missing id or text")
            continue

        correct_id, correct_text = get_correct_option(options)
        tag_info = extract_tag_fields(tags)

        rows.append({
            "question_id": question_id,
            "content": build_content(question_text, correct_text, tags),
            "correct_option_id": correct_id,
            **tag_info
        })

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_excel(output_path, index=False)
    
    logger.info(f"Processed {len(rows)} questions successfully")
    logger.info(f"Saved processed data to {output_path}")

    return df



if __name__ == "__main__":
    preprocess(
        json_path="data/raw/questiondetailwithtag.json",
        output_path="data/processed/data_processed.xlsx"
    )

