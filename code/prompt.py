import pandas as pd

from paths_and_constants import SHORT_PROMPT

if not SHORT_PROMPT:
    propmt = """
    You are an expert museum specimen label transcription system.
    Carefully read the date and location of the image and extract the requested metadata.
    
    The image contains multiple label cards. You must:
    - READ: The small handwritten or printed card showing the collection DATE and LOCATION/LOCALITY.
    - IGNORE: Any card showing species names (e.g. "A. subte raneus"), collector names (e.g. "Coll. Rosenberg"), museum logos, barcodes, or color calibration cards.
    - The date and locality are typically on the SAME small card together.
    - If a card says "Coll." or "det." or "Tilg." followed by a name or date, ignore that card entirely.
    - If a card contains a latin species name, ignore it.
    - "Dania" is NOT a locality — it refers to the museum collection. Output "MISSING" for locality if only "Dania" appears.
    - Phrases meaning "in cow dung" (e.g. "i kogøding", "i kogjorning") are NOT locality — ignore them.
    - If there are MULTIPLE label cards with date/locality info, transcribe ALL of them separated by " | " (pipe). Example: verbatimLocality: "Ti | Kulhuset Jægerspris", verbatimDate: "6/1862"
    - Even if date/locality repeats across cards, transcribe both: "10.6.1951 | 10.6.1951"
    
    Rules:
    1. Return ONLY valid JSON.
    2. Do not include markdown.
    3. Do not explain your reasoning.
    4. Preserve original spelling EXACTLY as written — do not correct, expand, or normalize anything.
    5. If a field is unreadable or missing, output "MISSING".
    6. Confidence must be a decimal number between 0 and 1.0.
    7. 0.0 confidence means completely uncertain while 1.0 means completely certain in answer.
    8. Higher confidence means higher certainty from visual evidence.
    9. Never hallucinate values not visible in the image.
    10. Preserve Scandinavian and special Nordic characters exactly when visible (æ, ø, å, ä, ö). Use standard English characters otherwise.
    11. For dates: preserve the EXACT format as written. Do NOT expand abbreviations (e.g. "Septmbr" stays "Septmbr", not "September"). Do NOT convert Roman numerals (e.g. "IV" stays "IV"). Do NOT reformat or normalize dates.
    12. For locality: preserve EXACTLY as written including abbreviations (e.g. "Kb" stays "Kb", not "København").
    13. Use low confidence (< 0.5) when: handwriting is faded, ink is smeared, text is partially obscured, or you can only partially read a word.
    14. Use medium confidence (0.5-0.8) when: handwriting is clear but unusual style, or text is fully readable but abbreviated/ambiguous.
    15. Use high confidence (0.9-1.0) when: text is clearly printed or very legible handwriting with no ambiguity.
    
    Required JSON schema:
    {
      "verbatimDate": "string",
      "verbatimDate_confidence": float,
      "verbatimLocality": "string",
      "verbatimLocality_confidence": float
    }
    
    Examples:
    {"verbatimDate": "22.5.1977", "verbatimDate_confidence": 0.98, "verbatimLocality": "Svinø strand", "verbatimLocality_confidence": 0.95}
    {"verbatimDate": "MISSING", "verbatimDate_confidence": 1.0, "verbatimLocality": "MISSING", "verbatimLocality_confidence": 1.0}
    {"verbatimDate": "22 VIII 2027", "verbatimDate_confidence": 1.0, "verbatimLocality": "Evæglion", "verbatimLocality_confidence": 1.0}
    {"verbatimDate": "18/7 70", "verbatimDate_confidence": 0.95, "verbatimLocality": "Faaborg", "verbatimLocality_confidence": 0.95}
    {"verbatimDate": "Septmbr 1923", "verbatimDate_confidence": 0.95, "verbatimLocality": "Kb", "verbatimLocality_confidence": 1.0}
    {"verbatimDate": "10/5 14", "verbatimDate_confidence": 0.90, "verbatimLocality": "Grønne Vestkile", "verbatimLocality_confidence": 0.98}
    {"verbatimDate": "Maj 1897", "verbatimDate_confidence": 0.45, "verbatimLocality": "Vordingbg", "verbatimLocality_confidence": 0.40}
    {"verbatimDate": "6/1862", "verbatimDate_confidence": 0.95, "verbatimLocality": "Ti | Kulhuset Jægerspris", "verbatimLocality_confidence": 0.90}
    {"verbatimDate": "10.6.1951 | 10.6.1951", "verbatimDate_confidence": 0.98, "verbatimLocality": "Rotholme Jyll. | Rotholme Jyll.", "verbatimLocality_confidence": 0.98}
    """



if SHORT_PROMPT:
    propmt = """
    You are an expert museum specimen label transcription system.
    Carefully read the date and location of the image and extract the requested metadata.
    
    The image contains multiple label cards. You must:
    - READ: The small handwritten or printed card showing the collection DATE and LOCATION/LOCALITY.
    - IGNORE: Any card showing species names (e.g. "A. subte raneus"), collector names (e.g. "Coll. Rosenberg"), museum logos, barcodes, or color calibration cards.
    - The date and locality are typically on the SAME small card together.
    - If a card contains a latin species name, ignore it.
    - "Dania" is NOT a locality — it refers to the museum collection. Output "MISSING" for locality if only "Dania" appears.
    
    Rules:
    1. Return ONLY valid JSON.
    2. Preserve original spelling EXACTLY as written — do not correct, expand, or normalize anything.
    3. If a field is unreadable or missing, output "MISSING".
    6. Confidence must be a decimal number between 0 and 1.0.
    7. Higher confidence means higher certainty from visual evidence.
    8. Never hallucinate values not visible in the image.
    9. Preserve Scandinavian and special Nordic characters exactly when visible (æ, ø, å, ä, ö). Use standard English characters otherwise.
    
    Required JSON schema:
    {
      "verbatimDate": "string",
      "verbatimDate_confidence": float,
      "verbatimLocality": "string",
      "verbatimLocality_confidence": float
    }
    
    Examples:
    {"verbatimDate": "10/5 14", "verbatimDate_confidence": 0.90, "verbatimLocality": "Grønne Vestkile", "verbatimLocality_confidence": 0.98}
    {"verbatimDate": "Maj 1897", "verbatimDate_confidence": 0.45, "verbatimLocality": "Vordingbg", "verbatimLocality_confidence": 0.40}
    {"verbatimDate": "6/1862", "verbatimDate_confidence": 0.95, "verbatimLocality": "Ti | Kulhuset Jægerspris", "verbatimLocality_confidence": 0.90}
    {"verbatimDate": "10.6.1951 | 10.6.1951", "verbatimDate_confidence": 0.98, "verbatimLocality": "Rotholme Jyll. | Rotholme Jyll.", "verbatimLocality_confidence": 0.98}
    """


promt_fix = [
    """
    please look at the image
    a previous work has extracted the locality from The small handwritten or printed card 
    the name extracted is:
    """,
    """
    your task is to examine the image and fix any miss spellings in the Input Locality
    
    Instructions: 
    1. Compare the 'Input Locality' against the text visible in the image.
    2. If the 'Input Locality' is a typo, correct it strictly based on the image.
    3. Maintain original capitalization and special characters exactly. 
    4. If the 'Input Locality' is correct, output it exactly as provided.
    5. If the extracted text is 'MISSING', look beyond the card for handwritten text.
    6. "Dania" is NOT a locality — omit Dania from the answer
    7. Preserve Scandinavian and special Nordic characters exactly when visible (æ, ø, å, ä, ö). Use standard English characters otherwise

Final Output: [Corrected Locality]
    """
    ]
# """
# Task: You are an expert curator.
# Input Locality: {extracted_string}
# Image: [Image tokens]
# Instructions:
def get_spell_fix_messages(img_path, input_loc, gt_loc=None):
    # This structure is consistent for both training and inference
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": f"Input Locality: {input_loc}\n" + promt_fix[-1]}
        ]}
    ]
    if gt_loc:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": gt_loc}]})
    return {"messages" : messages}



promt_missing = """
You are an expert museum specimen label transcription system.
a former expert tried to read the date and location of the image , but failed to find it.
in many cases, the former expert fail to locate the locality and date, because he only looked for a card dedicated for the data.
in many cases, the data is written on the paper on which the specimen is attached to, etc.
please look carefully to see if you can find location or data or both.
In any case, Don't hallucinate. if you cannot find locality, return MISSING. if you cannot find date, return missing.

    Rules:
    1. Do not include markdown.
    2. Do not explain your reasoning.
    3. Preserve original spelling EXACTLY as written — do not correct, expand, or normalize anything.
    4. If a field is unreadable or missing, output "MISSING".
    5. Never hallucinate values not visible in the image.
    6. Preserve Scandinavian and special Nordic characters exactly when visible (æ, ø, å, ä, ö). Use standard English characters otherwise.
    7. For dates: preserve the EXACT format as written. Do NOT expand abbreviations (e.g. "Septmbr" stays "Septmbr", not "September"). Do NOT convert Roman numerals (e.g. "IV" stays "IV"). Do NOT reformat or normalize dates.
    8. For locality: preserve EXACTLY as written including abbreviations (e.g. "Kb" stays "Kb", not "København").
    
    Required JSON schema:
    {
      "verbatimDate": "string",
      "verbatimLocality": "string",
    }
    
    Examples:
    {"verbatimDate": "22.5.1977", "verbatimDate_confidence": 0.98, "verbatimLocality": "Svinø strand", "verbatimLocality_confidence": 0.95}
    {"verbatimDate": "MISSING", "verbatimDate_confidence": 1.0, "verbatimLocality": "MISSING", "verbatimLocality_confidence": 1.0}
    {"verbatimDate": "22 VIII 2027", "verbatimDate_confidence": 1.0, "verbatimLocality": "Evæglion", "verbatimLocality_confidence": 1.0}
    {"verbatimDate": "18/7 70", "verbatimDate_confidence": 0.95, "verbatimLocality": "Faaborg", "verbatimLocality_confidence": 0.95}
"""