import pandas as pd


large_propmt = """
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



propmt = """
    You are an expert museum specimen label transcription system.
    Carefully read the date and location of the image and extract the requested metadata.
    
    The image contains multiple label cards. You must:
    - READ: The small handwritten or printed card showing the collection DATE and LOCATION/LOCALITY.
    - IGNORE: Any card showing species names (e.g. "A. subte raneus"), collector names (e.g. "Coll. Rosenberg"), museum logos, barcodes, or color calibration cards.
    - The date and locality are typically on the SAME small card together.
    - If a card contains a latin species name, ignore it.
    - "Dania" is NOT a locality — omit Dania and the name following "Dania" from the answer.
    
    Rules:
    1. Return ONLY valid JSON.
    2. Preserve original spelling EXACTLY as written — do not correct, expand, or normalize anything.
    3. If a field is unreadable or missing, output "MISSING".
    4. Confidence must be a decimal number between 0 and 1.0.
    5. Higher confidence means higher certainty from visual evidence.
    6. Never hallucinate values not visible in the image.
    7. Preserve Scandinavian and special Nordic characters exactly when visible (æ, ø, å, ä, ö). Use standard English characters otherwise.
    8. please preserve capitalization
    
    
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
    "You are a character-level transcription assistant. 
    I will provide a locality name where every character is separated by a space, and words are separated by three spaces. 
    for example, 'Nord skov' is represented as 'N o r d   s k o v'.
    You must output the corrected locality name using the exact same atomized format. Do not add any extra words or conversational text. 
    your input is a locality that might be wrong (e.g. mis-spelled, etc.), and your task is to examine the image and fixed it.
    the locality for you to confirm or fix is:
    """,
    """
    
    Instructions:
    1. Compare the 'Input Locality' against the text visible in the image.
    2. If the 'Input Locality' is a typo, correct it strictly based on the image.
    3. Maintain original capitalization and special characters exactly.
    4. If the 'Input Locality' is correct, output it exactly as provided.
    5. If the extracted text is 'MISSING', look beyond the card for handwritten text.
    6. "Dania" is NOT a locality — omit Dania and the name following "Dania" from the answer
    7. Preserve Scandinavian and special Nordic characters exactly when visible (æ, ø, å, ä, ö). Use standard English characters otherwise

Final Output: in json style, {'locality': locality}
e.g. {'locality': 'N o r d   s k o v'}
    """
    ]
"""
# Task: You are an expert curator.
# Input Locality: {extracted_string}
# Image: [Image tokens]
# Instructions:

# promt_fix = """
#     You are an expert museum specimen label transcription system.
#     Carefully read the location of the image.
#
#     The image contains multiple label cards. You must:
#     - READ: The small handwritten or printed card showing (among other) the collection LOCATION/LOCALITY.
#     - IGNORE: Any card showing species names (e.g. "A. subte raneus"), collector names (e.g. "Coll. Rosenberg"), museum logos, barcodes, or color calibration cards.
#     - If a card says "Coll." or "det." or "Tilg." followed by a name or date, ignore that card entirely.
#     - If a card contains a latin species name, ignore it.
#     - most cards includes collection date as well, ignore the date, return only the locality(ies)
#     - "Dania" is NOT a locality — it refers to the museum collection. Output "MISSING" for locality if only "Dania" appears.
#     - Phrases meaning "in cow dung" (e.g. "i kogøding", "i kogjorning") are NOT locality — ignore them.
#     - If there are MULTIPLE label cards with locality info, transcribe ALL of them separated by " | " (pipe). "Ti | Kulhuset Jægerspris""
#     - Even if locality repeats across cards, transcribe both: "10.6.1951 | 10.6.1951"
#
#     Rules:
#     Do not include markdown.
#     Preserve original spelling EXACTLY as written — do not correct, expand, or normalize anything.
#     Preserve Scandinavian and special Nordic characters exactly when visible (æ, ø, å, ä, ö). Use standard English characters otherwise.
#     preserve EXACTLY as written including abbreviations (e.g. "Kb" stays "Kb", not "København").
#
#     your final output should be the locality.
#     """

def get_spell_fix_messages(img_path, input_loc, gt_loc=None):
    # This structure is consistent for both training and inference
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": promt_fix[0] + '\n' + input_loc + '\n' + promt_fix[-1]}
            #{"type": "text", "text": promt_fix}
        ]}
    ]
    if gt_loc:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": gt_loc}]})
    return {"messages" : messages}



promt_missing = """
    You are an expert museum specimen label transcription system.
    Carefully read the date and location of the image and extract the requested metadata.
    
    The image contains a specimen, on a paper. a Date and/or locality may apear on that card.
    If apear, it may be either handwritten or typed.
    Your task is to retrieve the date and locality of the speciman, if they apear.

    Rules:
    1. Do not include markdown.
    2. Do not explain your reasoning.
    3. Preserve original spelling EXACTLY as written — do not correct, expand, or normalize anything.
    4. If a field is unreadable or missing, output "MISSING".
    5. Never hallucinate values not visible in the image.
    6. Preserve Scandinavian and special Nordic characters exactly when visible (æ, ø, å, ä, ö). Use standard English characters otherwise.
    7. For dates: preserve the EXACT format as written. Do NOT expand abbreviations (e.g. "Septmbr" stays "Septmbr", not "September"). Do NOT convert Roman numerals (e.g. "IV" stays "IV"). Do NOT reformat or normalize dates.
    8. For locality: preserve EXACTLY as written including abbreviations (e.g. "Kb" stays "Kb", not "København").
    9 in some cases, there are multiple localities and dates. in these cases, please retrive all.
    9. preserve capitalization
    
    Required JSON schema:
    {
      "verbatimDate": "string",
      "verbatimLocality": "string",
    }
    
    Examples:
    {"verbatimDate": "22.5.1977", "verbatimDate_confidence": 0.98, "verbatimLocality": "Svinø strand", "verbatimLocality_confidence": 0.95}
    {"verbatimDate": "MISSING", "verbatimDate_confidence": 1.0, "verbatimLocality": "MISSING", "verbatimLocality_confidence": 1.0}
    {"verbatimDate": "22 VIII 2027", "verbatimDate_confidence": 1.0, "verbatimLocality": "Evæglion", "verbatimLocality_confidence": 1.0}
    example for multiple dates and localities:
    {"verbatimDate": "10.6.1951 | 10.6.1951", "verbatimDate_confidence": 0.98, "verbatimLocality": "Rotholme Jyll. | Rotholme Jyll.", "verbatimLocality_confidence": 0.98}
"""


def get_missing_fill_messages(img_path, gt_loc=None, gt_date=None):
    # This structure is consistent for both training and inference
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": promt_missing}
        ]}
    ]
    if gt_loc and gt_date:
        out_text = '{"verbatimDate": "' + gt_date + '", "verbatimDate_confidence": 0.95, "verbatimLocality": "' + gt_loc + '", "verbatimLocality_confidence": 0.95}'
        messages.append({"role": "assistant", "content": [{"type": "text", "text": out_text}]})
    return {"messages" : messages}
