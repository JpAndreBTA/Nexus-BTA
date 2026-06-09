from __future__ import annotations

import json
import re
from typing import Any


_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def ideogram4_prompt_json_text(value: dict[str, Any], *, compact: bool = True) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":") if compact else None, indent=None if compact else 2)


def parse_ideogram4_prompt_json(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw or not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        return normalize_ideogram4_caption(parsed)
    except ValueError:
        return None


def is_ideogram4_prompt_json(text: str) -> bool:
    return parse_ideogram4_prompt_json(text) is not None


def normalize_ideogram4_caption(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Ideogram 4 JSON prompt must be an object.")
    high_level = _clean_text(value.get("high_level_description"))
    style_value = value.get("style_description")
    comp_value = value.get("compositional_deconstruction")
    if not high_level or not isinstance(comp_value, (dict, list)):
        raise ValueError("Ideogram 4 JSON prompt needs high_level_description and compositional_deconstruction.")
    if not isinstance(style_value, dict):
        style_value = {}

    style: dict[str, Any] = {
        "aesthetics": _clean_text(style_value.get("aesthetics")) or "natural, coherent, detailed",
        "lighting": _clean_text(style_value.get("lighting")) or "natural balanced lighting",
        "photo": _clean_text(style_value.get("photo")) or "clear photo, realistic composition",
        "medium": _clean_text(style_value.get("medium")) or "photograph",
    }
    palette = _palette(style_value.get("color_palette"))
    if palette:
        style["color_palette"] = palette[:16]

    elements = []
    raw_elements = comp_value.get("elements") if isinstance(comp_value, dict) else comp_value
    if isinstance(raw_elements, list):
        for item in raw_elements[:24]:
            element = _normalize_element(item)
            if element:
                elements.append(element)
    if not elements:
        elements.append({"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": high_level})

    return {
        "high_level_description": high_level,
        "style_description": style,
        "compositional_deconstruction": {
            "background": _clean_background(comp_value.get("background") if isinstance(comp_value, dict) else "") or "A coherent background matching the main scene.",
            "elements": elements,
        },
    }


def normalize_ideogram4_magic_caption(value: dict[str, Any], prompt: str, regions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    caption = normalize_ideogram4_caption(value)
    if regions or _prompt_requests_regional_layout(prompt):
        return caption

    comp = caption["compositional_deconstruction"]
    elements = comp.get("elements") if isinstance(comp, dict) else []
    element_text = []
    scene = _clean_text(prompt) or caption["high_level_description"]
    if isinstance(elements, list):
        for item in elements:
            if not isinstance(item, dict):
                continue
            text_value = _clean_text(item.get("text"))
            desc_value = _clean_magic_desc(item.get("desc"))
            if text_value:
                element_text.append(f'text "{text_value}"')
            if desc_value and scene.lower() in desc_value.lower():
                desc_value = re.sub(re.escape(scene), "", desc_value, flags=re.IGNORECASE).strip(" ;:-")
            if desc_value and desc_value.lower() != scene.lower():
                element_text.append(desc_value)
    merged_desc = "; ".join(dict.fromkeys([scene, *element_text]))
    merged_desc = _remove_panel_language(merged_desc)
    caption["high_level_description"] = _remove_panel_language(caption["high_level_description"])
    photo_rule = "wide coherent composition with all requested subjects visible in one shared camera view"
    photo_text = _clean_text(caption["style_description"].get("photo", ""))
    if photo_rule.lower() not in photo_text.lower():
        caption["style_description"]["photo"] = _clean_text(f"{photo_text}; {photo_rule}")
    merged_desc = _normalize_simple_magic_scene(merged_desc, prompt)
    if not _has_concrete_background(comp.get("background")):
        comp["background"] = _simple_scene_background(prompt, caption["high_level_description"])
    counted_elements = _counted_subject_elements(prompt, merged_desc or caption["high_level_description"])
    comp["elements"] = counted_elements or [
        {
            "type": "obj",
            "bbox": [0, 0, 1000, 1000],
            "desc": (
                "Unified full-canvas wide scene with the requested subjects visible in one shared environment: "
                f"{merged_desc or caption['high_level_description']}"
            ),
        }
    ]
    return caption


def build_ideogram4_template_caption(prompt: str, width: int = 1024, height: int = 1024, regions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    scene = _clean_text(prompt) or "A natural coherent scene."
    elements: list[dict[str, Any]] = []
    if not regions:
        subjects = _prompt_subjects(scene)
        if len(subjects) > 1 and _prompt_requests_regional_layout(scene):
            layouts = [
                [260, 40, 860, 620],
                [180, 520, 820, 960],
                [80, 250, 450, 780],
                [520, 220, 940, 840],
            ]
            for index, subject in enumerate(subjects[:4]):
                elements.append(
                    {
                        "type": "obj",
                        "bbox": layouts[index],
                        "desc": f"Clear visible subject: {subject}",
                    }
                )
        else:
            elements.append(
                {
                    "type": "obj",
                    "bbox": [0, 0, 1000, 1000],
                    "desc": f"Unified full-canvas wide action scene with the main subjects visible and interacting in one shared environment: {scene}",
                }
            )
    else:
        elements.append(
            {
                "type": "obj",
                "bbox": [0, 0, 1000, 1000],
                "desc": f"Overall visual direction: {scene}",
            }
        )
        for item in regions[:24]:
            if not isinstance(item, dict):
                continue
            element = _region_to_element(item)
            if element:
                elements.append(element)

    orientation = "square"
    if width > height:
        orientation = "landscape"
    elif height > width:
        orientation = "portrait"
    return {
        "high_level_description": scene,
        "style_description": {
            "aesthetics": f"natural, coherent, detailed, production-ready {orientation} image",
            "lighting": "natural balanced lighting with clear subject separation",
            "photo": "high resolution, realistic surfaces, coherent composition",
            "medium": "photograph",
        },
        "compositional_deconstruction": {
            "background": f"A coherent background and environment that supports the requested scene: {scene}",
            "elements": elements,
        },
    }


def _region_to_element(item: dict[str, Any]) -> dict[str, Any] | None:
    bbox = _bbox_from_region(item)
    if not bbox:
        return None
    desc = _clean_text(item.get("prompt") or item.get("desc"))
    text = _clean_text(item.get("text"))
    palette = _palette(item.get("colors") or ([item.get("color")] if item.get("color") else []))
    if str(item.get("type") or "").lower() == "text" or text:
        element: dict[str, Any] = {"type": "text", "bbox": bbox, "text": text or desc, "desc": desc}
    else:
        element = {"type": "obj", "bbox": bbox, "desc": desc or "Regional visual element."}
    if palette:
        element["color_palette"] = palette[:5]
    return element


def _normalize_element(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    element_type = "text" if str(value.get("type") or "").lower() == "text" or value.get("text") else "obj"
    desc = _clean_text(value.get("desc")) or _clean_text(value.get("description"))
    bbox = _bbox(value.get("bbox"))
    palette = _palette(value.get("color_palette"))
    if element_type == "text":
        text = _clean_text(value.get("text"))
        if not text and not desc:
            return None
        element: dict[str, Any] = {"type": "text"}
        if bbox:
            element["bbox"] = bbox
        element["text"] = text or desc
        element["desc"] = desc
    else:
        if not desc:
            return None
        element = {"type": "obj"}
        if bbox:
            element["bbox"] = bbox
        element["desc"] = desc
    if palette:
        element["color_palette"] = palette[:5]
    return element


def _bbox_from_region(item: dict[str, Any]) -> list[int] | None:
    try:
        x = _unit(item.get("x"))
        y = _unit(item.get("y"))
        w = max(0.01, min(1.0 - x, _unit(item.get("w"), 0.25)))
        h = max(0.01, min(1.0 - y, _unit(item.get("h"), 0.25)))
    except (TypeError, ValueError):
        return None
    return [round(y * 1000), round(x * 1000), round((y + h) * 1000), round((x + w) * 1000)]


def _bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        raw = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if raw and max(raw) <= 100:
        raw = [item * 10 for item in raw]
    y1, x1, y2, x2 = [max(0, min(1000, int(round(item)))) for item in raw]
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 - y1 < 10 or x2 - x1 < 10:
        return None
    return [y1, x1, y2, x2]


def _palette(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.values()
    if not isinstance(value, list) and not isinstance(value, tuple):
        return []
    colors = []
    for item in value:
        raw = str(item or "").strip()
        if not raw or not _HEX_RE.match(raw):
            continue
        colors.append(("#" + raw.lstrip("#")).upper())
    return colors


def _unit(value: Any, default: float = 0.0) -> float:
    if value is None:
        number = default
    else:
        number = float(value)
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_background(value: Any) -> str:
    if isinstance(value, dict):
        return _clean_text(value.get("desc") or value.get("description") or value.get("prompt") or value.get("text"))
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_clean_background(item))
            else:
                parts.append(_clean_text(item))
        return "; ".join(part for part in parts if part)
    return _clean_text(value)


def _prompt_requests_regional_layout(prompt: str) -> bool:
    text = _clean_text(prompt).lower()
    return bool(re.search(r"\b(split[- ]?screen|collage|grid|quadrant|panel|comic panel|diptych|triptych|four panels|two panels|left side|right side|top half|bottom half)\b", text))


def _remove_panel_language(value: str) -> str:
    text = _clean_text(value)
    text = re.sub(r"\b(?:split[- ]?screen|collage|grid|quadrant|panel|four panels|two panels)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return _clean_text(text)


def _clean_magic_desc(value: Any) -> str:
    text = _remove_panel_language(_clean_text(value))
    text = re.sub(r"^single continuous scene,\s*not\s+a\s*(?:or\s*)?:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^single continuous scene,\s*not\s+a\s+.*?:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^single continuous scene:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^unified full-canvas scene:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^unified full-canvas wide scene with the requested subjects visible in one shared environment:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^unified full-canvas wide action scene with the main subjects visible and interacting in one shared environment:\s*", "", text, flags=re.IGNORECASE)
    return _clean_text(text)


def _normalize_simple_magic_scene(desc: str, prompt: str) -> str:
    original = _clean_text(prompt)
    _background_cue, foreground_cue = _scene_depth_cues(original)
    text = _clean_text(desc)
    if foreground_cue:
        text = _clean_text(f"{foreground_cue}; {text}")
    if original and original.lower() not in text.lower():
        text = _clean_text(f"{original}; {text}")
    text = _dedupe_semicolon_phrases(text)
    text = _preserve_requested_count(text, original)
    text = _preserve_requested_text(text, original)
    text = _remove_vague_prompt_expansion(text)
    return _clean_text(text or original)


def _dedupe_semicolon_phrases(value: str) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for part in re.split(r"\s*;\s*", _clean_text(value)):
        clean = _clean_text(part).strip(" .")
        if not clean:
            continue
        key = re.sub(r"\b(?:the|a|an|um|uma|o|a)\b", "", clean.lower())
        key = re.sub(r"[^a-z0-9À-ÿ]+", " ", key).strip()
        if key in seen:
            continue
        seen.add(key)
        parts.append(clean)
    return "; ".join(parts)


def _preserve_requested_count(desc: str, prompt: str) -> str:
    counts = [
        (r"\b(?:2|two|dois|duas)\b", "Exactly two"),
        (r"\b(?:3|three|três|tres)\b", "Exactly three"),
        (r"\b(?:4|four|quatro)\b", "Exactly four"),
        (r"\b(?:5|five|cinco)\b", "Exactly five"),
        (r"\b(?:6|six|seis)\b", "Exactly six"),
    ]
    for pattern, label in counts:
        if re.search(pattern, prompt, flags=re.IGNORECASE) and label.lower() not in desc.lower():
            return _clean_text(f"{label} requested primary subjects or objects, no extra duplicate figures; {desc}")
    if _prompt_requests_crowd_or_group(prompt):
        if "multiple requested" not in desc.lower():
            return _clean_text(f"Multiple requested subjects arranged coherently as a visible group or crowd in one scene; {desc}")
    return desc


def _prompt_requests_crowd_or_group(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(?:several|many|multiple|various|group|groups|crowd|crowds|people|v[áa]rios|v[áa]rias|muitos|muitas|grupo|grupos|multid[ãa]o|multid[oõ]es|pessoas)\b",
            prompt,
            flags=re.IGNORECASE,
        )
    )


def _requested_count(prompt: str) -> int | None:
    patterns = [
        (r"\b(?:2|two|dois|duas)\b", 2),
        (r"\b(?:3|three|três|tres)\b", 3),
        (r"\b(?:4|four|quatro)\b", 4),
        (r"\b(?:5|five|cinco)\b", 5),
        (r"\b(?:6|six|seis)\b", 6),
    ]
    for pattern, count in patterns:
        if re.search(pattern, prompt, flags=re.IGNORECASE):
            return count
    return None


def _counted_subject_elements(prompt: str, desc: str) -> list[dict[str, Any]]:
    count = _requested_count(prompt)
    if not count or count < 2 or count > 6:
        return []
    lowered = prompt.lower()
    if _prompt_requests_regional_layout(prompt):
        return []
    if re.search(r"\b(?:word|text|title|palavra|texto|t[ií]tulo)\b", lowered, flags=re.IGNORECASE):
        return []
    layouts: dict[int, list[list[int]]] = {
        2: [[120, 70, 930, 520], [120, 480, 930, 930]],
        3: [[140, 50, 900, 360], [140, 345, 900, 655], [140, 640, 900, 950]],
        4: [[90, 70, 500, 470], [90, 530, 500, 930], [500, 70, 910, 470], [500, 530, 910, 930]],
        5: [[80, 70, 460, 390], [80, 610, 460, 930], [330, 340, 720, 660], [560, 70, 940, 390], [560, 610, 940, 930]],
        6: [[90, 60, 420, 340], [90, 360, 420, 640], [90, 660, 420, 940], [560, 60, 890, 340], [560, 360, 890, 640], [560, 660, 890, 940]],
    }
    _background_cue, foreground_cue = _scene_depth_cues(prompt)
    base_desc = foreground_cue or desc
    base_desc = re.sub(r"^exactly\s+\w+\s+requested primary subjects or objects,\s*no extra duplicate figures;\s*", "", base_desc, flags=re.IGNORECASE)
    return [
        {
            "type": "obj",
            "bbox": bbox,
            "desc": f"Requested subject or object {index + 1} of {count}: {base_desc}. Keep it distinct, visible, and within this region.",
        }
        for index, bbox in enumerate(layouts[count])
    ]


def _preserve_requested_text(desc: str, prompt: str) -> str:
    matches = re.findall(r"\b(?:word|text|title|palavra|texto|título|titulo)\s+['\"]?([A-Z0-9][A-Z0-9 _-]{1,32})['\"]?", prompt, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\b([A-Z0-9]{4,16})\b", prompt)
    for match in matches[:3]:
        token = _clean_text(match).strip(" .,:;\"'")
        if token and token.lower() not in desc.lower():
            desc = _clean_text(f'{desc}; visible text "{token}"')
    return desc


def _remove_vague_prompt_expansion(desc: str) -> str:
    text = re.sub(r"\b(?:representing combat and action|symbolizing defense and medieval themes)\b", "", desc, flags=re.IGNORECASE)
    text = re.sub(r"\s*;\s*;", ";", text)
    return _clean_text(text.strip(" ;"))


def _has_concrete_background(value: Any) -> bool:
    text = _clean_background(value).lower()
    if not text:
        return False
    return not any(phrase in text for phrase in ["coherent background matching", "background matching the main scene", "plain or subtle"])


def _simple_scene_background(prompt: str, high_level: str) -> str:
    background_cue, _foreground_cue = _scene_depth_cues(prompt)
    if background_cue:
        return _clean_text(f"Background depth cue from the user: {background_cue}. Keep it behind the foreground subjects with clear depth separation.")
    text = f"{prompt} {high_level}".lower()
    def has(pattern: str) -> bool:
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    if has(r"\b(forest|floresta|woods|jungle|selva)\b"):
        return "A coherent forest environment with natural depth, ground texture, and clear subject separation."
    if has(r"\b(city|street|cidade|rua|urban)\b"):
        return "A coherent urban environment with readable streetscape depth and clear subject placement."
    if has(r"\b(beach|ocean|sea|praia|mar|oceano)\b"):
        return "A coherent coastal environment with natural horizon, ground texture, and clear subject separation."
    if has(r"\b(room|interior|bedroom|kitchen|sala|quarto|cozinha)\b"):
        return "A coherent interior environment with readable room layout and clear subject separation."
    if has(r"\b(poster|logo|emblem|text|word|title|palavra|texto|t[ií]tulo)\b"):
        return "A clean subtle poster background that supports the requested layout without adding extra subjects."
    if has(r"\b(medieval|warrior|warriors|knight|knights|castle|sword|swords|armor|armadura|guerreiro|guerreiros|cavaleiro|cavaleiros|espada|espadas)\b"):
        return "A coherent medieval stone courtyard or castle setting with natural ground texture and daylight."
    if has(r"\b(cat|dog|animal|gato|cachorro)\b"):
        return "A simple natural environment with clear subject separation."
    if has(r"\b(car|sport car|vehicle|autom[oó]vel|carro)\b"):
        return "A clean road or studio setting with the vehicle clearly visible."
    return "A coherent environment that supports the requested scene with natural visual context."


def _scene_depth_cues(prompt: str) -> tuple[str, str]:
    text = _clean_text(prompt)
    if not text:
        return "", ""
    clauses = [
        _clean_text(part)
        for part in re.split(r"\s*(?:[,;.]|\s+\be\b\s+|\s+\band\b\s+)\s*", text, flags=re.IGNORECASE)
        if _clean_text(part)
    ]
    background_parts: list[str] = []
    foreground_parts: list[str] = []
    background_re = re.compile(r"\b(?:no fundo|ao fundo|fundo|background|in the background|behind|atrás|atras|plano de fundo)\b", re.IGNORECASE)
    foreground_re = re.compile(r"\b(?:na frente|da frente|em frente|primeiro plano|foreground|in front|front)\b", re.IGNORECASE)
    for clause in clauses:
        if background_re.search(clause):
            background_parts.append(clause)
        if foreground_re.search(clause):
            foreground_parts.append(clause)

    if not background_parts:
        match = re.search(
            r"([^.;,]{0,80}\b(?:no fundo|ao fundo|in the background|background|behind|atrás|atras|plano de fundo)\b[^.;,]{0,100})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            background_parts.append(_clean_text(match.group(1)))
    if not foreground_parts:
        match = re.search(
            r"([^.;,]{0,100}\b(?:na frente|da frente|em frente|primeiro plano|foreground|in front|front)\b[^.;,]{0,100})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            foreground_parts.append(_clean_text(match.group(1)))

    return _dedupe_semicolon_phrases("; ".join(background_parts)), _dedupe_semicolon_phrases("; ".join(foreground_parts))


def _prompt_subjects(prompt: str) -> list[str]:
    text = _clean_text(prompt).rstrip(".")
    if not text:
        return []
    parts = re.split(r"\s*(?:,|;|\band\b|\be\b|\+)\s*", text, flags=re.IGNORECASE)
    subjects = []
    for part in parts:
        clean = re.sub(r"^(?:photo|image|picture|render|shot)\s+of\s+", "", part.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"^(?:a|an|the|um|uma|o|a)\s+", "", clean.strip(), flags=re.IGNORECASE)
        if clean and len(clean) >= 3:
            subjects.append(clean)
    return subjects[:4] or [text]
