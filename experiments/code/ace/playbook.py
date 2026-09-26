"""Parse, retrieve, filter, and update DuoGate playbook entries."""
import json
import re
from .utils import get_section_slug


APP_KEYWORDS = {
    "venmo": ("venmo", "payment", "pay", "paid", "request", "charge", "transaction", "balance", "funding", "counterparty"),
    "splitwise": ("splitwise", "debt", "owed", "owes", "expense", "settle", "group", "split"),
    "spotify": ("spotify", "playlist", "song", "track", "album", "artist", "music"),
    "simple_note": ("simplenote", "simple_note", "simple note", "note", "notes", "metadata"),
    "phone": ("phone", "sms", "text", "message", "voice message", "call", "reply", "contact"),
    "file": ("file", "csv", "spreadsheet", "table", "json", "export", "import"),
    "email": ("email", "gmail", "mail", "inbox", "send", "reply", "forward"),
    "calendar": ("calendar", "event", "meeting", "invite", "invitation", "schedule"),
}

OPERATION_KEYWORDS = (
    "all", "every", "each", "collection", "request", "transaction", "playlist",
    "song", "note", "invitation", "message", "complete_task", "answer",
    "pagination", "page", "delete", "update", "create", "send", "latest",
    "exact", "amount", "status", "funding", "metadata", "direction",
)

CRITICAL_KEYWORDS = (
    "complete_task", "answer", "action-only", "bare complete", "set-level",
    "all/every/each", "mutation", "evidence", "pagination", "venmo",
    "direction", "counterparty", "amount", "request id", "status",
    "simplenote", "metadata", "splitwise", "phone", "latest reply",
    "delete", "update", "send", "create",
)


# ---------------------------------------------------------------------------
# Harmful-content filter
# ---------------------------------------------------------------------------
#
# Earlier runs polluted the trained playbook with bullets that told the
# generator to call `complete_task(answer='completed')` or pass a
# descriptive answer as a "workaround" when the confidence gate blocked
# the agent. AppWorld grades action-only tasks against
# `ground_truth_answer = null`, so those bullets directly cause
# `assert answers match` failures.  This filter rejects any curator
# operation whose `content` matches these patterns.
_HARMFUL_PLAYBOOK_PATTERNS = (
    r"complete_task\(\s*answer\s*=\s*['\"]completed",
    r"complete_task\(\s*answer\s*=\s*['\"]success",
    r"complete_task\(\s*answer\s*=\s*['\"]done",
    r"Action completed successfully",
    r"brief descriptive answer",
    r"descriptive answer as workaround",
    r"answer\s*=.*as workaround",
    r"transmitted content as the answer",
    r"always return the transmitted content",
    r"return the transmitted content",
    r"if blocked.*complete_task\(\s*answer",
    r"workaround.*complete_task",
    r"complete_task.*workaround",
)


def is_harmful_playbook_content(content):
    """True if `content` looks like a harmful completion workaround.

    Curators occasionally invent rules of the form "if the confidence gate
    blocks complete_task, just pass answer='completed' as a workaround".
    Those bullets are catastrophic for AppWorld action tasks because they
    train the generator to emit non-null answers on tasks whose
    `ground_truth_answer` is null. Reject them at curator-apply time so
    they never make it into the playbook on disk.
    """
    if not content:
        return False
    text = str(content)
    try:
        return any(
            re.search(p, text, flags=re.IGNORECASE)
            for p in _HARMFUL_PLAYBOOK_PATTERNS
        )
    except Exception:
        return False


def parse_playbook_line(line):
    """Parse a single playbook line to extract components.

    Supports both formats:
    1) "[id] helpful=X harmful=Y :: content"
    2) "[id] content" (counts default to 0)
    """
    text = line.strip()
    # New/primary format with counts
    pattern_full = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    match = re.match(pattern_full, text)
    if match:
        return {
            'id': match.group(1),
            'helpful': int(match.group(2)),
            'harmful': int(match.group(3)),
            'content': match.group(4),
            'raw_line': line
        }
    # Fallback simple format without counts
    pattern_simple = r'\[([^\]]+)\]\s*(.*)'
    match2 = re.match(pattern_simple, text)
    if match2:
        return {
            'id': match2.group(1),
            'helpful': 0,
            'harmful': 0,
            'content': match2.group(2).strip(),
            'raw_line': line
        }
    return None

def get_next_global_id(playbook_text):
    """Extract highest global ID and return next one"""
    max_id = 0
    lines = playbook_text.strip().split('\n')
    
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            # Extract numeric part from ID
            id_match = re.search(r'-(\d+)$', parsed['id'])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)
    
    return max_id + 1


def format_playbook_line(bullet_id, helpful, harmful, content):
    """Format a bullet into playbook line format (counts removed)."""
    return f"[{bullet_id}] {content}"


def _normalize_for_dedup(text):
    text = str(text or "").lower()
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_playbook_entries(playbook_text):
    entries = []
    current_section = "general"
    for line in (playbook_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            current_section = stripped[2:].strip() or "general"
            entries.append({
                "type": "section",
                "section": current_section,
                "raw": line,
                "content": stripped,
            })
            continue
        parsed = parse_playbook_line(line)
        if parsed:
            entries.append({
                "type": "bullet",
                "section": current_section,
                "raw": line,
                "content": parsed.get("content", ""),
                "id": parsed.get("id", ""),
            })
        elif stripped and not stripped.startswith("#"):
            entries.append({
                "type": "text",
                "section": current_section,
                "raw": line,
                "content": stripped,
            })
    return entries


def detect_task_apps(task_instruction):
    text = (task_instruction or "").lower()
    detected = []
    for app, keywords in APP_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            detected.append(app)
    if "note" in text and "simple_note" not in detected:
        detected.append("simple_note")
    if not detected:
        detected.append("generic")
    return detected


def _score_playbook_entry(entry, task_terms, detected_apps):
    text = (entry.get("content") or "") + " " + (entry.get("section") or "")
    lower = text.lower()
    score = 0
    for app in detected_apps:
        if app == "generic":
            continue
        for keyword in APP_KEYWORDS.get(app, (app,)):
            if keyword in lower:
                score += 8
                break
    score += sum(2 for term in task_terms if len(term) >= 4 and term in lower)
    score += sum(3 for keyword in OPERATION_KEYWORDS if keyword in lower)
    score += sum(5 for keyword in CRITICAL_KEYWORDS if keyword in lower)
    if entry.get("type") == "section":
        score += 2
    content_len = len(entry.get("content") or "")
    if content_len > 1800 and "```" in lower:
        score -= 10
    elif content_len > 1400:
        score -= 4
    return score


def select_playbook_for_task(
    full_playbook: str,
    task_instruction: str,
    max_chars: int = 60000,
    min_chars: int = 12000,
    max_bullets: int = 80,
) -> tuple[str, dict]:
    """Select existing task-relevant guidance without adding static instructions."""
    full_playbook = full_playbook or ""
    max_chars = max(2000, int(max_chars or 60000))
    min_chars = max(0, int(min_chars or 12000))
    max_bullets = max(1, int(max_bullets or 80))
    detected_apps = detect_task_apps(task_instruction)

    if len(full_playbook) <= max_chars:
        selected = full_playbook
        metadata = {
            "full_playbook_chars": len(full_playbook),
            "selected_playbook_chars": len(selected),
            "selected_ratio": (len(selected) / len(full_playbook)) if full_playbook else 1.0,
            "detected_apps": detected_apps,
            "selected_section_count": len([l for l in selected.splitlines() if l.strip().startswith("##")]),
            "selected_bullet_count": len([l for l in selected.splitlines() if parse_playbook_line(l)]),
            "always_included_count": 0,
            "retrieval_mode": "full_under_budget",
        }
        return selected, metadata

    task_terms = set(re.findall(r"[a-z0-9_]+", (task_instruction or "").lower()))
    entries = _split_playbook_entries(full_playbook)
    selected_lines = []
    selected_sections = set()
    selected_bullets = 0
    seen_norm = set()

    # Include the first top-level/context section if present. Many playbooks
    # keep run-wide policy in the opening lines before app sections.
    for entry in entries[:20]:
        if entry.get("type") == "section":
            selected_lines.append(entry["raw"])
            selected_sections.add(entry.get("section") or "general")
            break
        if entry.get("type") in ("text", "bullet"):
            raw = entry.get("raw", "")
            if raw.strip() and len("\n".join(selected_lines)) + len(raw) < max_chars:
                selected_lines.append(raw)
                if entry.get("type") == "bullet":
                    selected_bullets += 1

    scored = []
    for index, entry in enumerate(entries):
        if entry.get("type") not in ("bullet", "text"):
            continue
        norm = _normalize_for_dedup(entry.get("content") or entry.get("raw"))
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        score = _score_playbook_entry(entry, task_terms, detected_apps)
        if score <= 0 and "generic" not in detected_apps:
            section = (entry.get("section") or "").lower()
            if any(app.replace("_", " ") in section or app in section for app in detected_apps):
                score = 3
        scored.append((score, index, entry))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    current_chars = len("\n".join(selected_lines))
    for score, _, entry in scored:
        if selected_bullets >= max_bullets:
            break
        raw = entry.get("raw", "")
        if not raw.strip():
            continue
        projected = current_chars + len(raw) + 1
        if projected > max_chars:
            continue
        section = entry.get("section") or "general"
        if section not in selected_sections:
            selected_lines.extend(["", f"## {section}"])
            selected_sections.add(section)
            current_chars = len("\n".join(selected_lines))
        selected_lines.append(raw)
        current_chars += len(raw) + 1
        if entry.get("type") == "bullet":
            selected_bullets += 1

    # If relevance filtering was too sparse, backfill in original order up to
    # the minimum size. This is deliberately conservative for success rate.
    if current_chars < min_chars:
        selected_norms = {_normalize_for_dedup(line) for line in selected_lines}
        for entry in entries:
            if selected_bullets >= max_bullets:
                break
            if entry.get("type") not in ("bullet", "text"):
                continue
            raw = entry.get("raw", "")
            norm = _normalize_for_dedup(raw)
            if not raw.strip() or norm in selected_norms:
                continue
            if current_chars + len(raw) + 1 > max_chars:
                break
            section = entry.get("section") or "general"
            if section not in selected_sections:
                selected_lines.extend(["", f"## {section}"])
                selected_sections.add(section)
                current_chars = len("\n".join(selected_lines))
            selected_lines.append(raw)
            selected_norms.add(norm)
            current_chars += len(raw) + 1
            if entry.get("type") == "bullet":
                selected_bullets += 1

    selected = "\n".join(selected_lines).strip()
    metadata = {
        "full_playbook_chars": len(full_playbook),
        "selected_playbook_chars": len(selected),
        "selected_ratio": (len(selected) / len(full_playbook)) if full_playbook else 1.0,
        "detected_apps": detected_apps,
        "selected_section_count": len(selected_sections),
        "selected_bullet_count": selected_bullets,
        "always_included_count": 0,
        "retrieval_mode": "keyword_selection",
        "max_chars": max_chars,
        "min_chars": min_chars,
        "max_bullets": max_bullets,
    }
    return selected, metadata


def sanitize_curator_operations(
    operations,
    existing_playbook,
    max_new_bullets=2,
    max_bullet_chars=800,
    max_total_chars=1600,
    enable_dedup=True,
):
    """Cap, trim, and deduplicate ADD operations before playbook mutation."""
    max_new_bullets = max(0, int(max_new_bullets or 2))
    max_bullet_chars = max(120, int(max_bullet_chars or 800))
    max_total_chars = max(200, int(max_total_chars or 1600))
    existing_norms = {
        _normalize_for_dedup(entry.get("content"))
        for entry in _split_playbook_entries(existing_playbook or "")
        if entry.get("type") == "bullet"
    }
    kept = []
    seen = set()
    skipped_duplicate = 0
    skipped_too_long = 0
    skipped_low_value = 0
    total_chars = 0
    for op in operations or []:
        content = str((op or {}).get("content", "")).strip()
        if not content or is_harmful_playbook_content(content):
            skipped_low_value += 1
            continue
        if "```" in content or re.search(r"\{.*\}|\[.*\]", content, flags=re.DOTALL):
            skipped_low_value += 1
            continue
        if len(content) > max_bullet_chars:
            content = content[:max_bullet_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
            skipped_too_long += 1
        norm = _normalize_for_dedup(content)
        if not norm:
            skipped_low_value += 1
            continue
        prefix = norm[:180]
        if enable_dedup and (
            norm in existing_norms
            or prefix in seen
            or any(prefix and (prefix in ex or ex[:180] == prefix) for ex in existing_norms)
        ):
            skipped_duplicate += 1
            continue
        value_score = sum(1 for k in CRITICAL_KEYWORDS + OPERATION_KEYWORDS if k in content.lower())
        if value_score == 0:
            skipped_low_value += 1
            continue
        if total_chars + len(content) > max_total_chars:
            skipped_too_long += 1
            continue
        new_op = dict(op)
        new_op["content"] = content
        kept.append(new_op)
        seen.add(prefix)
        total_chars += len(content)
        if len(kept) >= max_new_bullets:
            break
    metadata = {
        "original_curator_chars": sum(len(str((op or {}).get("content", ""))) for op in operations or []),
        "sanitized_curator_chars": sum(len(str((op or {}).get("content", ""))) for op in kept),
        "original_bullet_count": len(operations or []),
        "kept_bullet_count": len(kept),
        "skipped_duplicate_count": skipped_duplicate,
        "skipped_too_long_count": skipped_too_long,
        "skipped_low_value_count": skipped_low_value,
    }
    return kept, metadata

def apply_curator_operations(playbook_text, operations, next_id):
    """Apply supported ADD operations to the playbook."""
    lines = playbook_text.strip().split('\n')
    
    # Build section map
    sections = {}
    current_section = "general"
    section_line_map = {}  # Track which line each section header is on
    for i, line in enumerate(lines):
        if line.strip().startswith('##'):
            # Extract section name and normalize it
            section_header = line.strip()[2:].strip()
            # Normalize: lowercase, spaces->_, &->and, strip trailing ':'
            normalized = section_header.lower().replace(' ', '_').replace('&', 'and').rstrip(':')
            current_section = normalized
            section_line_map[current_section] = i
            if current_section not in sections:
                sections[current_section] = []
        elif line.strip():
            sections[current_section].append((i, line))
    
    # Process operations
    bullets_to_add = []
    
    for op in operations:
        op_type = op['type']
        
        if op_type == 'ADD':
            # Normalize section name from operation
            section_raw = op.get('section', 'general')
            section = section_raw.lower().replace(' ', '_').replace('&', 'and').rstrip(':')

            # Check if section exists, if not use 'others'
            if section not in sections and section != 'general':
                print(f"Warning: Section '{section_raw}' not found, adding to OTHERS")
                section = 'others'

            content = op.get('content', '')

            # Reject curator suggestions that would teach the generator to
            # call `complete_task(answer='completed')` etc. on action-only
            # tasks. These bullets directly produce `predicted_answer !=
            # null` failures in AppWorld grading.
            if is_harmful_playbook_content(content):
                preview = (content or "")[:120].replace('\n', ' ')
                print(f"  Skipped harmful playbook bullet: {preview}")
                continue

            slug = get_section_slug(section)
            new_id = f"{slug}-{next_id:05d}"
            next_id += 1

            new_line = format_playbook_line(new_id, 0, 0, content)
            bullets_to_add.append((section, new_line))
            print(f"  Added bullet {new_id} to section {section}")
            

    
    # Rebuild playbook
    new_lines = []
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Add new bullets to appropriate sections
    final_lines = []
    current_section = None
    
    for line in new_lines:
        if line.strip().startswith('##'):
            # Before moving to new section, add any bullets for current section
            if current_section:
                section_adds = [b for s, b in bullets_to_add if s == current_section]
                final_lines.extend(section_adds)
                # Clear added bullets
                bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
            
            section_header = line.strip()[2:].strip()
            current_section = section_header.lower().replace(' ', '_').replace('&', 'and').rstrip(':')
        final_lines.append(line)
    
    # Add remaining bullets to current section
    if current_section:
        section_adds = [b for s, b in bullets_to_add if s == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
    
    # If there are still bullets to add (for sections that don't exist), add them to OTHERS
    if bullets_to_add:
        print(f"Warning: {len(bullets_to_add)} bullets have no matching section, adding to OTHERS")
        others_bullets = [b for s, b in bullets_to_add]
        # Find OTHERS section
        others_idx = -1
        for i, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                others_idx = i
                break
        
        if others_idx >= 0:
            # Insert after OTHERS header
            for i, bullet in enumerate(others_bullets):
                final_lines.insert(others_idx + 1 + i, bullet)
        else:
            # Append to end
            final_lines.extend(others_bullets)
    
    return '\n'.join(final_lines), next_id

def extract_json_from_text(text, json_key=None):
    """Extract JSON object from text, handling various formats"""
    try:
        # First, try to parse the entire response as JSON (JSON mode)
        try:
            result = json.loads(text.strip())
            return result
        except json.JSONDecodeError:
            pass
        
        # Fallback: Look for ```json blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Try each match until we find valid JSON
            for match in matches:
                try:
                    json_str = match.strip()
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Improved JSON extraction using balanced brace counting
        # This handles deeply nested structures better
        def find_json_objects(text):
            """Find JSON objects using balanced brace counting"""
            json_objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    # Found start of potential JSON object
                    brace_count = 1
                    start = i
                    i += 1
                    
                    while i < len(text) and brace_count > 0:
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                        elif text[i] == '"':
                            # Handle quoted strings to avoid counting braces inside strings
                            i += 1
                            while i < len(text) and text[i] != '"':
                                if text[i] == '\\':
                                    i += 1  # Skip escaped character
                                i += 1
                        i += 1
                    
                    if brace_count == 0:
                        # Found complete JSON object
                        json_candidate = text[start:i]
                        json_objects.append(json_candidate)
                else:
                    i += 1
            
            return json_objects
        
        # Find all potential JSON objects
        json_objects = find_json_objects(text)
        
        for json_str in json_objects:
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"Failed to extract JSON: {e}")
        if text is None:
            print("[WARN] text is None in extract_json_from_text()")
            return None
        else:
            if len(text) > 500:
                print(f"Raw content preview:\n{text[:500]}...")
            else:
                print(f"Raw content:\n{text}")
        
    return None
