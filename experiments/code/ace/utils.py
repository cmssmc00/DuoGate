"""Section identifiers used when adding playbook entries."""


def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match new playbook sections
    slug_map = {
        "strategies_and_hard_rules": "shr",
        "hard_rules": "hr",
        "strategies_and_insights": "si",
        "apis_to_use_for_specific_information": "api",
        "useful_code_snippets_and_templates": "code",
        "code_snippets_and_templates": "code",
        "common_mistakes_and_correct_strategies": "cms",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics_and_workflows": "psw",
        "problem_solving_heuristics": "prob",
        "verification_checklist": "vc",
        "troubleshooting_and_pitfalls": "ts",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])
