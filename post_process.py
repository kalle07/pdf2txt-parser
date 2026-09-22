"""Text cleanup utilities and output post-processing."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("pdf_parser.post_process")

# Default JSON keys whose list items receive descriptions. OCR (or any other
# pipeline) can pass its own keys so the same post-processor can be reused by
# searching for a different phrase.
DEFAULT_TARGET_KEYS = ("drawings_with_resolution", "images_with_resolution", "formulas_on_page")


def remove_nonprintable_chars(text: str) -> str:
    """Remove non-printable/control characters while preserving Unicode."""
    # Ensure we are working with a string (e.g., convert non‑str inputs)
    if not isinstance(text, str):
        text = str(text)

    # Collapse Windows line endings and replace them with a single space
    cleaned = re.sub(r"[\r\n]+", " ", text)
    # Remove ASCII control characters except line feed and vertical tab
    cleaned = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", "", cleaned)
    # Replace tabs with a single space
    cleaned = cleaned.replace("\t", " ")
    # Collapse any sequence of spaces into a single space
    cleaned = re.sub(r" +", " ", cleaned)
    # Strip leading/trailing whitespace
    return cleaned.strip()


def count_non_whitespace_chars(text: str) -> int:
    """Count characters excluding all whitespace (spaces, tabs, newlines)."""
    if not isinstance(text, str):
        return 0
    # List comprehension keeps only characters that are NOT whitespace
    return len([c for c in text if not c.isspace()])


def count_characters(text: str, include_whitespace: bool = True) -> int:
    """Count characters in text, optionally excluding whitespace."""
    if not isinstance(text, str):
        return 0
    if include_whitespace:
        return len(text)
    return count_non_whitespace_chars(text)


def count_words(text: str) -> int:
    """Count word-like tokens in extracted text."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b[\w']+\b", text, flags=re.UNICODE))


def fix_hyphenated_lines(text: str) -> Tuple[str, int]:
    """Fix hyphenation issues from PDF line breaks."""
    if not isinstance(text, str):
        return text, 0

    fix_count = 0
    lines = text.split("\n")
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            fixed_lines.append("")
            i += 1
            continue

        # Detect a line that ends with a hyphen (possible PDF line‑break)
        is_hyphenated_split = bool(re.match(r"^.*-\s*$", line)) or bool(re.match(r"^.*-$", line))
        # If it is hyphenated and there is a following line, attempt to join them
        if is_hyphenated_split and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line:
                # Remove trailing hyphen and surrounding whitespace from the current line
                current_clean = re.sub(r"\s*-\s*$", "", line)
                # Join the cleaned current line with the next line
                joined_word = f"{current_clean}{next_line}"
                fixed_lines.append(joined_word)
                fix_count += 1
                i += 2
                continue

        # Otherwise keep the line as‑is
        fixed_lines.append(line)
        i += 1

    # Re‑assemble the processed lines into a single string
    result = " ".join(fixed_lines)
    return result, fix_count


class PostProcessor:
    """Post-process output files to add descriptions.

    Works standalone on a ``.txt`` plus its media folder — the source PDF is
    never required. ``target_keys`` selects which JSON keys get descriptions,
    so OCR (or any other pipeline) can reuse this by passing a different phrase.
    Every file access is guarded: a missing or unreadable txt/image/folder is
    logged and skipped, never raised, so one bad file can't abort a batch.
    """

    def __init__(
        self,
        media_folder_name: str,
        target_keys: Optional[Tuple[str, ...]] = None,
    ):
        """Initialize with the media folder name and the keys to search for."""
        self.media_folder_name = media_folder_name
        self.target_keys: Tuple[str, ...] = tuple(target_keys or DEFAULT_TARGET_KEYS)

    def process(self, output_file: str) -> Dict[str, Any]:
        """Post‑process an output file and return statistics."""
        # Initialize a dictionary to keep track of various processing metrics
        stats = {
            "json_sections_processed": 0,
            "first_pass_additions": 0,
            "second_pass_updates": 0,
            "files_not_found": 0,
            "errors_occurred": 0,
        }

        # If the target file does not exist, report it and return empty stats
        if not os.path.exists(output_file):
            logger.warning("Output file '%s' not found — skipped", output_file)
            stats["files_not_found"] += 1
            return stats

        # Read the entire file content (UTF‑8 encoded). A locked or unreadable
        # file is logged and skipped rather than raising into the batch.
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as exc:
            logger.warning("Cannot read '%s': %s — skipped", output_file, exc)
            stats["errors_occurred"] += 1
            return stats

        # Pattern to match JSON blocks delimited by ```json ... ```
        json_block_pattern = r"(```json\n)(.*?)(\n```)"

        # Callback executed for each JSON block found in the text
        def process_json_block(match):
            # Extract the three capture groups: prefix, body, suffix
            prefix = match.group(1)
            body = match.group(2)
            suffix = match.group(3)

            # Increment the counter for processed JSON sections
            stats["json_sections_processed"] += 1

            try:
                # Parse the JSON body into a Python dict
                data = json.loads(body)

                # Keys we are interested in (configurable — OCR passes its own)
                target_keys = self.target_keys

                # If none of the target keys exist, return the original block unchanged
                if not any(key in data for key in target_keys):
                    return match.group(0)

                # Flag to know whether we actually modified the JSON data
                modified = False

                # Iterate over each target key that may be present
                for key in target_keys:
                    if key in data and isinstance(data[key], list):
                        # Process each item in the list
                        for item in data[key]:
                            # Determine if the item has a "saved" flag that indicates it was saved
                            has_saved_flag = item.get("saved") is True or item.get("saved") == "true"
                            # Check if the item already has a non‑empty description
                            has_description = "description" in item and item["description"]
                            # Determine eligibility for first‑pass (needs description) or second‑pass (already has description)
                            first_pass_eligible = has_saved_flag and not has_description
                            second_pass_eligible = has_description

                            # Skip items that are neither first‑ nor second‑pass eligible
                            if not first_pass_eligible and not second_pass_eligible:
                                continue

                            # Extract the filename (without extension) for later lookup
                            filename = item.get("filename", "")
                            if not filename:
                                continue

                            base_name, _ = os.path.splitext(filename)
                            txt_filename = f"{base_name}.txt"

                            # Possible directories where the .txt file might reside
                            search_paths = [
                                os.path.dirname(output_file),
                                os.path.join(os.path.dirname(output_file), self.media_folder_name),
                            ]

                            txt_content = None
                            # Search each possible path for the .txt file
                            for search_path in search_paths:
                                if os.path.exists(search_path):
                                    txt_path = os.path.join(search_path, txt_filename)
                                    if os.path.exists(txt_path):
                                        # Read the description file; a locked or
                                        # unreadable one is skipped, not fatal.
                                        try:
                                            with open(txt_path, "r", encoding="utf-8") as txt_file:
                                                txt_content = txt_file.read()
                                        except OSError as exc:
                                            logger.warning(
                                                "Cannot read description '%s': %s — skipped",
                                                txt_path, exc,
                                            )
                                            stats["errors_occurred"] += 1
                                        break

                            # If no .txt file was found, increment the "not found" counter and skip
                            if txt_content is None:
                                stats["files_not_found"] += 1
                                continue

                            # Clean the text content:
                            #   - Remove the words "start" or "end" (case‑insensitive)
                            #   - Collapse multiple spaces into a single space and strip
                            cleaned_content = re.sub(r"\bstart\b|\bend\b", "", txt_content, flags=re.IGNORECASE)
                            cleaned_content = re.sub(r" +", " ", cleaned_content).strip()

                            # Keep the original description for comparison later
                            old_description = item.get("description", "")
                            # Replace the description with the cleaned content or a fallback message
                            item["description"] = (
                                cleaned_content if cleaned_content else "No description available"
                            )
                            # Mark that we performed a modification
                            modified = True

                            # Update counters based on whether this is a first‑pass addition or second‑pass update
                            if not old_description:
                                stats["first_pass_additions"] += 1
                            else:
                                stats["second_pass_updates"] += 1

                        # If any items were modified, replace the whole JSON block with the updated JSON
                        if modified:
                            return prefix + json.dumps(data, indent=2, ensure_ascii=False) + suffix

                # If we never entered the modification branch, return the original block unchanged
                return match.group(0)

            # If JSON parsing fails, count the error and return the original block
            except json.JSONDecodeError:
                stats["errors_occurred"] += 1
                return match.group(0)

        # Apply the callback to all JSON blocks in the file content
        modified_content = re.sub(json_block_pattern, process_json_block, content, flags=re.DOTALL)

        # Nothing changed → skip the rewrite entirely. This keeps repeated runs
        # idempotent and avoids needlessly touching the file on a second pass.
        if modified_content == content:
            logger.info("Post-processing: no changes for '%s'", output_file)
            return stats

        # Write the possibly‑modified content back to the same file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(modified_content)
        except OSError as exc:
            logger.warning("Cannot write '%s': %s — left unchanged", output_file, exc)
            stats["errors_occurred"] += 1
            return stats

        logger.info(
            "Post-processing complete for '%s' — sections=%d, added=%d, updated=%d, "
            "missing=%d, errors=%d",
            os.path.basename(output_file),
            stats["json_sections_processed"], stats["first_pass_additions"],
            stats["second_pass_updates"], stats["files_not_found"],
            stats["errors_occurred"],
        )

        # Return the statistics dictionary to the caller
        return stats


def _media_folder_for(txt_path: str, media_suffix: str) -> Optional[str]:
    """Return the media folder name paired with a .txt, or None if absent.

    Prefers the ``Media Directory:`` header the converter writes; falls back to
    the ``<stem><media_suffix>`` naming convention. Only returns a name whose
    folder actually exists next to the txt — otherwise there is nothing to
    describe and the caller skips the file.
    """
    directory = os.path.dirname(txt_path)
    stem = os.path.splitext(os.path.basename(txt_path))[0]

    candidates: List[str] = []
    # 1) header line, if the file is readable
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for _ in range(20):  # header sits at the top; don't scan huge files
                line = f.readline()
                if not line:
                    break
                if line.startswith("Media Directory:"):
                    name = line.split(":", 1)[1].strip()
                    if name:
                        candidates.append(name)
                    break
    except OSError as exc:
        logger.warning("Cannot read header of '%s': %s", txt_path, exc)

    # 2) naming convention
    candidates.append(f"{stem}{media_suffix}")

    for name in candidates:
        if os.path.isdir(os.path.join(directory, name)):
            return name
    return None


def process_directory(
    directory: str,
    *,
    media_suffix: str = "_media",
    target_keys: Optional[Tuple[str, ...]] = None,
    require_media: bool = True,
) -> Dict[str, Any]:
    """Post-process every ``.txt`` in ``directory`` — no source PDF needed.

    This is the standalone entry point: point it at a folder of converted
    ``.txt`` files (each with its ``_media`` folder) and it fills in image /
    drawing descriptions. Missing files, missing media folders and unreadable
    files are logged and skipped so a single bad file never aborts the run.

    ``target_keys`` lets OCR (or any pipeline) reuse this by searching for a
    different phrase. When ``require_media`` is True, txt files with no media
    folder are skipped (nothing to describe); set False to still scan them.
    """
    aggregate: Dict[str, Any] = {
        "files_seen": 0,
        "files_processed": 0,
        "files_skipped_no_media": 0,
        "json_sections_processed": 0,
        "first_pass_additions": 0,
        "second_pass_updates": 0,
        "files_not_found": 0,
        "errors_occurred": 0,
    }

    if not os.path.isdir(directory):
        logger.warning("Post-process directory '%s' does not exist", directory)
        return aggregate

    try:
        entries = sorted(os.listdir(directory))
    except OSError as exc:
        logger.warning("Cannot list '%s': %s", directory, exc)
        return aggregate

    for entry in entries:
        if not entry.lower().endswith(".txt"):
            continue
        txt_path = os.path.join(directory, entry)
        if not os.path.isfile(txt_path):
            continue
        aggregate["files_seen"] += 1

        media_name = _media_folder_for(txt_path, media_suffix)
        if media_name is None and require_media:
            aggregate["files_skipped_no_media"] += 1
            logger.info("No media folder for '%s' — skipped", entry)
            continue

        processor = PostProcessor(media_name or "", target_keys=target_keys)
        try:
            stats = processor.process(txt_path)
        except Exception as exc:  # last-resort guard: one file never kills the batch
            aggregate["errors_occurred"] += 1
            logger.warning("Unexpected error post-processing '%s': %s", entry, exc)
            continue

        aggregate["files_processed"] += 1
        for key in (
            "json_sections_processed", "first_pass_additions",
            "second_pass_updates", "files_not_found", "errors_occurred",
        ):
            aggregate[key] += stats.get(key, 0)

    logger.info(
        "Directory post-processing done: %d/%d files processed, %d skipped (no media), %d errors",
        aggregate["files_processed"], aggregate["files_seen"],
        aggregate["files_skipped_no_media"], aggregate["errors_occurred"],
    )
    return aggregate
