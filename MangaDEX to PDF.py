import requests
import os
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import io
import html  # Import the html module to decode entities
import time
import json  # Import the json library

try:
    from google.cloud import vision
    import google.generativeai as genai
    from google.auth.exceptions import DefaultCredentialsError

    GOOGLE_LIBS_AVAILABLE = True
except ImportError:
    GOOGLE_LIBS_AVAILABLE = False


def get_chapter_id_from_url(url):
    """Extracts the chapter UUID from a MangaDex URL."""
    match = re.search(r'/chapter/([a-f0-9-]+)', url)
    if match:
        return match.group(1)
    return None


def get_chapter_metadata(chapter_id):
    """Fetches metadata for the chapter, including manga title and chapter number."""
    try:
        url = f"https://api.mangadex.org/chapter/{chapter_id}?includes[]=manga"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        chapter_number = data['data']['attributes']['chapter']
        manga_data = next((rel for rel in data['data']['relationships'] if rel['type'] == 'manga'), None)

        if manga_data:
            manga_title = manga_data['attributes']['title'].get('en', 'N/A')
        else:
            manga_title = "Unknown Manga"

        # Sanitize title and chapter for use in a filename
        safe_title = re.sub(r'[\\/*?:"<>|]', "", manga_title)
        safe_chapter = f"Chapter {chapter_number}" if chapter_number else "Oneshot"

        return f"{safe_title} - {safe_chapter}.pdf"

    except requests.exceptions.RequestException as e:
        print(f"Error fetching chapter metadata: {e}")
        return f"{chapter_id}.pdf"  # Fallback filename


def get_image_urls(chapter_id):
    """Fetches the image URLs for a given chapter ID."""
    try:
        url = f"https://api.mangadex.org/at-home/server/{chapter_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data['result'] != 'ok':
            print("Failed to get chapter data from MangaDex API.")
            return None

        base_url = data['baseUrl']
        chapter_hash = data['chapter']['hash']
        page_filenames = data['chapter']['data']  # Use high quality images

        image_urls = [f"{base_url}/data/{chapter_hash}/{filename}" for filename in page_filenames]
        return image_urls

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image URLs: {e}")
        return None


def _wrap_text(text, font, max_width):
    """Helper function to wrap text to fit a max width."""
    lines = []
    words = text.split()
    if not words:
        return lines

    current_line = words[0]
    for word in words[1:]:
        # Use getbbox for more accurate width calculation
        if font.getbbox(current_line + " " + word)[2] <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def _are_boxes_close(box1_verts, box2_verts, threshold):
    """Checks if two bounding boxes are close to each other, either by overlap or proximity."""
    # Bounding box format: (x_min, y_min, x_max, y_max)
    b1_min_x = min(v.x for v in box1_verts)
    b1_min_y = min(v.y for v in box1_verts)
    b1_max_x = max(v.x for v in box1_verts)
    b1_max_y = max(v.y for v in box1_verts)

    b2_min_x = min(v.x for v in box2_verts)
    b2_min_y = min(v.y for v in box2_verts)
    b2_max_x = max(v.x for v in box2_verts)
    b2_max_y = max(v.y for v in box2_verts)

    # Check for overlap
    if not (b1_max_x < b2_min_x or b1_min_x > b2_max_x or b1_max_y < b2_min_y or b1_min_y > b2_max_y):
        return True

    # Calculate distance between non-overlapping boxes
    dx = max(0, b1_min_x - b2_max_x, b2_min_x - b1_max_x)
    dy = max(0, b1_min_y - b2_max_y, b2_min_y - b1_max_y)

    # Use squared distance to avoid expensive sqrt operation
    return (dx ** 2 + dy ** 2) < (threshold ** 2)


def _translate_single_image(path, vision_client, gemini_model, font_path):
    """Helper that handles the translation for one image with improved text fitting."""
    try:
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = vision_client.document_text_detection(image=image)
        if response.error.message:
            raise Exception(f'{response.error.message}')

        doc_text = response.full_text_annotation
        pil_image = Image.open(path)
        draw = ImageDraw.Draw(pil_image)

        if not doc_text.pages:
            return

        for page in doc_text.pages:
            # --- LANGUAGE DETECTION & SPATIAL CLUSTERING LOGIC ---
            # 1. Get a flat list of all paragraphs, now including language info.
            all_paragraphs = []
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    paragraph_text = ""
                    # --- NEW: Language Detection ---
                    lang_code = "und"  # Undetermined
                    if paragraph.property and paragraph.property.detected_languages:
                        lang_code = paragraph.property.detected_languages[0].language_code
                    # --- END NEW ---

                    for word in paragraph.words:
                        for symbol in word.symbols:
                            paragraph_text += symbol.text
                        if not (word.symbols[-1].property and word.symbols[-1].property.detected_break.type in [3, 5]):
                            paragraph_text += ' '

                    if paragraph_text.strip():
                        all_paragraphs.append({
                            'text': paragraph_text,
                            'box': paragraph.bounding_box,
                            'lang': lang_code  # Store the detected language
                        })

            if not all_paragraphs:
                continue

            # 2. Separate text to be translated from text to be kept as-is
            # --- UPDATED LOGIC ---
            # Translate ANY text that is not confidently detected as English.
            to_translate_indices = [i for i, p in enumerate(all_paragraphs) if p['lang'] != 'en']
            # --- END UPDATED LOGIC ---
            to_translate_paragraphs = [all_paragraphs[i] for i in to_translate_indices]

            final_translations = [""] * len(all_paragraphs)

            # Populate the final list with the text we're keeping
            for i in range(len(all_paragraphs)):
                if i not in to_translate_indices:
                    final_translations[i] = all_paragraphs[i]['text']

            # Only call the LLM if there's something to translate
            if to_translate_paragraphs:
                expected_item_count = len(to_translate_paragraphs)
                combined_prompt_text = "\n".join(
                    [f"{i + 1}. {p['text']}" for i, p in enumerate(to_translate_paragraphs)])

                # --- NEW: Language-Agnostic Prompt ---
                prompt = (
                    "You are an expert manga localizer. Your goal is to translate text from a manga page into natural, authentic English. "
                    "Do not perform a literal translation. Instead, capture the original intent, emotion, and character voice. "
                    "Below are several numbered text blocks from a single manga page. Translate each one into English. "
                    "You MUST return a translation for every single numbered item. "
                    "Respond with ONLY a single JSON object. The object should have one key, \"translations\", "
                    "which contains a list of strings with the translated text in the correct order. The number of items in the list MUST exactly match the number of input items. "
                    "For example: {\"translations\": [\"Translation one\", \"Translation two\"]}"
                    "Do not add any extra commentary or explanations. Just the JSON object."
                    f"\n\n---\n\n{combined_prompt_text}"
                )
                # --- END NEW ---

                llm_translations = []
                for attempt in range(2):  # Try up to two times
                    try:
                        gemini_response = gemini_model.generate_content(prompt)
                        if not gemini_response.parts: raise ValueError("Blocked response")
                        response_text = gemini_response.text
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if not json_match: raise json.JSONDecodeError("No JSON found", response_text, 0)

                        parsed_json = json.loads(json_match.group(0))
                        llm_translations = parsed_json.get("translations", [])

                        if len(llm_translations) == expected_item_count:
                            break
                        else:
                            error_message = f"Mismatch: expected {expected_item_count}, got {len(llm_translations)}."
                            prompt = f"{error_message}\n\nOriginal request:\n{prompt}"
                            if attempt == 0: print(f"Warning: Mismatch for {os.path.basename(path)}. Retrying...")
                    except (json.JSONDecodeError, ValueError, Exception) as e:
                        if attempt == 0:
                            faulty_response = gemini_response.text if 'gemini_response' in locals() else str(e)
                            prompt = f"Invalid response: '{faulty_response}'. Retry providing ONLY a valid JSON with {expected_item_count} items."
                            print(f"Warning: Parse fail for {os.path.basename(path)}. Retrying...")
                        else:
                            print(f"Error on {os.path.basename(path)}: {e}")
                            llm_translations = ["[Error]"] * expected_item_count

                # Merge the LLM translations back into the final list
                for i, trans_text in enumerate(llm_translations):
                    original_index = to_translate_indices[i]
                    final_translations[original_index] = trans_text

            # 3. Spatially cluster the paragraphs into logical groups (speech bubbles, etc.)
            avg_height = sum(p['box'].vertices[2].y - p['box'].vertices[0].y for p in all_paragraphs) / len(
                all_paragraphs) if all_paragraphs else 0
            merge_threshold = avg_height * 1.5

            groups = [[i] for i in range(len(all_paragraphs))]
            # (Clustering logic remains the same)
            merged_in_pass = True
            while merged_in_pass:
                merged_in_pass = False
                i = 0
                while i < len(groups):
                    j = i + 1
                    while j < len(groups):
                        should_merge = False
                        for p1_idx in groups[i]:
                            for p2_idx in groups[j]:
                                if _are_boxes_close(all_paragraphs[p1_idx]['box'].vertices,
                                                    all_paragraphs[p2_idx]['box'].vertices, merge_threshold):
                                    should_merge = True
                                    break
                            if should_merge:
                                break

                        if should_merge:
                            groups[i].extend(groups[j])
                            groups.pop(j)
                            merged_in_pass = True
                        else:
                            j += 1
                    i += 1

            # 4. Iterate through the new logical groups to typeset the text
            for indices in groups:
                block_paragraphs = [all_paragraphs[i] for i in indices]
                block_translations = [final_translations[i] for i in indices]

                # Don't draw empty blocks
                if not any(block_translations):
                    continue

                full_block_text = " ".join(block_translations)
                min_x = min(p['box'].vertices[0].x for p in block_paragraphs)
                min_y = min(p['box'].vertices[0].y for p in block_paragraphs)
                max_x = max(p['box'].vertices[1].x for p in block_paragraphs)
                max_y = max(p['box'].vertices[2].y for p in block_paragraphs)
                block_width = max_x - min_x
                block_height = max_y - min_y

                original_heights = [p['box'].vertices[2].y - p['box'].vertices[0].y for p in block_paragraphs if
                                    p['lang'] != 'en']
                avg_original_height = sum(original_heights) / len(original_heights) if original_heights else 30
                max_font_size_cap = int(avg_original_height * 1.5)

                font_size = min(int(block_height), max_font_size_cap)
                final_wrapped_lines = []
                best_font = None

                while font_size > 6:
                    font = ImageFont.truetype(font_path, size=font_size)
                    wide_wrapped_lines = _wrap_text(full_block_text, font, block_width * 0.95)
                    if not wide_wrapped_lines:
                        font_size -= 2
                        continue
                    line_height = font.getbbox("A")[3] - font.getbbox("A")[1] if "A" in full_block_text else 10
                    wide_total_height = len(wide_wrapped_lines) * (line_height * 1.2)
                    if wide_total_height > block_height:
                        font_size -= 2
                        continue

                    box_aspect_ratio = block_width / block_height if block_height > 0 else 1
                    if box_aspect_ratio < 2.5:
                        tighter_wrap_width = block_width * 0.8
                        tighter_wrapped_lines = _wrap_text(full_block_text, font, tighter_wrap_width)
                        tighter_total_height = len(tighter_wrapped_lines) * (line_height * 1.2)
                        if tighter_total_height <= block_height:
                            final_wrapped_lines = tighter_wrapped_lines
                        else:
                            final_wrapped_lines = wide_wrapped_lines
                    else:
                        final_wrapped_lines = wide_wrapped_lines
                    best_font = font
                    break

                if not best_font:
                    best_font = ImageFont.truetype(font_path, size=6)
                    final_wrapped_lines = _wrap_text(full_block_text, best_font, block_width * 0.95)

                # Erase original text boxes
                for p_data in block_paragraphs:
                    # Only erase if it's not English text we decided to keep
                    if p_data['lang'] != 'en':
                        draw.polygon([(v.x, v.y) for v in p_data['box'].vertices], fill='white', outline='white')

                # Draw the full wrapped text block
                if final_wrapped_lines:
                    line_height = best_font.getbbox(final_wrapped_lines[0])[3] - \
                                  best_font.getbbox(final_wrapped_lines[0])[1]
                    total_text_height = len(final_wrapped_lines) * (line_height * 1.2)
                    y_start = min_y + (block_height - total_text_height) / 2

                    # --- NEW: Alignment Heuristic ---
                    is_wide_box = (block_width / block_height if block_height > 0 else 1) > 2.0
                    for line in final_wrapped_lines:
                        if is_wide_box:  # Left-align text in wide narration boxes
                            x_start = min_x + (block_width * 0.05)  # Small left padding
                        else:  # Center-align in speech bubbles
                            line_width = best_font.getbbox(line)[2]
                            x_start = min_x + (block_width - line_width) / 2
                        draw.text((x_start, y_start), line, font=best_font, fill='black')
                        y_start += line_height * 1.2

        pil_image.save(path)

    except Exception as e:
        print(f"\nFATAL ERROR while translating {os.path.basename(path)}: {e}")


def translate_images(image_paths, target_language='en'):
    """
    Translates text within a list of images using Google Cloud APIs.
    """
    # --- NEW: Font Customization ---
    # To use a different font, change the path below.
    # Download a manga/comic font (e.g., from Google Fonts, Blambot) and place the .ttf file
    # in the same folder as the script, or provide the full path to it.
    font_path = "WildWords-Regular.ttf"
    # --- END NEW ---

    print("\nInitializing Google Cloud clients for Vision and Gemini...")
    try:
        vision_client = vision.ImageAnnotatorClient()
        gemini_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    except Exception as e:
        print("\nFailed to initialize Google Cloud clients.")
        raise e

    try:
        ImageFont.truetype(font_path, size=10)
    except IOError:
        print(f"\n--- FONT ERROR ---")
        print(f"Font file '{font_path}' not found. Translation may not work correctly.")
        print("Please download a font, place it in the same directory, and update the `font_path` variable.")
        font_path = "arial.ttf"  # Fallback to Arial
        try:
            ImageFont.truetype(font_path, size=10)
            print("Falling back to 'arial.ttf'.")
        except IOError:
            print("Fallback font 'arial.ttf' also not found. Please install a standard font.")
            return

    print("Translating images with LLM-powered localization (this may take a while)...")
    for path in tqdm(image_paths, desc="Translating Pages"):
        _translate_single_image(path, vision_client, gemini_model, font_path)


def download_images(image_urls, chapter_id, limit=0):
    """Downloads images into a temporary folder."""
    temp_dir = f"temp_{chapter_id}"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    image_paths = []

    urls_to_download = image_urls
    if limit > 0:
        urls_to_download = image_urls[:limit]
        print(f"\n--- DEV MODE: Downloading only the first {limit} page(s). ---")

    print("Downloading pages...")
    for i, url in enumerate(tqdm(urls_to_download, desc="Downloading")):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            file_extension = os.path.splitext(url)[1]
            filepath = os.path.join(temp_dir, f"page_{i:03d}{file_extension}")

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            image_paths.append(filepath)
        except requests.exceptions.RequestException as e:
            print(f"\nCould not download {url}: {e}")

    return image_paths, temp_dir


def create_pdf_from_images(image_paths, output_filename):
    """Creates a PDF from a list of image paths."""
    if not image_paths:
        print("No images were downloaded, cannot create PDF.")
        return

    print(f"\nCreating PDF: {output_filename}")

    images = []
    try:
        for path in tqdm(image_paths, desc="Processing"):
            img = Image.open(path).convert('RGB')
            images.append(img)

        if images:
            images[0].save(
                output_filename,
                save_all=True,
                append_images=images[1:],
                resolution=100.0,
                quality=95
            )
            print("PDF created successfully!")
        else:
            print("Image list is empty, PDF not created.")

    except Exception as e:
        print(f"An error occurred while creating the PDF: {e}")
    finally:
        for img in images:
            img.close()


def cleanup(directory):
    """Removes the temporary directory and its contents."""
    if not os.path.exists(directory):
        return
    try:
        for filename in os.listdir(directory):
            os.remove(os.path.join(directory, filename))
        os.rmdir(directory)
        print("Temporary files cleaned up.")
    except OSError as e:
        print(f"Error during cleanup: {e}")


def main():
    """Main function to run the scraper."""
    chapter_url = input("Enter the MangaDex chapter URL: ")

    chapter_id = get_chapter_id_from_url(chapter_url)
    if not chapter_id:
        print("Invalid MangaDex URL. Please make sure it's a chapter URL.")
        return

    pdf_filename = get_chapter_metadata(chapter_id)
    image_urls = get_image_urls(chapter_id)

    if not image_urls:
        return

    page_limit = 0  # Default to all pages
    translate_choice = 'n'

    if GOOGLE_LIBS_AVAILABLE:
        translate_choice = input("\nDo you want to attempt to translate this chapter? (y/n): ").lower()
        if translate_choice == 'y':
            limit_choice = input("Translate all pages or a limited number for testing? (all/limit): ").lower()
            if limit_choice == 'limit':
                while True:
                    try:
                        num_pages = int(input(f"How many pages to translate? (1-{len(image_urls)}): "))
                        if 1 <= num_pages <= len(image_urls):
                            page_limit = num_pages
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(image_urls)}.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")

    image_paths, temp_dir = download_images(image_urls, chapter_id, limit=page_limit)

    if image_paths:
        if GOOGLE_LIBS_AVAILABLE and translate_choice == 'y':
            print("\nStarting LLM-powered translation process...")
            print("Note: This now requires the Google Generative AI SDK and the Vision API.")
            try:
                translate_images(image_paths)
            except DefaultCredentialsError:
                print("\n[AUTHENTICATION ERROR] Google Cloud Authentication failed.")
                print("Please ensure your credentials are set up correctly by following the README.")
                print("Skipping translation.")
            except Exception as e:
                print(f"\nAn unexpected error occurred during translation: {e}")
                print("Skipping translation.")
        elif not GOOGLE_LIBS_AVAILABLE and translate_choice != 'n':
            print("\nGoogle Cloud libraries not found. Skipping translation option.")
            print("To enable translation, run: pip install google-cloud-vision google-generativeai")

        create_pdf_from_images(image_paths, pdf_filename)

    cleanup(temp_dir)


if __name__ == "__main__":
    main()

