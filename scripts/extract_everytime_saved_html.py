import argparse
import csv
import email
import html
import os
import re
import urllib.parse
from collections import defaultdict
from email import policy
from html.parser import HTMLParser


DEFAULT_INPUT_FILES = []  # 추출할 Everytime HTML/MHTML 파일 경로들
DEFAULT_OUTPUT = ""  # 추출 결과 CSV output 경로


def class_tokens(attrs):
    class_name = attrs.get("class", "")
    return set(class_name.split())


class ReviewHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.reviews = []
        self.current = None
        self.depth = 0
        self.capture = None
        self.capture_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        tokens = class_tokens(attrs)

        if tag == "div" and "article" in tokens and self.current is None:
            self.current = {
                "semester": "",
                "rating": "",
                "raw_review_text": "",
            }
            self.depth = 1
            return

        if self.current is not None:
            self.depth += 1

            if tag == "span" and "on" in tokens and "style" in attrs:
                match = re.search(r"width:\s*(\d+)%", attrs["style"])
                if match:
                    self.current["rating"] = str(round(int(match.group(1)) / 20))

            if tag == "span" and "semester" in tokens:
                self.capture = "semester"
                self.capture_depth = self.depth

            if tag == "div" and "text" in tokens:
                self.capture = "raw_review_text"
                self.capture_depth = self.depth

    def handle_endtag(self, tag):
        if self.current is None:
            return

        if self.capture and self.depth == self.capture_depth:
            self.capture = None
            self.capture_depth = 0

        self.depth -= 1
        if self.depth == 0:
            if self.current["raw_review_text"].strip():
                self.current["semester"] = clean_text(self.current["semester"])
                self.current["raw_review_text"] = clean_text(
                    self.current["raw_review_text"]
                )
                self.reviews.append(self.current)
            self.current = None

    def handle_data(self, data):
        if self.current is None or self.capture is None:
            return
        self.current[self.capture] += data


def clean_text(value):
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def extract_initial_state(source):
    match = re.search(
        r'<script id="__INITIAL_STATE__" type="text/plain">(.*?)</script>',
        source,
        re.S,
    )
    if not match:
        metadata = extract_page_metadata(source)
        source_url = extract_source_url(source)
        lecture_id = re.search(r"/lecture/view/(\d+)", source_url)
        if lecture_id:
            metadata["lecture_id"] = lecture_id.group(1)
        return metadata

    decoded = urllib.parse.unquote(match.group(1))
    course = re.search(r'"name":"([^"]+)"', decoded)
    professor = re.search(r'"professor":"([^"]*)"', decoded)
    lecture_id = re.search(r'"id":(\d+)', decoded)
    return {
        "lecture_id": lecture_id.group(1) if lecture_id else "",
        "course_name": course.group(1) if course else "",
        "professor": professor.group(1) if professor else "",
    }


def extract_page_metadata(source):
    title = re.search(r"<title>(.*?)</title>", source, re.S)
    course_name = ""
    if title:
        title_text = clean_text(title.group(1))
        course_name = title_text.replace(" 강의실 - 에브리타임", "")
    return {
        "lecture_id": "",
        "course_name": course_name,
        "professor": "",
    }


def extract_meta_url(source):
    match = re.search(r'<meta[^>]+property="og:url"[^>]+content="([^"]+)"', source)
    if match:
        return html.unescape(match.group(1))
    return ""


def extract_source_url(source):
    match = re.search(r"saved from url=\(\d+\)(.*?) -->", source)
    if match:
        return match.group(1).strip()
    return extract_meta_url(source)


def get_text_part(part):
    payload = part.get_payload(decode=True)
    if payload is not None:
        header = payload[:500].decode("ascii", errors="ignore")
        match = re.search(r"charset=([\w-]+)", header, re.I)
        charset = part.get_content_charset() or (match.group(1) if match else "utf-8")
        return payload.decode(charset, errors="replace")

    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except Exception:
        pass
    return ""


def read_sources(input_path):
    if input_path.lower().endswith((".mhtml", ".mht")):
        with open(input_path, "rb") as file:
            message = email.message_from_binary_file(file, policy=policy.default)

        sources = []
        for part in message.walk():
            if part.get_content_type() == "text/html":
                sources.append(get_text_part(part))
        return sources

    with open(input_path, encoding="utf-8") as file:
        return [file.read()]


def parse_reviews(source):
    parser = ReviewHTMLParser()
    parser.feed(source)
    metadata = extract_initial_state(source)
    source_url = extract_source_url(source)
    return metadata, source_url, parser.reviews


def read_reviews(input_paths):
    rows = []
    seen = set()
    lecture_counts = defaultdict(int)

    for input_path in input_paths:
        for source in read_sources(input_path):
            metadata, source_url, reviews = parse_reviews(source)
            if not reviews:
                continue

            for review in reviews:
                dedupe_key = (
                    metadata.get("lecture_id", ""),
                    review["semester"],
                    review["rating"],
                    review["raw_review_text"],
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                lecture_id = metadata.get("lecture_id", "")
                lecture_counts[lecture_id] += 1

                rows.append(
                    {
                        "review_id": f"{lecture_id}-{lecture_counts[lecture_id]:03d}",
                        "source_url": source_url,
                        "lecture_id": lecture_id,
                        "course_name": metadata.get("course_name", ""),
                        "professor": metadata.get("professor", ""),
                        "semester": review["semester"],
                        "rating": review["rating"],
                        "raw_review_text": review["raw_review_text"],
                        "difficulty_label": "",
                        "workload_label": "",
                        "grading_strictness_label": "",
                    }
                )
    return rows


def write_csv(output_path, rows):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "review_id",
        "source_url",
        "lecture_id",
        "course_name",
        "professor",
        "semester",
        "rating",
        "raw_review_text",
        "difficulty_label",
        "workload_label",
        "grading_strictness_label",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Everytime lecture reviews from saved HTML or MHTML files."
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        help="Saved Everytime HTML/MHTML file paths.",
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="input_options",
        default=[],
        metavar="PATH",
        help="Saved Everytime HTML/MHTML file path. Can be used multiple times.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="CSV file path to write extracted Everytime reviews.",
    )
    args = parser.parse_args()

    input_paths = DEFAULT_INPUT_FILES + args.input_files + args.input_options
    if not input_paths:
        parser.error(
            "HTML/MHTML 파일 경로를 지정하거나 DEFAULT_INPUT_FILES에 추출할 파일 경로를 입력하세요."
        )
    if not args.output:
        parser.error("--output 경로를 지정하거나 DEFAULT_OUTPUT에 저장할 CSV 경로를 입력하세요.")

    input_paths = [os.path.expanduser(path) for path in input_paths]
    output_path = os.path.expanduser(args.output)

    rows = read_reviews(input_paths)
    write_csv(output_path, rows)
    print(f"wrote {len(rows)} reviews to {output_path}")


if __name__ == "__main__":
    main()
