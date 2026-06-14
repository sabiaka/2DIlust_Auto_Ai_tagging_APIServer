import argparse
import csv
from pathlib import Path

from imgutils.tagging.pixai import _open_tags


def normalize_tag(tag: str) -> str:
    return tag.replace("_", " ").strip()


def load_translated_tags(path: Path) -> set[str]:
    if not path.exists():
        return set()

    translated = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_tag(row.get("name", ""))
            japanese_name = row.get("japanese_name", "").strip()
            if name and japanese_name:
                translated.add(name)
    return translated


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PixAI tags missing Japanese translations.")
    parser.add_argument("--model-name", default="v0.9")
    parser.add_argument("--base-csv", default="models_data/selected_tags.csv")
    parser.add_argument("--extra-csv", default="models_data/tag_translations_extra.csv")
    parser.add_argument("--output", default="models_data/pixai_missing_translations.csv")
    parser.add_argument("--category", type=int, help="Filter by PixAI category. Character tags are category 4.")
    args = parser.parse_args()

    base_csv = Path(args.base_csv)
    extra_csv = Path(args.extra_csv)
    output = Path(args.output)

    translated_tags = load_translated_tags(base_csv) | load_translated_tags(extra_csv)
    df_tags, _ = _open_tags(args.model_name)

    rows = []
    for item in df_tags.to_dict("records"):
        if args.category is not None and int(item.get("category", -1)) != args.category:
            continue

        name = normalize_tag(str(item["name"]))
        if name not in translated_tags:
            rows.append(
                {
                    "name": item["name"],
                    "category": item.get("category", ""),
                    "count": item.get("count", ""),
                    "japanese_name": "",
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "category", "count", "japanese_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} untranslated PixAI tags to {output}")


if __name__ == "__main__":
    main()
