# appeal.py - FineHero Appeal Letter Generator
#
# Generates a formatted PDF appeal letter for an NYC parking ticket,
# incorporating user narrative and AI analysis of submitted evidence.
#
# Usage:
#   python appeal.py
#   python appeal.py --ticket ticket.json --evidence photo1.jpg receipt.pdf --out output/appeal.pdf
#
# Requirements:
#   pip install -r requirements.txt
#   Set ANTHROPIC_API_KEY in your environment.

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

# Add src/ to path so sibling imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evidence import process_evidence
from strategy import select_strategy
from letter import generate_letter
from pdf_gen import build_pdf


def _prompt(label: str, default: str = "") -> str:
    val = input(f"  {label}" + (f" [{default}]" if default else "") + ": ").strip()
    return val if val else default


def _collect_ticket_interactive() -> dict:
    print("\n  --- Ticket Details ---")
    return {
        "summons_number":       _prompt("Summons number"),
        "violation_code":       _prompt("Violation code"),
        "violation_description":_prompt("Violation description", "Parking violation"),
        "issue_date":           _prompt("Issue date (YYYY-MM-DD)"),
        "violation_time":       _prompt("Violation time"),
        "fine_amount":          _prompt("Fine amount"),
        "precinct":             _prompt("Precinct"),
        "county":               _prompt("Borough/County"),
        "issuing_agency":       _prompt("Issuing agency", "TRAFFIC"),
        "street_address":       _prompt("Street address on ticket"),
        "plate":                _prompt("Plate number"),
        "plate_state":          _prompt("Plate state", "NY"),
        "vehicle_year":         _prompt("Vehicle year"),
        "vehicle_make":         _prompt("Vehicle make"),
        "vehicle_color":        _prompt("Vehicle color"),
    }


def _collect_owner_interactive() -> dict:
    print("\n  --- Your Information ---")
    return {
        "name":    _prompt("Full name"),
        "address": _prompt("Mailing address"),
        "phone":   _prompt("Phone number"),
        "email":   _prompt("Email address"),
    }


def _collect_evidence_interactive() -> list[str]:
    print("\n  --- Evidence Files ---")
    print("  Enter file paths one per line (photos, PDFs, text files).")
    print("  Press Enter on an empty line when done.\n")
    paths = []
    while True:
        p = input("  File path (or Enter to finish): ").strip().strip('"')
        if not p:
            break
        if not os.path.exists(p):
            print(f"    [WARN] File not found: {p}")
        else:
            paths.append(p)
    return paths


def _collect_narrative_interactive() -> str:
    print("\n  --- Your Statement ---")
    print("  Describe in your own words why this ticket is wrong.")
    print("  Press Enter twice when done.\n")
    lines = []
    while True:
        line = input("  > ")
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run(ticket: dict, owner: dict, narrative: str, evidence_paths: list[str], output_path: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("  Get your key at https://console.anthropic.com and run:")
        print("    set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print("\n" + "=" * 56)
    print("  FineHero Appeal Letter Generator")
    print("=" * 56)

    # 1. Process evidence
    exhibits = []
    if evidence_paths:
        print(f"\n  Processing {len(evidence_paths)} evidence file(s)...")
        exhibits = process_evidence(evidence_paths)
        print(f"  {len(exhibits)} exhibit(s) processed.")
    else:
        print("\n  No evidence files provided — letter will be argument-only.")

    # 2. Select legal strategy
    print("\n  Selecting dispute strategy...")
    strategy = select_strategy(ticket, narrative, exhibits)
    print(f"  Primary ground:  {strategy['primary_ground']}")
    print(f"  Dispute strength: {strategy['strength']}")

    # 3. Generate letter
    print("\n  Drafting appeal letter...")
    letter_text = generate_letter(ticket, owner, strategy, exhibits)
    print(f"  Letter generated ({len(letter_text.split())} words).")

    # 4. Build PDF
    print(f"\n  Building PDF -> {output_path}")
    build_pdf(letter_text, exhibits, output_path, ticket)
    size_kb = os.path.getsize(output_path) // 1024
    print(f"  PDF saved: {output_path} ({size_kb} KB)")

    print("\n" + "=" * 56)
    print(f"  DONE - {len(exhibits)} exhibit(s) attached")
    print(f"  Dispute strength: {strategy['strength']}")
    print(f"  Output: {output_path}")
    print("=" * 56)
    print("\n  Next steps:")
    print("  1. Review the letter and make any corrections")
    print("  2. Submit via: nyc.gov/dof  (online dispute form)")
    print("     OR mail to: NYC Department of Finance")
    print("                 P.O. Box 3600, New York, NY 10008-3600")
    print("  3. Keep a copy of everything for your records\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FineHero - Generate an NYC parking ticket appeal letter with AI."
    )
    parser.add_argument("--ticket",   help="Path to ticket JSON file")
    parser.add_argument("--owner",    help="Path to owner info JSON file")
    parser.add_argument("--evidence", nargs="*", default=[], help="Evidence file paths")
    parser.add_argument("--narrative", default="", help="Your explanation (string or path to .txt)")
    parser.add_argument("--out",      default="", help="Output PDF path")
    args = parser.parse_args()

    # Ticket
    if args.ticket and os.path.exists(args.ticket):
        with open(args.ticket) as f:
            ticket = json.load(f)
    else:
        ticket = _collect_ticket_interactive()

    # Owner
    if args.owner and os.path.exists(args.owner):
        with open(args.owner) as f:
            owner = json.load(f)
    else:
        owner = _collect_owner_interactive()

    # Narrative
    if args.narrative:
        if os.path.exists(args.narrative):
            with open(args.narrative) as f:
                narrative = f.read()
        else:
            narrative = args.narrative
    else:
        narrative = _collect_narrative_interactive()

    # Output path
    if args.out:
        output_path = args.out
    else:
        summons = ticket.get("summons_number", "appeal").replace(" ", "_")
        today = date.today().strftime("%Y%m%d")
        output_path = os.path.join(
            os.path.dirname(__file__), "output", f"appeal_{summons}_{today}.pdf"
        )

    run(ticket, owner, narrative, args.evidence, output_path)


if __name__ == "__main__":
    main()
