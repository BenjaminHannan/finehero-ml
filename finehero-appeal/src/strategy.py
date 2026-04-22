# strategy.py - Uses Claude to select the strongest NYC DOF dispute grounds
# and build a legal argument strategy from the ticket details + evidence.

import anthropic

client = anthropic.Anthropic()

NYC_DISPUTE_GROUNDS = """
NYC Department of Finance - Valid Grounds to Dispute a Parking Ticket:

1. SIGN MISSING OR BLOCKED: No posted sign, sign not visible, sign obscured, or conflicting signs.
2. SIGN UNREADABLE: Sign text too faded, damaged, or otherwise illegible.
3. METER MALFUNCTION: Meter not working, meter head missing, meter displayed wrong time, muni-meter receipt issue.
4. LEGALLY PARKED: Vehicle was lawfully parked under another applicable rule.
5. WRONG PLATE / VEHICLE INFO: Ticket has incorrect plate number, state, or vehicle description.
6. VEHICLE SOLD: Vehicle was sold before the ticket date.
7. STOLEN VEHICLE: Vehicle was reported stolen at time of violation.
8. EMERGENCY: Medical emergency, mechanical breakdown, or other unavoidable circumstance.
9. PRECINCT CORRECTION: Ticketed in one precinct but registered in another jurisdiction with different rules.
10. INCORRECT DATE/TIME: Ticket date or time recorded incorrectly by officer.
11. DOUBLE-PARKED BY NECESSITY: Blocked by another vehicle, delivery emergency, etc.
12. AUTHORIZED VEHICLE: Government plates, press plates, or other authorized exemption.
13. LOADING/UNLOADING: Commercial vehicle loading/unloading within allowed time.
14. HANDICAPPED PLACARD: Valid placard displayed but officer failed to note it.
15. OWNER vs OPERATOR: Owner was not driving; operator liability argument.
"""


def select_strategy(ticket: dict, narrative: str, exhibits: list[dict]) -> dict:
    """
    Ask Claude to pick the strongest dispute grounds and draft
    an argument outline based on ticket fields + evidence descriptions.
    """
    exhibit_summary = ""
    for ex in exhibits:
        exhibit_summary += f"\nExhibit {ex['exhibit_num']} ({ex['filename']}): {ex['description']}"

    prompt = f"""
You are an expert NYC parking ticket paralegal. Analyze the following ticket and evidence,
then select the strongest dispute grounds from the NYC DOF official list.

--- TICKET DETAILS ---
Summons Number:  {ticket.get('summons_number', 'N/A')}
Violation Code:  {ticket.get('violation_code', 'N/A')}
Violation Desc:  {ticket.get('violation_description', 'N/A')}
Issue Date:      {ticket.get('issue_date', 'N/A')}
Issue Time:      {ticket.get('violation_time', 'N/A')}
Fine Amount:     ${ticket.get('fine_amount', 'N/A')}
Precinct:        {ticket.get('precinct', 'N/A')}
County/Borough:  {ticket.get('county', 'N/A')}
Issuing Agency:  {ticket.get('issuing_agency', 'N/A')}
Street Address:  {ticket.get('street_address', 'N/A')}

--- DRIVER NARRATIVE ---
{narrative or 'No narrative provided.'}

--- SUBMITTED EVIDENCE ---
{exhibit_summary or 'No evidence submitted.'}

--- VALID GROUNDS ---
{NYC_DISPUTE_GROUNDS}

Respond in this exact format:

PRIMARY_GROUND: [ground number and name from list above]
SECONDARY_GROUND: [ground number and name, or NONE]
ARGUMENT_SUMMARY: [2-3 sentences summarizing the strongest argument]
KEY_FACTS: [bullet-pointed list of the most important facts from evidence + narrative]
STRENGTH: [STRONG / MODERATE / WEAK] — your honest assessment of this dispute's chances
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    def _extract(field: str) -> str:
        for line in raw.splitlines():
            if line.startswith(field + ":"):
                return line.split(":", 1)[1].strip()
        return ""

    def _extract_block(start_field: str, end_field: str) -> str:
        lines = raw.splitlines()
        capturing = False
        block = []
        for line in lines:
            if line.startswith(start_field + ":"):
                capturing = True
                rest = line.split(":", 1)[1].strip()
                if rest:
                    block.append(rest)
                continue
            if capturing:
                if any(line.startswith(f + ":") for f in [end_field, "ARGUMENT_SUMMARY", "KEY_FACTS", "STRENGTH", "PRIMARY_GROUND", "SECONDARY_GROUND"]):
                    break
                block.append(line)
        return "\n".join(block).strip()

    return {
        "primary_ground": _extract("PRIMARY_GROUND"),
        "secondary_ground": _extract("SECONDARY_GROUND"),
        "argument_summary": _extract("ARGUMENT_SUMMARY"),
        "key_facts": _extract_block("KEY_FACTS", "STRENGTH"),
        "strength": _extract("STRENGTH"),
        "raw_strategy": raw,
    }
