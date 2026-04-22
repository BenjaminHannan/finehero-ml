# letter.py - Generates the full appeal letter text using Claude,
# incorporating ticket details, dispute strategy, and exhibit references.

from datetime import date

import anthropic

client = anthropic.Anthropic()


def generate_letter(ticket: dict, owner: dict, strategy: dict, exhibits: list[dict]) -> str:
    """
    Generate a formal NYC DOF appeal letter.
    Returns the full letter as a plain-text string.
    """
    exhibit_refs = ""
    for ex in exhibits:
        exhibit_refs += f"\n- Exhibit {ex['exhibit_num']}: {ex['filename']} — {ex['description'][:120]}..."

    today = date.today().strftime("%B %d, %Y")

    prompt = f"""
Write a formal, professional appeal letter for an NYC parking ticket dispute.
This letter will be submitted to the NYC Department of Finance Adjudication Division.

Use this EXACT structure:
1. Owner address block (top left)
2. Date
3. NYC DOF address block
4. RE: line with summons number
5. Opening paragraph — state you are contesting the summons and why
6. Factual background — what happened, when, where
7. Legal argument — cite the dispute ground(s) and explain why the ticket should be dismissed
8. Evidence section — reference each exhibit by number and explain what it proves
9. Closing — request dismissal, offer to attend hearing
10. Signature block

--- OWNER INFO ---
Name:    {owner.get('name', 'Vehicle Owner')}
Address: {owner.get('address', '')}
Phone:   {owner.get('phone', '')}
Email:   {owner.get('email', '')}

--- TICKET INFO ---
Summons Number:  {ticket.get('summons_number', 'N/A')}
Violation:       {ticket.get('violation_code', '')} - {ticket.get('violation_description', '')}
Date/Time:       {ticket.get('issue_date', '')} at {ticket.get('violation_time', '')}
Location:        {ticket.get('street_address', 'N/A')}, {ticket.get('county', '')}
Fine Amount:     ${ticket.get('fine_amount', '')}
Plate:           {ticket.get('plate', 'N/A')} ({ticket.get('plate_state', 'NY')})
Vehicle:         {ticket.get('vehicle_year', '')} {ticket.get('vehicle_make', '')} {ticket.get('vehicle_color', '')}

--- TODAY'S DATE ---
{today}

--- DISPUTE STRATEGY ---
Primary Ground:   {strategy['primary_ground']}
Secondary Ground: {strategy['secondary_ground']}
Core Argument:    {strategy['argument_summary']}
Key Facts:
{strategy['key_facts']}

--- EXHIBIT REFERENCES ---
{exhibit_refs or 'No exhibits submitted.'}

Write the complete letter now. Use formal legal language appropriate for a DOF adjudication.
Do NOT include any commentary outside the letter itself. The letter should be ready to print and sign.
End with:
    Respectfully submitted,

    [signature line]
    {owner.get('name', 'Vehicle Owner')}
    {today}

    Enclosures: [list exhibits by number and filename]
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
