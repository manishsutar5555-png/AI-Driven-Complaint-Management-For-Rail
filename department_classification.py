from openai import OpenAI

client = OpenAI(api_key="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")

CLASSES = [
    "Coach - Maintenance/Facilities",
    "Electrical Equipment",
    "Medical Assistance",
    "Catering / Vending Services",
    "Water Availability",
    "Punctuality",
    "Security",
    "Unreserved / Reserved Ticketing",
    "Coach - Cleanliness",
    "Staff Behaviour",
    "Refund of Tickets",
    "Booking/Reservation Issues",
    "Platform Facilities",
    "Lost & Found",
    "Baggage Handling",
    "Delay Information",
    "Comfort/Seating",
    "Sanitation",
    "Ticketing Website/App",
    "Other"
]

def predict_department(text: str) -> str:
    prompt = f"""
    You are a classifier. Pick exactly one category from the list below
    that best fits the passenger complaint.

    Categories:
    {", ".join(CLASSES)}

    Complaint: {text}
    Answer with the category only:
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    sample = "My coach is filthy and there is no water in the restroom."
    print("Complaint:", sample)
    print("Predicted category:", predict_department(sample))
