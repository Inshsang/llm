query = "D) laptop is situated on the left and in front of clothesdryer.                     "
inst = (
    "[AGENT_INTENT]\n"
    f"UserQuestion: {query}\n\n"
)
prompt = (
    f"{inst}"
    f"\n### gpt:anything\n###"
)
print("[DEBUG]\n" + prompt)
