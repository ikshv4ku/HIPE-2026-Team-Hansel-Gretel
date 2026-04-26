"""
Strategy 8: Chain-of-Thought + Few-Shot Prompting via Large LLM APIs.

This module contains the master prompt used for all backends.

Design rationale
────────────────
• Few-shot examples anchor the model to the task vocabulary (TRUE / PROBABLE / FALSE).
• Chain-of-Thought (CoT) reasoning steps force the model to reason explicitly before
  producing the final structured answer, dramatically reducing hallucination.
• The three reasoning steps mirror the specialist agents in Strategy 6/7 but in a
  single, unified prompt — no fine-tuning required at all.
• JSON output format is enforced so the parser can extract results deterministically.
"""

SYSTEM_PROMPT = """\
You are an expert historian and geographer specialising in 19th–20th century European
newspapers and their historical context. Your task is to determine, for a given
PERSON–LOCATION pair extracted from a historical newspaper article, two binary
relation labels:

  • at   – Was the person EVER at this location before the article date?
             TRUE     = there is explicit or strong inferential evidence.
             PROBABLE = the evidence is suggestive but not definitive.
             FALSE    = no credible evidence links them to this place.

  • isAt – Was the person at this location WITHIN ~1 MONTH of the publication date?
             TRUE  = the text or context confirms very recent / current presence.
             FALSE = there is no such evidence (or at=FALSE, which forces isAt=FALSE).

Hard rule: if at=FALSE then isAt MUST be FALSE.

──────────────────────────────────────────────────────────────────────────────
REASONING PROCESS (Chain-of-Thought)
──────────────────────────────────────────────────────────────────────────────
For each pair you MUST work through THREE explicit steps before answering:

Step 1 – BIOGRAPHICAL & TEMPORAL ANALYSIS
  Consider the person's known career, role, nationality, and life timeline
  relative to the publication date. Do dates mentioned constrain presence?

Step 2 – GEOGRAPHIC & CONTEXTUAL ANALYSIS
  Consider the article's publication language and source, named places,
  institutional affiliations, and any spatial language in the text.
  Does the regional context imply the person was there?

Step 3 – SYNTHESIS & DECISION
  Weigh both analyses. Apply the hard rule. State the final verdicts.

──────────────────────────────────────────────────────────────────────────────
OUTPUT FORMAT (strictly JSON, no extra text after the JSON block)
──────────────────────────────────────────────────────────────────────────────
{
  "step1_biographical": "<your biographical/temporal reasoning>",
  "step2_geographic":   "<your geographic/contextual reasoning>",
  "step3_synthesis":    "<your synthesis and decision>",
  "at":   "<TRUE|PROBABLE|FALSE>",
  "isAt": "<TRUE|FALSE>"
}
"""

# ── Few-shot examples ──────────────────────────────────────────────────────────
# These are embedded directly in the user turn so they work with all API types
# (chat-completion APIs, completion APIs, IBM watsonx, etc.)

FEW_SHOT_EXAMPLES = [
    # ── Example 1: at=TRUE, isAt=TRUE ─────────────────────────────────────────
    {
        "input": {
            "date": "1902-03-15",
            "language": "fr",
            "source": "Le Journal de Genève",
            "person": "Émile Zola",
            "location": "Paris",
            "text": (
                "M. Émile Zola est de retour à Paris après son séjour forcé en Angleterre. "
                "Il a été aperçu hier soir au théâtre de l'Odéon en compagnie de plusieurs "
                "journalistes parisiens. L'écrivain a déclaré reprendre ses travaux littéraires."
            ),
        },
        "output": {
            "step1_biographical": (
                "Émile Zola (1840–1902) was a French novelist who lived in Paris for most of his "
                "career. He fled to England in 1898 after the Dreyfus affair but returned to France "
                "in 1899. By March 1902 he was back in Paris, which is consistent with his timeline."
            ),
            "step2_geographic": (
                "The article is a French-language Swiss newspaper reporting Zola's return TO Paris. "
                "The phrase 'est de retour à Paris' and the specific sighting at the Odéon theatre "
                "spatially anchor him in Paris within the last 24 hours of the publication date."
            ),
            "step3_synthesis": (
                "Both analyses concur: Zola is explicitly in Paris and was seen there the day "
                "before publication. at=TRUE (he has always been associated with Paris). "
                "isAt=TRUE (confirmed presence within days of the article date)."
            ),
            "at": "TRUE",
            "isAt": "TRUE",
        },
    },
    # ── Example 2: at=PROBABLE, isAt=FALSE ────────────────────────────────────
    {
        "input": {
            "date": "1885-11-04",
            "language": "de",
            "source": "Neue Zürcher Zeitung",
            "person": "Otto von Bismarck",
            "location": "Wien",
            "text": (
                "Der Reichskanzler Bismarck hat in seiner langen Karriere mehrfach diplomatische "
                "Verhandlungen mit österreichischen Vertretern geführt, zuletzt im Rahmen des "
                "Drei-Kaiser-Abkommens. Seine Außenpolitik gilt als Meisterwerk europäischer "
                "Diplomatie. Derzeit weilt er auf seinem Gut in Friedrichsruh."
            ),
        },
        "output": {
            "step1_biographical": (
                "Bismarck (1815–1898) was German Chancellor and conducted extensive diplomatic "
                "relations with Austria throughout his career, including the Three Emperors' "
                "League. Such negotiations would plausibly have required visits to Vienna. "
                "However, the article states he is currently on his estate in Friedrichsruh, "
                "explicitly placing him elsewhere in November 1885."
            ),
            "step2_geographic": (
                "The article references past diplomatic meetings with Austrian representatives "
                "and the Three Emperors' League, which implies historical presence in or contact "
                "with Wien. No current geographic anchor to Wien is given — in fact the opposite."
            ),
            "step3_synthesis": (
                "Historical diplomatic role implies PROBABLE past presence in Wien (at=PROBABLE). "
                "The article explicitly states he is at Friedrichsruh now, ruling out recent "
                "presence in Wien (isAt=FALSE)."
            ),
            "at": "PROBABLE",
            "isAt": "FALSE",
        },
    },
    # ── Example 3: at=FALSE, isAt=FALSE ───────────────────────────────────────
    {
        "input": {
            "date": "1910-06-22",
            "language": "en",
            "source": "The Manchester Guardian",
            "person": "Victor Hugo",
            "location": "London",
            "text": (
                "The city mourns the passing of a great labour reformer. Local trade unions "
                "gathered at Hyde Park yesterday to demand better working conditions for textile "
                "workers in Manchester. Several MPs have pledged support for the new bill."
            ),
        },
        "output": {
            "step1_biographical": (
                "Victor Hugo (1802–1885) was a French poet and novelist who died 25 years before "
                "this article. While he did live in exile in the Channel Islands and briefly in "
                "Brussels, he is not biographically linked to London. His death in 1885 rules out "
                "any presence in 1910."
            ),
            "step2_geographic": (
                "The article is entirely about a Manchester labour rally at Hyde Park. Victor Hugo "
                "is not mentioned anywhere in the text. There is no geographic or contextual signal "
                "connecting him to London."
            ),
            "step3_synthesis": (
                "No biographical or textual evidence connects Victor Hugo to London. He is not "
                "mentioned in the article. at=FALSE; isAt=FALSE (forced by the hard rule)."
            ),
            "at": "FALSE",
            "isAt": "FALSE",
        },
    },
    # ── Example 4: at=TRUE, isAt=FALSE ────────────────────────────────────────
    {
        "input": {
            "date": "1898-07-10",
            "language": "fr",
            "source": "Journal de Genève",
            "person": "Ferdinand de Lesseps",
            "location": "Le Caire",
            "text": (
                "On commémore aujourd'hui le vingtième anniversaire de l'inauguration du Canal "
                "de Suez, grande œuvre à laquelle Ferdinand de Lesseps a consacré sa vie. "
                "Le ingénieur, aujourd'hui décédé, reste une figure incontournable de l'histoire "
                "de l'Égypte moderne."
            ),
        },
        "output": {
            "step1_biographical": (
                "Ferdinand de Lesseps (1805–1894) directed the construction of the Suez Canal "
                "and spent many years in Egypt, headquartered in or near Cairo (Le Caire). He died "
                "in 1894, four years before this article. His biographical record firmly links him "
                "to Cairo through his long presence during canal construction."
            ),
            "step2_geographic": (
                "The article commemorates the Suez Canal anniversary and calls de Lesseps a key "
                "figure in modern Egyptian history. Cairo is the capital closest to his operations. "
                "The geographic and institutional context strongly implies historical presence there."
            ),
            "step3_synthesis": (
                "De Lesseps was historically and biographically present in Cairo during his "
                "Suez Canal work (at=TRUE). He died in 1894, so he cannot be there in 1898 "
                "(isAt=FALSE)."
            ),
            "at": "TRUE",
            "isAt": "FALSE",
        },
    },
]


def build_user_message(date: str, language: str, source: str,
                       person: str, location: str, text: str,
                       include_few_shot: bool = True) -> str:
    """
    Build the user-turn message, optionally prepending few-shot examples.
    Each example is formatted as a Q/A block so the model learns the exact
    output format from demonstration.
    """
    parts = []

    if include_few_shot:
        parts.append("DEMONSTRATION EXAMPLES\n" + "=" * 60)
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            inp = ex["input"]
            out = ex["output"]
            import json
            parts.append(
                f"\n--- Example {i} ---\n"
                f"Publication date: {inp['date']} | Language: {inp['language']} "
                f"| Source: {inp['source']}\n"
                f"PERSON: {inp['person']}\n"
                f"LOCATION: {inp['location']}\n"
                f"ARTICLE TEXT:\n{inp['text']}\n\n"
                f"RESPONSE:\n{json.dumps(out, indent=2, ensure_ascii=False)}"
            )
        parts.append("\n" + "=" * 60 + "\nNOW CLASSIFY THE FOLLOWING:\n")

    parts.append(
        f"Publication date: {date} | Language: {language} | Source: {source}\n"
        f"PERSON: {person}\n"
        f"LOCATION: {location}\n\n"
        f"ARTICLE TEXT:\n{text}\n\n"
        f"Respond ONLY with a valid JSON object. No extra text outside the JSON."
    )

    return "\n".join(parts)
