# سنڌي هه (Heh) Unicode rules

Canonical forms used by `heh_normalize.py` for labels, synth text, and post-OCR cleanup.

| Glyph | Unicode name | Code | Role |
|-------|--------------|------|------|
| ھ | ARABIC LETTER HEH DOACHASHMEE | U+06BE | Start / middle strong heh; also final after ا |
| ه | ARABIC LETTER HEH | U+0647 | After ج or گ (light / aspirated digraph) |
| ہ | ARABIC LETTER HEH GOAL | U+06C1 | Normal word-final heh (not after ا) |

## Precedence (apply in order per heh slot)

1. **After ج or گ → ه (U+0647)**  
   Examples: جهنگ، جهولي، سمجه، گهر، گهوڙو، سگه

2. **Word-final after ا (alif family) → ھ (U+06BE)**  
   Alif family: ا آ أ إ ٱ  
   Examples: واھ، شاھ، باھ، راھ، پناھ، گواھ، درگاھ، سياھ، تباھ

3. **Other word-final → ہ (U+06C1)**  
   Examples: نہ، تہ، بہ، ڳالہ، ھميشہ، ڪجھہ

4. **Start or middle (not covered above) → ھ (U+06BE)**  
   Examples: ھٿ، ھو، ھتي، سھڻو، مھرباني، شھيد، پڙھڻ

## Notes

- Combining marks (zabar, zer, sukun, …) are skipped when finding the previous letter and when deciding “final”.
- Any of ه / ھ / ہ in the input is rewritten to the canonical form for its slot.
- Use the same normalizer on corpus lines **before** synth render, on gold labels, and on model predictions before CER / review.
