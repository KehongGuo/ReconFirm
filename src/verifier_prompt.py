"""The verifier's question: name the object in the centre of a crop from a closed vocabulary, without being told
what the detector said (so its errors stay largely independent of the detector's), say whether the view is clear, and
flag reflections. Answers are parsed in code; anything unparseable is reported, not guessed.
"""
from __future__ import annotations
import re

# The detector's own vocabulary, so the two models answer over the same options.
VOCABULARY = ["chair", "table", "sofa", "bed", "cabinet", "plant", "picture", "tv_monitor"]

PROMPT = """Look at the object in the centre of this image and answer in exactly this format, with no other text:

CATEGORY: <one of: {options}, or none>
CLEAR: <yes if the image is sharp and large enough to identify the centre object, no if it is not>
DESCRIPTION: <one sentence describing the centre object and what surrounds it>
REFLECTION: <yes if the centre object is a reflection in a mirror or glass, otherwise no>

Choose "none" if the centre of the image is a wall, a floor, a ceiling, empty space, or an object that is not in the list."""
MIRROR = ('If the centre object is seen in a mirror or another reflective surface, it is only a '
          'reflection: answer CATEGORY: none, say in the description that it is a reflection, and '
          'answer REFLECTION: yes, since the robot should turn around.')
PICTURE = ('If the centre object is only depicted, in a painting, photo, poster or on a screen, it is not '
           'a real object: answer CATEGORY: picture and say in the description what it depicts.')
SMALL = ('Small or low-detail objects (for example potted plants and flowerpots) often look blurry in '
         'these images. If the shape and colour fit a category, name that category and answer CLEAR: no, '
         'rather than answering none because of the blur.')
PROMPT = PROMPT + '\n' + MIRROR + '\n' + PICTURE + '\n' + SMALL


def build_prompt(vocabulary=None):
    return PROMPT.format(options=', '.join(vocabulary or VOCABULARY))


def canonical(word, vocabulary=None):
    """Map a free-text answer onto a vocabulary entry, ignoring case, spaces and
    underscores, so "chest of drawers", "chest_of_drawers" and "Chest of Drawers"
    are one answer. Anything else stays flagged with a leading '?'."""
    key = word.strip().strip('.').strip().lower().replace('_', ' ')
    key = ' '.join(key.split())
    if key == 'none':
        return 'none'
    for entry in (vocabulary or VOCABULARY):
        if ' '.join(entry.lower().replace('_', ' ').split()) == key:
            return entry
    return '?' + key.replace(' ', '_')


def field(name, line):
    """Match `NAME: value`, tolerating the markdown an instruct model adds anyway."""
    m = re.match(rf'[\s*_#>-]*{name}\s*[:：]?[\s*_]*[:：]?\s*(.+)', line, re.I)
    return m.group(1).strip().strip('*_ ').strip() if m else None


def parse(text, vocabulary=None):
    """Pull the three fields out; anything unparseable is reported, not guessed."""
    out = {'category': None, 'clear': None, 'description': None}
    for line in text.splitlines():
        value = field('CATEGORY', line)
        if value:
            out['category'] = canonical(value, vocabulary)
        value = field('CLEAR', line)
        if value:
            out['clear'] = value.split()[0].strip('.,;:!').lower() in ('yes', 'true')
        value = field('DESCRIPTION', line)
        if value:
            out['description'] = value
    return out
