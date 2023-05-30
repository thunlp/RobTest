from .anthro.anthro_lib import ANTHRO
anthro = ANTHRO()
anthro.load('AttackMethod/RuleBased/Char/anthro/ANTHRO_Data_V1.0')
word = 'yes'
candidate_words = list(anthro.get_similars(word, level=1, distance=1, strict=True))
print(candidate_words)