from .Char.typo_transform import TypoTransform
from .Char.glyph_transform import GlyphTransform
from .Char.natural_transform import NaturalTransform
from .Char.phonetic_transform import PhoneticTransform
from .Word.synonym_transform import SynonymTransform
from .Word.contextual_transform import ContextualTransform
from .Sentence.distraction_transform import DistractionTransform
from .Word.inflection_transform import InflectionTransform


transformer_dict = {
    "typo": TypoTransform,
    "glyph": GlyphTransform,
    "natural": NaturalTransform,
    "phonetic": PhoneticTransform,
    "synonym": SynonymTransform,
    "contextual": ContextualTransform,
    "distraction": DistractionTransform, 
    "inflect": InflectionTransform
}




def load_rule_transformer(name, degree, aug_num,dataset, dis_type="char"):
    if name == "synonym" or name=="contextual" or name=="inflect":
        return transformer_dict[name](degree, aug_num)
    elif name == "distraction":
        return transformer_dict[name](degree, aug_num, dataset)
    else:
        return transformer_dict[name](degree, aug_num,dis_type)
