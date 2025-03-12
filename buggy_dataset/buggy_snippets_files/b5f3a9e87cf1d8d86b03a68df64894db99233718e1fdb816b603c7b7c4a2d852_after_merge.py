    def analyze(self, text: str) -> Doc:
        """The primary method for the NLP object, to which raw text strings are passed.

        >>> from cltkv1 import NLP
        >>> from cltkv1.languages.example_texts import get_example_text
        >>> from cltkv1.core.data_types import Doc
        >>> cltk_nlp = NLP(language="lat")
        >>> cltk_doc = cltk_nlp.analyze(text=get_example_text("lat"))
        >>> isinstance(cltk_doc, Doc)
        True
        >>> cltk_doc.words[0] # doctest: +ELLIPSIS
        Word(index_char_start=None, index_char_stop=None, index_token=0, index_sentence=0, string='Gallia', pos='NOUN', lemma='mallis', scansion=None, xpos='A1|grn1|casA|gen2', upos='NOUN', dependency_relation='nsubj', governor=3, features={'Case': 'Nom', 'Degree': 'Pos', 'Gender': 'Fem', 'Number': 'Sing'}, embedding=..., stop=False, named_entity=True)

        >>> from cltkv1.languages.example_texts import get_example_text
        >>> cltk_nlp = NLP(language="grc")
        >>> cltk_doc = cltk_nlp.analyze(text=get_example_text("grc"))
        >>> cltk_doc.words[0] # doctest: +ELLIPSIS
        Word(index_char_start=None, index_char_stop=None, index_token=0, index_sentence=0, string='ὅτι', pos='ADV', lemma='ὅτι', scansion=None, xpos='Df', upos='ADV', dependency_relation='advmod', governor=6, features={}, embedding=..., stop=True, named_entity=False)

        >>> cltk_nlp = NLP(language="chu")
        >>> cltk_doc = cltk_nlp.analyze(text=get_example_text("chu"))
        >>> cltk_doc.words[0] # doctest: +ELLIPSIS
        Word(index_char_start=None, index_char_stop=None, index_token=0, index_sentence=0, string='отьчє', pos='NOUN', lemma='отьць', scansion=None, xpos='Nb', upos='NOUN', dependency_relation='vocative', governor=7, features={'Case': 'Voc', 'Gender': 'Masc', 'Number': 'Sing'}, embedding=None, stop=None, named_entity=None)

        >>> cltk_nlp = NLP(language="fro")
        >>> cltk_doc = cltk_nlp.analyze(text=get_example_text("fro"))
        >>> cltk_doc.words[0] # doctest: +ELLIPSIS
        Word(index_char_start=None, index_char_stop=None, index_token=0, index_sentence=0, string='Une', pos='DET', lemma=None, scansion=None, xpos='DETndf', upos='DET', dependency_relation=None, governor=-1, features={'Definite': 'Ind', 'PronType': 'Art'}, embedding=None, stop=False, named_entity=False)

        >>> cltk_nlp = NLP(language="got")
        >>> cltk_doc = cltk_nlp.analyze(text=get_example_text("got"))
        >>> cltk_doc.words[0] # doctest: +ELLIPSIS
        Word(index_char_start=None, index_char_stop=None, index_token=0, index_sentence=0, string='swa', pos='ADV', lemma='swa', scansion=None, xpos='Df', upos='ADV', dependency_relation='advmod', governor=1, features={}, embedding=..., stop=None, named_entity=None)
        >>> len(cltk_doc.sentences)
        3

        # >>> cltk_nlp = NLP(language="cop")
        # >>> cltk_doc = cltk_nlp.analyze(text=get_example_text("cop"))
        # >>> cltk_doc.words[0] # doctest: +ELLIPSIS
        # Word(index_char_start=None, index_char_stop=None, index_token=0, index_sentence=0, string='ⲧⲏⲛ', pos='VERB', lemma='ⲧⲏⲛ', scansion=None, xpos='VSTAT', upos='VERB', dependency_relation='root', governor=-1, features={'VerbForm': 'Fin'}, embedding=None, stop=None, named_entity=None)

        >>> cltk_nlp = NLP(language="lzh")
        >>> cltk_doc = cltk_nlp.analyze(text=get_example_text("lzh"))
        >>> cltk_doc.words[0] # doctest: +ELLIPSIS
        Word(index_char_start=None, index_char_stop=None, index_token=0, index_sentence=0, string='黃', pos='NOUN', lemma='黃', scansion=None, xpos='n,名詞,描写,形質', upos='NOUN', dependency_relation='nmod', governor=1, features={}, embedding=None, stop=None, named_entity=None)
        """
        doc = Doc(language=self.language.iso_639_3_code, raw=text)

        for process in self.pipeline.processes:
            a_process = process(input_doc=doc, language=self.language.iso_639_3_code)
            a_process.run()
            doc = a_process.output_doc

        return doc