    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {iterable of :class:`~gensim.models.doc2vec.TaggedDocument`, iterable of list of str}
            A collection of tagged documents used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.d2vmodel.D2VTransformer`
            The trained model.

        """
        if isinstance([i for i in X[:1]][0], doc2vec.TaggedDocument):
            d2v_sentences = X
        else:
            d2v_sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(X)]
        self.gensim_model = models.Doc2Vec(
            documents=d2v_sentences, dm_mean=self.dm_mean, dm=self.dm,
            dbow_words=self.dbow_words, dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
            docvecs=self.docvecs, docvecs_mapfile=self.docvecs_mapfile, comment=self.comment,
            trim_rule=self.trim_rule, vector_size=self.size, alpha=self.alpha, window=self.window,
            min_count=self.min_count, max_vocab_size=self.max_vocab_size, sample=self.sample,
            seed=self.seed, workers=self.workers, min_alpha=self.min_alpha, hs=self.hs,
            negative=self.negative, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn,
            epochs=self.iter, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words
        )
        return self