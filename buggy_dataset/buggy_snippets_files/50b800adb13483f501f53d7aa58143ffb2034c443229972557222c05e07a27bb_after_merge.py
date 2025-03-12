    def parse(self, stream, start_symbol=None):
        # Define parser functions
        start_symbol = start_symbol or self.start_symbol
        delayed_matches = defaultdict(list)

        text_line = 1
        text_column = 0

        def predict(nonterm, column):
            assert not isinstance(nonterm, Terminal), nonterm
            return [Item(rule, 0, column, None) for rule in self.predictions[nonterm]]

        def complete(item):
            name = item.rule.origin
            return [i.advance(item.tree) for i in item.start.to_predict if i.expect == name]

        def predict_and_complete(column):
            while True:
                to_predict = {x.expect for x in column.to_predict.get_news()
                              if x.ptr}  # if not part of an already predicted batch
                to_reduce = column.to_reduce.get_news()
                if not (to_predict or to_reduce):
                    break

                for nonterm in to_predict:
                    column.add( predict(nonterm, column) )
                for item in to_reduce:
                    new_items = list(complete(item))
                    for new_item in new_items:
                        if new_item.similar(item):
                            raise ParseError('Infinite recursion detected! (rule %s)' % new_item.rule)
                    column.add(new_items)

        def scan(i, token, column):
            to_scan = column.to_scan.get_news()

            for x in self.ignore:
                m = x.match(stream, i)
                if m:
                    # TODO add partial matches for ignore too?
                    delayed_matches[m.end()] += to_scan

            for item in to_scan:
                m = item.expect.match(stream, i)
                if m:
                    t = Token(item.expect.name, m.group(0), i, text_line, text_column)
                    delayed_matches[m.end()].append(item.advance(t))

                    s = m.group(0)
                    for j in range(1, len(s)):
                        m = item.expect.match(s[:-j])
                        if m:
                            delayed_matches[m.end()].append(item.advance(m.group(0)))

            next_set = Column(i+1)
            next_set.add(delayed_matches[i+1])
            del delayed_matches[i+1]    # No longer needed, so unburden memory

            return next_set

        # Main loop starts
        column0 = Column(0)
        column0.add(predict(start_symbol, column0))

        column = column0
        for i, token in enumerate(stream):
            predict_and_complete(column)
            column = scan(i, token, column)

            if token == '\n':
                text_line += 1
                text_column = 0
            else:
                text_column += 1


        predict_and_complete(column)

        # Parse ended. Now build a parse tree
        solutions = [n.tree for n in column.to_reduce
                     if n.rule.origin==start_symbol and n.start is column0]

        if not solutions:
            raise ParseError('Incomplete parse: Could not find a solution to input')
        elif len(solutions) == 1:
            tree = solutions[0]
        else:
            tree = Tree('_ambig', solutions)

        if self.resolve_ambiguity:
            tree = self.resolve_ambiguity(tree)

        return ApplyCallbacks(self.postprocess).transform(tree)