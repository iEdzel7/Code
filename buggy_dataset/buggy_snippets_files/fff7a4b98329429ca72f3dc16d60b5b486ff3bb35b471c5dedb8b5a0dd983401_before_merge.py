def upgrade():
    try:
        op.create_table('tokenowner',
        sa.Column('id', sa.Integer()),
        sa.Column('token_id', sa.Integer(), nullable=True),
        sa.Column('resolver', sa.Unicode(length=120), nullable=True),
        sa.Column('user_id', sa.Unicode(length=320), nullable=True),
        sa.Column('realm_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['realm_id'], ['realm.id'], ),
        sa.ForeignKeyConstraint(['token_id'], ['token.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_tokenowner_resolver'), 'tokenowner', ['resolver'], unique=False)
        op.create_index(op.f('ix_tokenowner_user_id'), 'tokenowner', ['user_id'], unique=False)
    except Exception as exx:
        print("Can not create table 'tokenowner'. It probably already exists")
        print (exx)

    try:
        bind = op.get_bind()
        session = orm.Session(bind=bind)
        # For each token, that has an owner, create a tokenowner entry
        for token in session.query(Token).filter(Token.user_id):
            token_realms = TokenRealm.query.filter(TokenRealm.token_id == token.id).all()
            realm_id = None
            if not token_realms:
                sys.stderr.write(u"{serial!s}, {userid!s}, {resolver!s}, "
                                 u"Error while migrating token assignment. "
                                 u"This token has no realm assignments!".format(serial=token.serial,
                                                                                userid=token.user_id,
                                                                                resolver=token.resolver))
            elif len(token_realms) == 1:
                realm_id = token_realms[0].realm_id
            elif len(token_realms) > 1:
                # If the resolver is only contained in one realm, we fetch the realms:
                reso_realms = ResolverRealm.query.filter(ResolverRealm.resolver == token.resolver).all()
                if not reso_realms:
                    sys.stderr.write(u"{serial!s}, {userid!s}, {resolver!s}, "
                                     u"The token is assigned, but the assigned resolver is not "
                                     u"contained in any realm!".format(serial=token.serial,
                                                                       userid=token.user_id,
                                                                       resolver=token.resolver))
                elif len(reso_realms) == 1:
                    # The resolver is only in one realm, so this is the new realm of the token!
                    realm_id = reso_realms[0].realm_id
                elif len(reso_realms) > 1:
                    # The resolver is contained in two realms, we have to apply more logic between the realms in which
                    # the resolver is contained and the realms, to which the token is assigend.
                    found_realm_ids = []
                    for token_realm in token_realms:
                        if token_realm.realm_id in [r.realm_id for r in reso_realms]:
                            # The token realm, that also fits the resolver_realm is used as owner realm
                            found_realm_ids.append(realm_id)
                        if len(found_realm_ids) > 1:
                            sys.stderr.write(u"{serial!s}, {userid!s}, {resolver!s}, "
                                             u"Your realm configuration for the token is not distinct!. "
                                             u"The tokenowner could be in multiple realms! "                                  
                                             u"The token is assigned to the following realms and the resolver is also "
                                             u"contained in these realm IDs: {realms!s}.".format(serial=token.serial,
                                                                                                 userid=token.user_id,
                                                                                                 resolver=token.resolver,
                                                                                                 realms=found_realm_ids))
                        elif len(found_realm_ids) == 1:
                            realm_id = found_realm_ids[0]
                        else:
                            sys.stderr.write(u"{serial!s}, {userid!s}, {resolver!s}, "
                                             u"Can not assign token. The resolver is not contained in any "
                                             u"realms, to which the token is assigned!")

            to = TokenOwner(token_id=token.id, user_id=token.user_id,
                            resolver=token.resolver, realm_id=realm_id)
            session.add(to)
        session.commit()

        # Now we drop the columns
        op.drop_column('token', 'user_id')
        op.drop_column('token', 'resolver')
        op.drop_column('token', 'resolver_type')

    except Exception as exx:
        session.rollback()
        print("Failed to migrate token assignment data!")
        print (exx)