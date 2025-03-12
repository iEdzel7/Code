    async def userinfo(self, ctx, *, user: discord.Member = None):
        """Show information about a user.

        This includes fields for status, discord join date, server
        join date, voice state and previous names/nicknames.

        If the user has no roles, previous names or previous nicknames,
        these fields will be omitted.
        """
        author = ctx.author
        guild = ctx.guild

        if not user:
            user = author

        #  A special case for a special someone :^)
        special_date = datetime(2016, 1, 10, 6, 8, 4, 443000)
        is_special = user.id == 96130341705637888 and guild.id == 133049272517001216

        roles = user.roles[-1:0:-1]
        names, nicks = await self.get_names_and_nicks(user)

        joined_at = user.joined_at if not is_special else special_date
        since_created = (ctx.message.created_at - user.created_at).days
        if joined_at is not None:
            since_joined = (ctx.message.created_at - joined_at).days
            user_joined = joined_at.strftime("%d %b %Y %H:%M")
        else:
            since_joined = "?"
            user_joined = _("Unknown")
        user_created = user.created_at.strftime("%d %b %Y %H:%M")
        voice_state = user.voice
        member_number = (
            sorted(guild.members, key=lambda m: m.joined_at or ctx.message.created_at).index(user)
            + 1
        )

        created_on = _("{}\n({} days ago)").format(user_created, since_created)
        joined_on = _("{}\n({} days ago)").format(user_joined, since_joined)

        activity = _("Chilling in {} status").format(user.status)
        if user.activity is None:  # Default status
            pass
        elif user.activity.type == discord.ActivityType.playing:
            activity = _("Playing {}").format(user.activity.name)
        elif user.activity.type == discord.ActivityType.streaming:
            activity = _("Streaming [{}]({})").format(user.activity.name, user.activity.url)
        elif user.activity.type == discord.ActivityType.listening:
            activity = _("Listening to {}").format(user.activity.name)
        elif user.activity.type == discord.ActivityType.watching:
            activity = _("Watching {}").format(user.activity.name)

        if roles:
            roles = ", ".join([x.mention for x in roles])
        else:
            roles = None

        data = discord.Embed(description=activity, colour=user.colour)
        data.add_field(name=_("Joined Discord on"), value=created_on)
        data.add_field(name=_("Joined this server on"), value=joined_on)
        if roles is not None:
            data.add_field(name=_("Roles"), value=roles, inline=False)
        if names:
            # May need sanitizing later, but mentions do not ping in embeds currently
            val = filter_invites(", ".join(names))
            data.add_field(name=_("Previous Names"), value=val, inline=False)
        if nicks:
            # May need sanitizing later, but mentions do not ping in embeds currently
            val = filter_invites(", ".join(nicks))
            data.add_field(name=_("Previous Nicknames"), value=val, inline=False)
        if voice_state and voice_state.channel:
            data.add_field(
                name=_("Current voice channel"),
                value="{0.mention} ID: {0.id}".format(voice_state.channel),
                inline=False,
            )
        data.set_footer(text=_("Member #{} | User ID: {}").format(member_number, user.id))

        name = str(user)
        name = " ~ ".join((name, user.nick)) if user.nick else name
        name = filter_invites(name)

        if user.avatar:
            avatar = user.avatar_url_as(static_format="png")
            data.set_author(name=name, url=avatar)
            data.set_thumbnail(url=avatar)
        else:
            data.set_author(name=name)

        await ctx.send(embed=data)