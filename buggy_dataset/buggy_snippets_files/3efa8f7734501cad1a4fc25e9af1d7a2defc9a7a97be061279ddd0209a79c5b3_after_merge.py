    async def disconnect_timer(self):
        stop_times = {}
        pause_times = {}
        while True:
            for p in lavalink.all_players():
                server = p.channel.guild

                if [self.bot.user] == p.channel.members:
                    stop_times.setdefault(server.id, time.time())
                    pause_times.setdefault(server.id, time.time())
                else:
                    stop_times.pop(server.id, None)
                    if p.paused and server.id in pause_times:
                        try:
                            await p.pause(False)
                        except Exception:
                            log.error(
                                "Exception raised in Audio's emptypause_timer.", exc_info=True
                            )
                    pause_times.pop(server.id, None)
            servers = stop_times.copy()
            servers.update(pause_times)
            for sid in servers:
                server_obj = self.bot.get_guild(sid)
                if sid in stop_times and await self.config.guild(server_obj).emptydc_enabled():
                    emptydc_timer = await self.config.guild(server_obj).emptydc_timer()
                    if (time.time() - stop_times[sid]) >= emptydc_timer:
                        stop_times.pop(sid)
                        try:
                            player = lavalink.get_player(sid)
                            await player.stop()
                            await player.disconnect()
                        except Exception as err:
                            log.error("Exception raised in Audio's emptydc_timer.", exc_info=True)
                            if "No such player for that guild" in str(err):
                                stop_times.pop(sid, None)
                elif (
                    sid in pause_times and await self.config.guild(server_obj).emptypause_enabled()
                ):
                    emptypause_timer = await self.config.guild(server_obj).emptypause_timer()
                    if (time.time() - pause_times.get(sid)) >= emptypause_timer:
                        try:
                            await lavalink.get_player(sid).pause()
                        except Exception as err:
                            if "No such player for that guild" in str(err):
                                pause_times.pop(sid, None)
                            log.error(
                                "Exception raised in Audio's emptypause_timer.", exc_info=True
                            )
            await asyncio.sleep(5)