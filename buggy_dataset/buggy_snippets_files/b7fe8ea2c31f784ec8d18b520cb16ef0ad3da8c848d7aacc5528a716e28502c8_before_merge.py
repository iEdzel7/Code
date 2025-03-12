def generate_youtube_url(raw_song, meta_tags, tries_remaining=5):
    """ Search for the song on YouTube and generate a URL to its video. """
    # prevents an infinite loop but allows for a few retries
    if tries_remaining == 0:
        log.debug('No tries left. I quit.')
        return

    query = { 'part'       : 'snippet',
              'maxResults' :  50,
              'type'       : 'video' }

    if const.args.music_videos_only:
        query['videoCategoryId'] = '10'

    if not meta_tags:
        song = raw_song
        query['q'] = song
    else:
        song = '{0} - {1}'.format(meta_tags['artists'][0]['name'],
                                  meta_tags['name'])
        query['q'] = song
    log.debug('query: {0}'.format(query))

    data = pafy.call_gdata('search', query)
    data['items'] = list(filter(lambda x: x['id'].get('videoId') is not None,
                                data['items']))
    query_results = {'part': 'contentDetails,snippet,statistics',
              'maxResults': 50,
              'id': ','.join(i['id']['videoId'] for i in data['items'])}
    log.debug('query_results: {0}'.format(query_results))

    vdata = pafy.call_gdata('videos', query_results)

    videos = []
    for x in vdata['items']:
        duration_s = pafy.playlist.parseISO8591(x['contentDetails']['duration'])
        youtubedetails = {'link': x['id'], 'title': x['snippet']['title'],
                          'videotime':internals.videotime_from_seconds(duration_s),
                          'seconds': duration_s}
        videos.append(youtubedetails)
        if not meta_tags:
            break

    if not videos:
        return None

    if const.args.manual:
        log.info(song)
        log.info('0. Skip downloading this song.\n')
        # fetch all video links on first page on YouTube
        for i, v in enumerate(videos):
            log.info(u'{0}. {1} {2} {3}'.format(i+1, v['title'], v['videotime'],
                  "http://youtube.com/watch?v="+v['link']))
        # let user select the song to download
        result = internals.input_link(videos)
        if not result:
            return None
    else:
        if not meta_tags:
            # if the metadata could not be acquired, take the first result
            # from Youtube because the proper song length is unknown
            result = videos[0]
            log.debug('Since no metadata found on Spotify, going with the first result')
        else:
            # filter out videos that do not have a similar length to the Spotify song
            duration_tolerance = 10
            max_duration_tolerance = 20
            possible_videos_by_duration = list()

            '''
            start with a reasonable duration_tolerance, and increment duration_tolerance
            until one of the Youtube results falls within the correct duration or
            the duration_tolerance has reached the max_duration_tolerance
            '''
            while len(possible_videos_by_duration) == 0:
                possible_videos_by_duration = list(filter(lambda x: abs(x['seconds'] - meta_tags['duration']) <= duration_tolerance, videos))
                duration_tolerance += 1
                if duration_tolerance > max_duration_tolerance:
                    log.error("{0} by {1} was not found.\n".format(meta_tags['name'], meta_tags['artists'][0]['name']))
                    return None

            result = possible_videos_by_duration[0]

    if result:
        url = "http://youtube.com/watch?v=" + result['link']
    else:
        url = None

    return url