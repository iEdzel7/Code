def update_addon(apps, schema_editor):
    """Update the repo_scope flag."""
    Addon = apps.get_model("addons", "Addon")
    Event = apps.get_model("addons", "Event")
    for addon in Addon.objects.filter(name="weblate.consistency.languages"):
        Event.objects.get_or_create(addon=addon, event=EVENT_DAILY)
        addon.event_set.filter(event=EVENT_POST_UPDATE).delete()